"""Two-user native-retrieval Mega cost pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from run_memory_pairing import ROOT, TaskModel, backend_for, encoder, nex_tokenizer, token_cost, sha
from digital_twin_methods import History, create
from timed_nex import request_model

sys.path.insert(0,str(ROOT/'isolated_abcd'))
from scorer import retrieve

MANIFEST=ROOT/'isolated_abcd/mega_cost_pilot_2x60_v2.json'
METHODS=('empty','direct','bm25','gpg','mem0','omem','pgmem')
APPEND_LOCK=Lock()


def append(path,obj):
    with APPEND_LOCK,path.open('a',encoding='utf8') as f:f.write(json.dumps(obj,ensure_ascii=False)+'\n')


class CostModel(TaskModel):
    def __init__(self,live):
        super().__init__(answer_limit=None,answer_effort='none')
        self.live=live

    def __call__(self,messages,temperature=0,**kwargs):
        with self.budget.lock:
            start=len(self.calls); tok=nex_tokenizer()
            reference=sum(len(tok.encode(str(m.get('content','')),add_special_tokens=False).ids) for m in messages)
            event={'stage':self.stage,'call_index':start,'reference_input_tokens':reference,'time':time.time()}
            append(self.live,dict(event,event='start'))
            try:return super().__call__(messages,temperature=temperature,**kwargs)
            finally:
                if len(self.calls)>start:
                    self.calls[-1]['reference_input_tokens']=reference
                    append(self.live,dict(self.calls[-1],event='end',call_index=start))


def input_text(h):return '\n'.join(f'{k.split("|",1)[-1]}\nAnswer: {v}' for k,v in h.items())


def trial(out,user,method,attempt,workers):
    h=dict(user['history']); assert len(h)==60
    folder=out/method/user['user_id']/f'attempt-{attempt}';folder.mkdir(parents=True,exist_ok=False)
    model=CostModel(folder/'calls.jsonl'); cost={}; memory=None;start=time.monotonic(); row={'id':user['id'],'user_id':user['user_id'],'method':method,'attempt':attempt,'history_count':60,'model':request_model('none'),'workers':workers,'status':'running'}
    stage='construct'; stage_start=time.monotonic()
    def advance(name):
        nonlocal stage,stage_start
        now=time.monotonic();cost[stage]=now-stage_start;stage=name;stage_start=now
    query=user['survey_text']; history=[History(f'{k.split("|",1)[-1]}\nAnswer: {v}',query=k.split('|',1)[-1],chosen=v) for k,v in h.items()]
    try:
        model.stage='memory_write'
        if method=='gpg':memory=create('gpg',task='mega',chat=model);memory.update(history)
        elif method in ('mem0','omem','pgmem'):memory=backend_for(method,'mega',model,folder/'memory');memory.update(history)
        advance('retrieve');model.stage='memory_read'
        if method=='empty':evidence='[Empty Persona Profile]'
        elif method=='direct':evidence=input_text(h)
        elif method=='bm25':evidence=input_text(dict(retrieve(h,query,'bm25')))
        else:evidence=memory.retrieve(query)
        if not isinstance(evidence,str) or not evidence.strip():raise ValueError('Empty retrieved evidence')
        (folder/'evidence.txt').write_text(evidence,encoding='utf8')
        row['evidence_reference_tokens']=len(nex_tokenizer().encode(evidence,add_special_tokens=False).ids)
        advance('answer');model.stage='answer'
        prompt=user['prompt_template'].replace('__FROZEN_PERSONA__',evidence,1)
        row['answer_prompt_sha256']=hashlib.sha256(prompt.encode()).hexdigest()
        value=model([{'role':'user','content':prompt}],temperature=0).strip()
        row['answer']=value;advance('validation')
        validation=subprocess.run([str(ROOT/'.mega-venv/Scripts/python.exe'),'-B','-c',"import json,sys;from mega import parse_prediction;x=json.load(sys.stdin);print(json.dumps(parse_prediction(x['answer'],x['survey'],{}),ensure_ascii=True))"],cwd=ROOT/'scripts',input=json.dumps({'answer':value,'survey':query},ensure_ascii=True),text=True,capture_output=True,check=True)
        row['prediction']=json.loads(validation.stdout);row['status']='complete'
    except Exception as exc:
        row.update(status='failed',failure_stage=stage,error_type=type(exc).__name__,error=str(exc)[:300])
    finally:
        advance('finished')
        if memory is not None and hasattr(memory,'close'):
            try:memory.close()
            except Exception as exc:row['close_error']=type(exc).__name__
    row.update(stage_seconds=cost,seconds=time.monotonic()-start,calls=model.calls)
    row['stage_costs']={s:token_cost([x for x in model.calls if x['stage']==s]) for s in ('memory_write','memory_read','answer')}
    (folder/'result.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
    append(out/'results.jsonl',row)
    print(method,user['id'],row['status'],'seconds',round(row['seconds'],2),flush=True)
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path);p.add_argument('--check',action='store_true');p.add_argument('--workers',type=int,default=1);p.add_argument('--manifest',type=Path,default=MANIFEST);a=p.parse_args()
    if not 1<=a.workers<=10:p.error('--workers must be 1–10')
    manifest=json.loads(a.manifest.read_text(encoding='utf8')); users=manifest['users']
    assert len(users) in (2,20,100) and len({u['id'] for u in users})==len(users) and all(len(u['history'])==60 and u['prompt_template'].count('__FROZEN_PERSONA__')==1 for u in users)
    if a.check:
        for u in users:
            ev=input_text(dict(u['history']));prompt=u['prompt_template'].replace('__FROZEN_PERSONA__',ev,1)
            print('history',len(u['history']),'direct prompt reference tokens',len(nex_tokenizer().encode(prompt,add_special_tokens=False).ids))
        return
    if request_model('none')!='Nex-N2.6-W8A8':raise ValueError('This pilot requires Nex 2.6 none')
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    cfg={'model':request_model('none'),'effort':'none','temperature':0,'history_count':60,'retrieval_budget':None,'answer_cap':None,'manifest_sha256':sha(a.manifest),'methods':list(METHODS),'bm25_top_k':8,'workers':a.workers}
    config=out/'config.json'
    if config.exists():
        old_config=json.loads(config.read_text(encoding='utf8'))
        if {k:v for k,v in old_config.items() if k!='workers'}!={k:v for k,v in cfg.items() if k!='workers'}:raise ValueError('Protocol differs')
    if not config.exists():config.write_text(json.dumps(cfg,indent=2),encoding='utf8')
    append(out/'execution_sessions.jsonl',{'time':time.time(),'workers':a.workers})
    log=out/'results.jsonl'; prior={}
    if log.exists():
        for line in log.read_text(encoding='utf8').splitlines():
            r=json.loads(line);prior[r['method'],r['id']]=r
    # Transformers initializes weights in process-global contexts. Finish loading
    # both shared encoders before worker threads begin to construct memory.
    if a.workers>1:
        started=time.monotonic()
        encoder('pgmem');encoder('mem0')
        append(out/'execution_sessions.jsonl',{'time':time.time(),'workers':a.workers,'shared_encoder_initialization_seconds':time.monotonic()-started})
    for _ in range(2):
        pending=[]
        for method in METHODS:
            for user in users:
                old=prior.get((method,user['id']))
                if old and old['status']=='complete':continue
                base=out/method/user['user_id']; attempts=[int(p.name.split('-')[-1]) for p in base.glob('attempt-*') if p.is_dir()]
                pending.append((method,user,max(attempts,default=0)+1))
        if not pending:break
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            jobs={pool.submit(trial,out,u,m,n,a.workers):(m,u['id']) for m,u,n in pending}
            for future in as_completed(jobs):prior[jobs[future]]=future.result()
    summary={m:[{k:r[k] for k in ('id','status','stage_seconds','stage_costs','seconds')} for (method,_),r in prior.items() if method==m] for m in METHODS}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print('PILOT_FINISHED',sum(r['status']=='complete' for r in prior.values()),'of',len(METHODS)*len(users),flush=True)


if __name__=='__main__':main()
