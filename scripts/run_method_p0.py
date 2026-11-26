"""M01/M02 Twin and M03 FingerTip paired memory experiments."""
import argparse
import json
import time
import uuid
from collections import defaultdict
from pathlib import Path

from run_memory_pairing import (ROOT, TaskModel, backend_for, cases, SUBSET, nex_tokenizer,
                               pack_omem, finger_template, screenshot_prompt, CONCISE_REASONING_NOTE,
                               score, token_cost, sha)
from digital_twin_methods import History
from digital_twin_methods.bench_inputs import fingertip_history
from fingertip import rows as finger_rows
from timed_nex import request_model

METHODS=('pgmem','omem','mem0')
TWIN=ROOT/'isolated_abcd/twin_method_test72_v1.json'


def append(path,row):
    with path.open('a',encoding='utf8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')


def memory_text(method,text):
    tok=nex_tokenizer()
    if method=='omem':return pack_omem(text,tok)
    return tok.decode(tok.encode(text,add_special_tokens=False).ids[:512],skip_special_tokens=True)


def answer(model,prompt):
    model.stage='answer'
    return model([{'role':'user','content':prompt}],temperature=0).strip()


def build(out,method,bench,history,model):
    model.stage='memory_write'
    memory=backend_for(method,bench,model,out/'memory'/uuid.uuid4().hex)
    try:memory.update(history)
    except BaseException:
        if hasattr(memory,'close'):memory.close()
        raise
    return memory


def new_model():return TaskModel(answer_limit=None,answer_effort='none')


def judge_none(task,record):
    from score_fingertip import judge_prompt,digest
    model=new_model(); prompt=judge_prompt(task['intentDescription'],record['prediction'])
    value=answer(model,prompt); valid=value in ('0','1')
    return {'id':record['id'],'prompt_sha256':digest(prompt.encode()),'status':'complete' if valid else 'failed','label':int(value) if valid else None,'raw':value,'calls':model.calls,'judge_model':request_model('none')}


def twin_groups():
    source=json.loads(TWIN.read_text(encoding='utf8'));return list(source['users'].items())


def twin_prompt(q,evidence):
    choices='\n'.join(f'{i+1}. {v}' for i,v in enumerate(q['options']))
    return f'Predict the participant’s answer using only the supplied historical evidence. Return exactly the number of one option.\nHistorical evidence:\n{evidence}\nQuestion: {q["text"]}\n{choices}'


def run_twin(out,phase):
    import numpy as np
    groups=twin_groups(); selected=groups[:2] if phase=='M01' else groups
    log=out/'predictions.jsonl'; records={}
    if log.exists():
        for line in log.read_text(encoding='utf8').splitlines():
            r=json.loads(line)
            if r['status']=='complete':records[r['method'],r['id']]=r
    for u,g in selected:
        h=[History(f'{k}: {v}',query=k,chosen=v) for k,v in g['history'].items()]
        targets=g['targets']
        for method in ('empty','full','bm25',*METHODS):
            pending=[r for r in targets if (method,r['id']) not in records]
            if not pending:continue
            memory=None; model=new_model(); start=time.monotonic()
            try:
                if method in METHODS:memory=build(out,method,'twin',h,model)
                prep_seconds=time.monotonic()-start; append(out/'builds.jsonl',{'user_id':u,'method':method,'history_count':len(h),'seconds':prep_seconds,'calls':model.calls,'status':'complete'})
                for row in pending:
                    q=row['question']; call_start=len(model.calls); model.stage='memory_read'; t=time.monotonic()
                    if memory is not None:ev=memory_text(method,memory.retrieve(q['text']+' '+str(q['options'])))
                    elif method=='full':ev='\n'.join(x.text for x in h)
                    elif method=='bm25':
                        # Reuse the frozen token BM25 reader used in A–C.
                        from scorer import retrieve
                        ev='\n'.join(f'{k}: {v}' for k,v in retrieve(g['history'],q['text']+' '+str(q['options']),'bm25'))
                    else:ev=''
                    read_seconds=time.monotonic()-t; t=time.monotonic(); value=answer(model,twin_prompt(q,ev));
                    valid=value.isdecimal() and 1<=int(value)<=len(q['options'])
                    r={'id':row['id'],'user_id':u,'method':method,'status':'complete' if valid else 'invalid_answer','answer':value,'prediction':int(value)-1 if valid else None,'early':row['early'],'late':row['late'],'options':q['options'],'evidence':ev,'read_seconds':read_seconds,'answer_seconds':time.monotonic()-t,'calls':model.calls[call_start:]}
                    append(log,r)
                    if valid:records[method,row['id']]=r
            except Exception as exc:append(out/'errors.jsonl',{'user_id':u,'method':method,'stage':model.stage,'type':type(exc).__name__,'error':str(exc)[:350]})
            finally:
                if memory is not None and hasattr(memory,'close'):memory.close()
    summary={}
    for method in ('empty','full','bm25',*METHODS):
        xs=[records[method,r['id']] for u,g in selected for r in g['targets'] if (method,r['id']) in records]
        summary[method]={'valid':len(xs),'expected':sum(len(g['targets']) for _,g in selected),'early_accuracy':sum(r['prediction']==r['early'] for r in xs)/len(xs) if xs else None,'late_accuracy':sum(r['prediction']==r['late'] for r in xs)/len(xs) if xs else None}
        for label in ('early','late'):
            by=defaultdict(list)
            for r in xs:
                ref=records.get(('bm25',r['id']))
                if ref:by[r['user_id']].append(int(r['prediction']==r[label])-int(ref['prediction']==ref[label]))
            if by:
                d=np.array([np.mean(v) for v in by.values()])*100; rng=np.random.default_rng(42); ci=np.percentile(d[rng.integers(len(d),size=(2000,len(d)))].mean(1),[2.5,97.5])
                summary[method][label+'_vs_bm25']={'users':len(d),'delta_pp':float(d.mean()),'ci95':ci.tolist()}
    (out/f'{phase}-summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    return all(v['valid']==v['expected'] for v in summary.values())


def finger_prompt(c,history,extra=''):
    previous=[r.text for r in history]
    if extra:previous.append('补充用户记忆：'+extra)
    text=screenshot_prompt(finger_template()(c['profile'],c['time'],c['scenario'],str(previous)),0)
    return text.replace('\n\n## Input','\n'+CONCISE_REASONING_NOTE+'\n\n## Input',1)


def finger_groups():
    past=finger_rows('total.csv')
    for c in cases('fingertip',SUBSET):
        full=fingertip_history(past,c['user_id'],c['time'],None)
        assert len(full)>=20 and all(h.timestamp<c['time'] for h in full)
        yield c,full


def run_finger(out):
    log=out/'predictions.jsonl'; records={}
    cached={}
    if (out/'builds.jsonl').exists():
        for line in (out/'builds.jsonl').read_text(encoding='utf8').splitlines():
            r=json.loads(line);cached[r['condition'],r['id']]=r['retrieved']
    if log.exists():
        for line in log.read_text(encoding='utf8').splitlines():
            r=json.loads(line)
            if r['status']=='complete':records[r['condition'],r['id']]=r
    expected=[]
    for c,full in finger_groups():
        groups=[('raw',None,None)] + [(f'{m}-{scope}',m,h) for m in METHODS for scope,h in [('20',c['history']),('all',full)]]
        for prefix,method,h in groups:
            conditions=['raw'] if method is None else [prefix+'-only',prefix+'-plus'];expected.extend((k,c['id']) for k in conditions)
            if all((k,c['id']) in records for k in conditions):continue
            memory=None; model=new_model(); start=time.monotonic()
            try:
                if (prefix,c['id']) in cached:ev=cached[prefix,c['id']]
                elif method:
                    memory=build(out,method,'fingertip',h,model)
                    if method=='pgmem':memory.set_prediction_time(c['time'])
                    model.stage='memory_read';ev=memory_text(method,memory.retrieve(c['query']))
                else:ev=''
                if (prefix,c['id']) not in cached:
                    prep=time.monotonic()-start;append(out/'builds.jsonl',{'id':c['id'],'condition':prefix,'history_count':len(h or []),'seconds':prep,'calls':model.calls,'retrieved':ev});cached[prefix,c['id']]=ev
                for condition in conditions:
                    if (condition,c['id']) in records:continue
                    t=time.monotonic(); n=len(model.calls); val=answer(model,finger_prompt(c,[] if condition.endswith('-only') else c['history'],ev));elapsed=time.monotonic()-t
                    r={'id':c['id'],'user_id':c['user_id'],'condition':condition,'method':method or 'raw','status':'complete','answer':val,'prediction':val,'calls':model.calls[n:],'generated_tokens':token_cost(model.calls[n:])['completion_tokens'],'stage_seconds':{'answer':elapsed},'seconds':elapsed,'retrieved_memory':ev,'memory_empty':not ev,'row_index':c['row_index'],'legacy_id':c['legacy_id'],'screenshot_count':0,'preparation_cost':dict(token_cost([]),wall_seconds=0),'answer_cost':dict(token_cost(model.calls[n:]),wall_seconds=elapsed)}
                    append(log,r);records[condition,c['id']]=r
            except Exception as exc:append(out/'errors.jsonl',{'id':c['id'],'condition':prefix,'stage':model.stage,'type':type(exc).__name__,'error':str(exc)[:350]})
            finally:
                if memory is not None and hasattr(memory,'close'):memory.close()
    if not all(k in records for k in expected):return False
    for condition in sorted({k for k,_ in expected}):
        directory=out/condition;directory.mkdir(exist_ok=True);xs=[r for (k,_),r in records.items() if k==condition]
        if len(xs)==20:score('fingertip',xs,directory,20,2,repeat_failures=True,judge_classifier=judge_none,judge_name=request_model('none')+' none substitution; no additional token cap')
    return all(k in records for k in expected)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['check','M01','M02','M03']);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.phase=='check':
        users=twin_groups(); assert len(users)==12 and sum(len(g['targets']) for _,g in users)==72
        for _,g in users:
            assert g['history'] and all(twin_prompt(r['question'],'test') for r in g['targets'])
        fs=list(finger_groups());assert len(fs)==20;assert all(finger_prompt(c,full[-20:]) for c,full in fs)
        print('P0 offline checks: Twin 12 users/72 tracks, FingerTip 20 tracks, strict timestamps',flush=True);return
    out=args.output;out.mkdir(parents=True,exist_ok=True)
    config=out/'config.json'
    protocol={'model':request_model('none'),'answer_effort':'none','memory_budget':512,'answer_cap':None,'judge_model':request_model('none'),'judge_effort':'none','twin_manifest_sha256':sha(TWIN),'finger_manifest_sha256':sha(SUBSET),'methods':list(METHODS)}
    if config.exists():
        if json.loads(config.read_text(encoding='utf8'))!=protocol:raise ValueError('Existing experiment protocol differs; use a new output directory')
    else:config.write_text(json.dumps(protocol,indent=2),encoding='utf8')
    while True:
        if (run_finger(out) if args.phase=='M03' else run_twin(out,args.phase)):
            print(args.phase,'COMPLETE',flush=True);return
        print(args.phase,'retry remaining tracks after 30 seconds',flush=True)
        time.sleep(30)


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT/'isolated_abcd'))
    main()
