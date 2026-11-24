"""Shared, resumable initial memory-method comparisons on frozen subsets.

The runner is not invoked on import. --check-inputs never makes model calls.
"""
import argparse
import os
import hashlib
import json
import statistics
import sys
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from threading import RLock, local

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
from digital_twin_methods import create, History
from digital_twin_methods.bench_inputs import fingertip_history
from run_lamp_gpg512 import retrieve as lamp_retrieve, pack as lamp_pack, load_source
from official_lamp_inputs import build as lamp_build
from fingertip import prompts as fingertip_prompts, rows as finger_rows, screenshot_prompt, ZERO_SCREENSHOT_NOTE
from fingertip_data import tasks as fingertip_tasks
from nex_vita_transport import chat, task_budget, TaskBudget, NexDiagnosticStop, TASK_TOKEN_BUDGET, PREPARATION_EFFORT, PRIMARY_EFFORT, token_cost
from timed_nex import request_model
from track_retry import retryable, connection_failure

SUBSET = ROOT/'data/memory_eval_subset_v1.json'
PAIRS = {'lamp2': ('mem0','pgmem','gpg','raw','no_history'),
         'fingertip': ('mem0','pgmem','omem','gpg','raw','no_history')}
FINGER_AUGMENT_TOKENS = 512
CONCISE_REASONING_NOTE = 'Reason briefly from the available evidence, select the most plausible intent, and stop once the answer is determined. Avoid repeating the same reasoning or reconsidering alternatives without new evidence.'
NEX_TOKENIZER = Path(os.getenv('NEX_TOKENIZER_PATH','D:/gcc_temp/nex-n2.5-pro-official-tokenizer.json'))
LABELS = ['sci-fi','based on a book','comedy','action','twist ending','dystopia','dark comedy','classic','psychology','fantasy','romance','thought-provoking','social commentary','violence','true story']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_check(spec):
    if sha(ROOT/spec['path']) != spec['sha256']:
        raise ValueError('Frozen source mismatch: '+spec['path'])


def cases(bench, subset_path=None):
    subset=json.loads((subset_path or SUBSET).read_text(encoding='utf-8'))
    parent=ROOT/subset['parent_manifest']
    if sha(parent)!=subset['parent_sha256']:
        raise ValueError('Parent evaluation manifest changed')
    manifest=json.loads(parent.read_text(encoding='utf-8'))
    ids=subset[bench]['ids']
    assert ids and len(ids)==len(set(ids))
    if bench=='lamp2':
        spec=manifest['lamp']['branches']['2']
        assert set(ids)<=set(spec['ids'])
        questions={str(q['id']):q for q in load_source(spec['source'])}
        golds={str(q['id']):q['output'] for q in load_source(spec['gold_source'])['golds']}
        assert len({questions[i]['user_id'] for i in ids})==len(ids)
        for i in ids:
            q=questions[i]; selected=lamp_retrieve(2,q)
            ordered=sorted(selected,key=lambda h:str(h.get('date','')))
            history=[History(f"description: {h['description']}\ntag: {h['tag']}",timestamp=str(h.get('date','')),query=h['description'],chosen=h['tag']) for h in ordered]
            yield {'id':i,'user_id':str(q['user_id']),'question':q,'selected':selected,'history':history,
                   'history_ids':[str(h['id']) for h in ordered],'query':q['input'],'gold':golds[i]}
    else:
        spec=manifest['fingertip']
        assert set(ids)<=set(spec['ids'])
        for field in ('source','history_source','profile_source'): source_check(spec[field])
        profiles={r['user_id']:r for r in finger_rows('user_profile.csv')}
        past=finger_rows('total.csv')
        task_map={r['id']:r for r in fingertip_tasks()}
        if not subset[bench].get('allow_repeated_users',False):
            assert len({task_map[i]['user_id'] for i in ids})==len(ids)
        for i in ids:
            t=task_map[i]; u=profiles[t['user_id']]
            profile=f"{u['sex']}，{u['age']}岁，职业为{u['occupation']}，现居{u['address']}，{u['family']}，使用{u['phone_brand']}手机"
            history=fingertip_history(past,t['user_id'],t['time'],20)
            assert len(history)==20 and all(h.timestamp<t['time'] for h in history)
            yield {'id':i,'user_id':t['user_id'],'row_index':t['row_index'],'legacy_id':t['legacy_id'],
                   'profile':profile,'time':t['time'],'scenario':t['scenario'],'history':history,
                   'history_ids':[f'{t["user_id"]}/{h.timestamp}' for h in history],
                   'query':f"时间：{t['time']}，地点：{t['scenario']}，预测该用户接下来最可能的手机操作意图。"}


@lru_cache(maxsize=1)
def finger_template():
    return fingertip_prompts()


@lru_cache(maxsize=1)
def nex_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(NEX_TOKENIZER))


def pack_omem(memory, tok, budget=FINGER_AUGMENT_TOKENS):
    """Serialize a prefix of whole native retrieval entries, in native channel order."""
    source = json.loads(memory)
    if not isinstance(source,dict) or any(not isinstance(items,list) for items in source.values()):
        raise ValueError('Invalid O-Mem retrieval structure')
    packed = {}
    for key, items in source.items():
        packed[key] = []
        for item in items:
            packed[key].append(item)
            if len(tok.encode(json.dumps(packed,ensure_ascii=False),add_special_tokens=False).ids)>budget:
                packed[key].pop()
                if not packed[key]:packed.pop(key)
                return json.dumps(packed,ensure_ascii=False)
        if len(tok.encode(json.dumps(packed,ensure_ascii=False),add_special_tokens=False).ids)>budget:
            packed.pop(key);break
    return json.dumps(packed,ensure_ascii=False)


def adapter_revision(bench, method):
    if bench=='fingertip' and method=='pgmem':return 'split-budget-prep-none-v4-pgmem-structured-v1-fingertip-clock-v2'
    if bench=='fingertip' and method=='omem':return 'split-budget-prep-none-v4-omem-whole-items-v2'
    return 'split-budget-prep-none-v4-pgmem-structured-v1' if method=='pgmem' else 'split-budget-prep-none-v4'


def prompt_for(bench, case, method, memory='', digest='', concise_reasoning=False):
    if bench=='lamp2':
        if method in ('raw','no_history'):
            prompt,n=lamp_build(2,case['question'],'bm25' if method=='raw' else None,4 if method=='raw' else 0)
            return prompt,{'input_tokens_reference':n}
        return lamp_pack(2,case['question'],case['selected'],memory,digest,memory_label='Personal profile' if method=='gpg' else 'Retrieved memory')
    previous=[] if method=='no_history' else [h.text for h in case['history']]
    if method not in ('raw','no_history') and memory.strip():
        tok=nex_tokenizer()
        if method=='omem':
            packed=pack_omem(memory,tok)
        else:
            ids=tok.encode(memory,add_special_tokens=False).ids
            packed=tok.decode(ids[:FINGER_AUGMENT_TOKENS],skip_special_tokens=True)
        previous.append('补充用户记忆：'+packed)
    text=screenshot_prompt(finger_template()(case['profile'],case['time'],case['scenario'],str(previous)),0)
    if concise_reasoning:
        text=text.replace('\n\n## Input','\n'+CONCISE_REASONING_NOTE+'\n\n## Input',1)
    return text,{'input_tokens_nex':len(nex_tokenizer().encode(text,add_special_tokens=False).ids)}


class TaskModel:
    """Bind the same ledger even when upstream code calls from a worker thread."""
    def __init__(self, request=chat, progress=None, answer_limit=TASK_TOKEN_BUDGET, answer_effort=PRIMARY_EFFORT):
        self.progress=progress
        self.preparation_budget=TaskBudget(limit=None,effort=PREPARATION_EFFORT,on_progress=self.report)
        self.answer_budget=TaskBudget(limit=answer_limit,effort=answer_effort,on_progress=self.report)
        self.request=request; self.calls=[]; self.stage='setup'
    def report(self,detail=None):
        if self.progress:
            self.progress({'stage':self.stage,'call_index':len(self.calls),'reasoning_effort':self.budget.effort,**(detail or {})})

    @property
    def budget(self):
        return self.answer_budget if self.stage=='answer' else self.preparation_budget

    def __call__(self,messages,temperature=0,**kwargs):
        with self.budget.lock:
            if self.budget.failure is not None:
                raise self.budget.failure
            row={'stage':self.stage,'reasoning_effort':self.budget.effort,'remaining_before':self.budget.remaining,'prompt_sha256':hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()}
            self.calls.append(row)
            self.report({'status':'request_started'})
            t0=time.monotonic()
            try:
                with task_budget(self.budget):
                    response=self.request(messages,temperature=temperature,**kwargs)
                phases=response.get('nex_phases') or []
                row.update(status='complete',usage=phases[-1].get('usage') if phases else response['usage'])
                text=response['choices'][0]['message']['content']
                if not isinstance(text,str) or not text.strip():
                    raise ValueError('Empty model content')
                return text
            except NexDiagnosticStop as e:
                p=e.phase_result
                row.update(status=e.reason,usage=p.get('usage'),finish_reason=p.get('finish_reason'),
                           incomplete_reason=p.get('incomplete_reason'),observed_tokens=e.tokens,diagnostic=e.diagnostic)
                raise
            except Exception as e:
                row.update(status='client_error',error=f'{type(e).__name__}: {str(e)[:200]}')
                raise
            finally:
                row.update(seconds=time.monotonic()-t0,remaining_after=self.budget.remaining)
                self.report({'status':row.get('status','interrupted'),'elapsed_seconds':row['seconds']})


class LocalEncoder:
    def __init__(self,path):
        from sentence_transformers import SentenceTransformer
        self.model=SentenceTransformer(str(path),device='cpu',local_files_only=True)
        self.lock=RLock()
    def encode(self,text,**kwargs):
        with self.lock:
            return self.model.encode(text,**kwargs)


@lru_cache(maxsize=2)
def encoder(method):
    return LocalEncoder(ROOT/('data/models/paraphrase-multilingual-MiniLM-L12-v2' if method=='mem0' else 'digital_twin_methods/models/all-MiniLM-L6-v2'))


_nlp=local()
def backend_for(method,bench,model,storage):
    if method=='gpg': return create('gpg',task='fingertip' if bench=='fingertip' else 'lamp2',chat=model)
    if method=='pgmem':
        if not hasattr(_nlp,'nlp'):
            import spacy
            _nlp.nlp=spacy.load('en_core_web_sm')
        return create('pgmem',chat=model,embedding=encoder('pgmem'),nlp=_nlp.nlp,timestamped=bench=='fingertip')
    if method=='omem': return create('omem',storage=storage,chat=model,embedding=encoder('pgmem'))
    if method=='mem0':
        memory=create('mem0',storage=storage,chat=model)
        # Same labelled local MiniLM replacement as the loopback service, but
        # in-process to avoid networking or a second resident encoder process.
        memory.memory.embedding_model.embed=lambda text,*a,**kw:encoder('mem0').encode(text.replace('\n',' '),normalize_embeddings=True,show_progress_bar=False).tolist()
        return memory
    raise ValueError(method)


def predict(bench,method,case,out,request=chat,make_backend=backend_for,answer_limit=TASK_TOKEN_BUDGET,answer_effort=PRIMARY_EFFORT,concise_reasoning=False):
    start=time.monotonic(); memory=None; context=''; digest=''
    live=out/'progress'/f"{case['id'].replace(':','_')}-{uuid.uuid4().hex}.json"
    live.parent.mkdir(parents=True,exist_ok=True)
    def progress(detail):
        try:
            tmp=live.with_suffix('.tmp')
            tmp.write_text(json.dumps({'id':case['id'],'bench':bench,'method':method,'updated_unix':time.time(),
                                       'task_elapsed_seconds':time.monotonic()-start,**detail},ensure_ascii=False),encoding='utf-8')
            tmp.replace(live)
        except OSError:
            pass  # Monitoring must not change the evaluation outcome.
    model=TaskModel(request,progress,answer_limit,answer_effort)
    item={'id':case['id'],'user_id':case['user_id'],'method':method,'bench':bench,'status':'running',
          'answer':'','history_ids':case['history_ids'],'calls':model.calls,'stage_seconds':{}}
    stage_start=start
    def advance(stage):
        nonlocal stage_start
        now=time.monotonic(); item['stage_seconds'][model.stage]=now-stage_start
        model.stage=stage; stage_start=now
        model.report({'status':'stage_started'})
    if bench=='lamp2':item['gold']=case['gold']
    else:item.update(row_index=case['row_index'],legacy_id=case['legacy_id'],screenshot_count=0)
    try:
        if method not in ('raw','no_history'):
            advance('memory_write')
            storage=out/'memory'/case['id'].replace(':','_')/uuid.uuid4().hex
            memory=make_backend(method,bench,model,storage)
            memory.update(case['history'])
            if model.budget.failure: raise model.budget.failure
            advance('memory_read')
            if bench=='fingertip' and method=='pgmem' and hasattr(memory,'set_prediction_time'):
                memory.set_prediction_time(case['time'])
            if method=='gpg': context,digest=memory.profile,memory.digest
            else:context=memory.retrieve(case['query']) or ''
            if model.budget.failure: raise model.budget.failure
            item.update(memory_empty=not context.strip(),retrieved_memory=context)
        advance('prompt')
        prompt,meta=prompt_for(bench,case,method,context,digest,concise_reasoning)
        item.update(meta,prompt=prompt)
        advance('answer')
        item['answer']=model([{'role':'user','content':prompt}],temperature=0).strip()
        item['status']='complete'
    except Exception as e:
        failure=model.budget.failure
        if failure is not None or isinstance(e,NexDiagnosticStop):
            stop=failure or e
            item.update(status=stop.reason,error=str(stop),failure_stage=model.stage,diagnostic=stop.diagnostic)
            if stop.budget_trace is not None:
                item['budget_trace']=stop.budget_trace
        else:
            item.update(status='method_error',error=f'{type(e).__name__}: {str(e)[:250]}',failure_stage=model.stage,
                        diagnostic={'category':'method_or_input_error','phase':model.stage,'exception_type':type(e).__name__})
    finally:
        item['stage_seconds'][model.stage]=time.monotonic()-stage_start
        if memory is not None and hasattr(memory,'close'):
            try:memory.close()
            except Exception as e:item['close_error']=type(e).__name__
    prep=token_cost([c for c in model.calls if c['stage']!='answer'])
    ans=token_cost([c for c in model.calls if c['stage']=='answer'])
    prep['wall_seconds']=sum(s for stage,s in item['stage_seconds'].items() if stage!='answer')
    ans['wall_seconds']=item['stage_seconds'].get('answer',0)
    item.update(prediction=item['answer'],preparation_cost=prep,answer_cost=ans,
                generated_tokens=prep['completion_tokens']+ans['completion_tokens'],seconds=time.monotonic()-start)
    model.report({'status':item['status'],'finished':True})
    return item


def score(bench,items,out,expected,workers,retries=None,first_pass=None,repeat_failures=False):
    valid=[r for r in items if r['status']=='complete']
    summary={'expected':expected,'attempted':len(items),'valid_predictions':len(valid),'statuses':dict(Counter(r['status'] for r in items)),
             'memory_empty':sum(r.get('memory_empty',False) for r in items),'median_generated_tokens':statistics.median(r['generated_tokens'] for r in items) if items else None,
             'model_calls_by_stage':dict(Counter(c['stage'] for r in items for c in r['calls']))}
    costs={}
    for r in items:
        for stage,seconds in r.get('stage_seconds',{}).items():
            costs.setdefault(stage,{'wall_seconds_sum':0,'completion_tokens':0,'reasoning_tokens':0,'missing_usage_calls':0})['wall_seconds_sum']+=seconds
        for c in r['calls']:
            cost=costs.setdefault(c['stage'],{'wall_seconds_sum':0,'completion_tokens':0,'reasoning_tokens':0,'missing_usage_calls':0})
            usage=c.get('usage')
            if usage:
                cost['completion_tokens']+=usage.get('completion_tokens',0) or 0
                cost['reasoning_tokens']+=usage.get('reasoning_tokens',0) or 0
            else:cost['missing_usage_calls']+=1
    summary['stage_costs']=costs
    calls=[c for r in items for c in r['calls']]
    summary['preparation_cost']=token_cost([c for c in calls if c['stage']!='answer'])
    summary['answer_cost']=token_cost([c for c in calls if c['stage']=='answer'])
    for key in ('preparation_cost','answer_cost'):
        summary[key]['wall_seconds_sum']=sum(r[key]['wall_seconds'] for r in items)
    summary['failure_attribution']=dict(Counter(r['diagnostic']['category'] for r in items if r.get('diagnostic')))
    if retries is not None:
        original={r['id']:r for r in first_pass}
        summary['retry_policy']='deferred rounds until real result, non-budget failures only' if repeat_failures else 'once, non-budget failure only'
        summary['retry_attempts']=len(retries)
        summary['retry_recovered']=sum(r['status']=='complete' for r in retries)
        summary['first_pass_statuses']=dict(Counter(r['status'] for r in original.values()))
        summary['retry_statuses']=dict(Counter(r['status'] for r in retries))
        all_attempts=list(original.values())+list(retries)
        for key, predicate in [('preparation_cost',lambda c:c['stage']!='answer'),('answer_cost',lambda c:c['stage']=='answer')]:
            cost=token_cost([c for r in all_attempts for c in r['calls'] if predicate(c)])
            cost['wall_seconds_sum']=sum(r[key]['wall_seconds'] for r in all_attempts)
            summary['all_attempts_'+key]=cost
    suffix='.after_retry' if retries is not None else ''
    if bench=='lamp2':
        from sklearn.metrics import f1_score
        from retrieval_client import map_label
        mapped=[]; errors=[]
        for r in valid:
            try:mapped.append((r['gold'],map_label(r['answer'],LABELS)['label']))
            except Exception as e:errors.append({'id':r['id'],'error':type(e).__name__})
        raw=sum(r['answer'].strip()==str(r['gold']).strip() for r in valid)
        strict_f1=f1_score([str(r['gold']).strip() for r in items],[r['answer'].strip() if r['status']=='complete' and r['answer'].strip() in LABELS else '__invalid__' for r in items],labels=LABELS,average='macro',zero_division=0) if items else None
        correct=sum(a==b for a,b in mapped)
        summary.update(raw_correct=raw,raw_accuracy_valid=raw/len(valid) if valid else None,raw_accuracy_attempted=raw/len(items) if items else None,
                       official_macro_f1_attempted=strict_f1,
                       mapped_correct=correct,mapped_count=len(mapped),mapping_errors=errors,
                       accuracy_valid=correct/len(mapped) if mapped else None,accuracy_attempted=correct/len(items) if items else None,
                       macro_f1_valid=f1_score([a for a,b in mapped],[b for a,b in mapped],labels=LABELS,average='macro',zero_division=0) if mapped else None)
    else:
        from score_fingertip import compute_sim1,classify
        task_map={r['id']:r for r in fingertip_tasks()}
        sim=compute_sim1(valid,task_map) if valid else []
        (out/f'sim1-items{suffix}.json').write_text(json.dumps(sim,ensure_ascii=False,indent=2),encoding='utf-8')
        path=out/('sr1-retry-items.jsonl' if retries is not None else 'sr1-items.jsonl'); judged={}
        for source in dict.fromkeys([out/'sr1-items.jsonl',path]):
            if source.exists():
                for line in source.read_text(encoding='utf-8').splitlines():
                    r=json.loads(line); judged[r['id']]=r
        rounds=0
        while repeat_failures or rounds<2:
            pending=[r for r in valid if r['id'] not in judged or (judged[r['id']]['status']!='complete' if repeat_failures else retryable(judged[r['id']],judged[r['id']].get('attempt',1)))]
            if not pending:break
            if rounds and repeat_failures:time.sleep(30)
            rounds+=1
            with path.open('a',encoding='utf-8') as stream,ThreadPoolExecutor(workers) as pool:
                futures={pool.submit(classify,task_map[r['id']],r):r['id'] for r in pending}
                for f in as_completed(futures):
                    i=futures[f]
                    try:r=f.result()
                    except Exception as e:r={'id':i,'status':'failed','label':None,'error':type(e).__name__}
                    r['attempt']=judged[i].get('attempt',1)+1 if i in judged else 1
                    judged[i]=r;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
        labels=[judged[r['id']]['label'] for r in valid if judged.get(r['id'],{}).get('status')=='complete']
        summary.update(Sim1_valid=sum(r['Sim1'] for r in sim)/len(sim) if sim else None,
                       Sim1_attempted=sum(r['Sim1'] for r in sim)/len(items) if items else None,
                       SR1_valid=sum(labels)/len(labels) if labels else None,SR1_attempted=sum(labels)/len(items) if items else None,
                       SR1_judged=len(labels),SR1_failed=len(valid)-len(labels),judge='Nex-medium substitution, no additional token cap')
    (out/f'scores{suffix}.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SCORES',json.dumps(summary,ensure_ascii=False),flush=True)


def deferred_history(out,saved):
    latest=dict(saved); attempts=[]
    path=out/'predictions.retry.jsonl'
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line);key=row['id'];previous=latest.get(key)
            if previous is None or not retryable(previous) or row.get('attempt')!=previous.get('attempt',1)+1:
                raise ValueError(f'Invalid deferred retry record: {key}')
            latest[key]=row;attempts.append(row)
    return latest,attempts


def deferred_round(bench,method,work,out,saved,workers,predict_options):
    latest,attempts=deferred_history(out,saved)
    selected=[c for c in work if retryable(latest[c['id']])]
    if selected:
        print('DEFERRED_RETRY',method,len(selected),flush=True)
        if method in ('mem0','pgmem','omem'):encoder('mem0' if method=='mem0' else 'pgmem')
        with (out/'predictions.retry.jsonl').open('a',encoding='utf-8') as stream,ThreadPoolExecutor(workers) as pool:
            futures=[pool.submit(predict,bench,method,c,out,**predict_options) for c in selected]
            for f in as_completed(futures):
                row=f.result();key=row['id'];row.update(attempt=latest[key].get('attempt',1)+1,first_status=saved[key]['status'],execution_workers=workers)
                latest[key]=row;stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print('RETRY_RESULT',key,row['attempt'],row['status'],flush=True)
    return sum(retryable(r) for r in latest.values())


def main():
    p=argparse.ArgumentParser();p.add_argument('--bench',choices=PAIRS,required=True);p.add_argument('--method',required=True)
    p.add_argument('--output',type=Path);p.add_argument('--workers',type=int,default=10);p.add_argument('--limit',type=int)
    p.add_argument('--subset',type=Path,default=SUBSET)
    p.add_argument('--uncapped-answer',action='store_true',help='Omit the client output token cap for this separate experiment')
    p.add_argument('--direct-fingertip',action='store_true',help='Use Nex none and a concise reasoning instruction for FingerTip answers')
    p.add_argument('--retry-connections-only',action='store_true',help='After a finished first pass, retry connection/stream failures only')
    p.add_argument('--deferred-phase',choices=('first','retry','score'),help='Finish all first-pass predictions, then retry exceptional samples in rounds')
    p.add_argument('--check-inputs',action='store_true');a=p.parse_args()
    if a.method not in PAIRS[a.bench] or not 1<=a.workers<=10 or (a.limit is not None and a.limit<1):p.error('Unsupported pairing or range')
    if a.direct_fingertip and a.bench!='fingertip':p.error('--direct-fingertip requires FingerTip')
    if a.deferred_phase and (a.limit or a.retry_connections_only):p.error('Deferred phases require the full frozen subset')
    work=list(cases(a.bench,a.subset))
    if a.check_inputs:
        for c in work:
            example=json.dumps({'persona attributes':['Example retrieved memory. '*100]}) if a.method=='omem' else 'Example retrieved memory. '*100
            prompt,meta=prompt_for(a.bench,c,a.method,example,concise_reasoning=a.direct_fingertip)
            assert prompt and (a.bench!='lamp2' or meta['input_tokens_reference']<=512)
        print('Offline input check:',a.bench,a.method,len(work),'valid');return
    if a.output is None:p.error('--output required')
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    predict_options={'answer_limit':None} if a.uncapped_answer else {}
    if a.direct_fingertip:predict_options.update(answer_effort='none',concise_reasoning=True)
    config={'bench':a.bench,'method':a.method,'subset_sha256':sha(a.subset),'expected':len(work),'workers':a.workers,
            'adapter_revision':adapter_revision(a.bench,a.method),
            'track_retry_limit':1,
            'model':request_model('none' if a.direct_fingertip else PRIMARY_EFFORT),'reasoning_effort':'none' if a.direct_fingertip else PRIMARY_EFFORT,'preparation_effort':PREPARATION_EFFORT,'fallback':None,'task_token_budget':None if a.uncapped_answer else TASK_TOKEN_BUDGET,
            'budget_scope':'answer_only_uncapped' if a.uncapped_answer else 'answer_only','preparation_token_budget':None,'scoring_token_cap':None,
            'history_protocol':'official BM25 Top-4, chronological write, same ranked evidence in final prompt' if a.bench=='lamp2' else 'official zero screenshot, most recent 20 prior events',
            'input_protocol':'same GPG-512 packing, 96 reference tokens for memory' if a.bench=='lamp2' else 'official raw 20 history with up to 512 Nex tokens of retrieved memory augmentation',
            'embedding_model':('paraphrase-multilingual-MiniLM-L12-v2 (local replacement)' if a.method=='mem0' else 'all-MiniLM-L6-v2' if a.method in ('pgmem','omem') else None),
            'nex_tokenizer_sha256':sha(NEX_TOKENIZER) if a.bench=='fingertip' else None}
    if a.bench=='fingertip':
        config['zero_screenshot_note']=ZERO_SCREENSHOT_NOTE
    if a.direct_fingertip:config['concise_reasoning_prompt']=CONCISE_REASONING_NOTE
    previous_config=dict(config)
    if a.deferred_phase:
        config.update(track_retry_limit=None,retry_strategy='all first passes before deferred exception retry rounds; genuine budget failures retained')
    cfg=out/'experiment.json'
    if cfg.exists():
        existing_config=json.loads(cfg.read_text(encoding='utf-8'))
        legacy={k:v for k,v in config.items() if k!='track_retry_limit'}
        if a.deferred_phase:
            comparable=dict(existing_config); changes=comparable.pop('worker_changes',[])
            old_workers=comparable['workers'];comparable['workers']=a.workers
            if comparable not in (config,previous_config):raise ValueError('Experiment configuration mismatch')
            if old_workers!=a.workers:
                changes.append({'from':old_workers,'to':a.workers,'unix':time.time(),
                                'first_pass_rows':len((out/'predictions.jsonl').read_text(encoding='utf-8').splitlines()) if (out/'predictions.jsonl').exists() else 0})
            if changes:config['worker_changes']=changes
            if existing_config!=config:
                tmp=cfg.with_suffix('.tmp');tmp.write_text(json.dumps(config,indent=2),encoding='utf-8');tmp.replace(cfg)
        elif existing_config!=config and not (a.retry_connections_only and existing_config==legacy):
            raise ValueError('Experiment configuration mismatch')
    if not cfg.exists():cfg.write_text(json.dumps(config,indent=2),encoding='utf-8')
    data=out/'predictions.jsonl';saved={};expected={c['id'] for c in work}
    if data.exists():
        for l in data.read_text(encoding='utf-8').splitlines():
            r=json.loads(l)
            if r['id'] in saved or r['id'] not in expected:raise ValueError('Invalid saved prediction ID')
            saved[r['id']]=r
    pending=[c for c in work if c['id'] not in saved]
    if a.deferred_phase in ('retry','score') and pending:p.error('Deferred phases require a completed first pass')
    if a.limit is not None:pending=pending[:a.limit]
    if a.retry_connections_only:
        first_score=out/'scores.json'
        if pending or a.limit is not None or not first_score.exists() or json.loads(first_score.read_text(encoding='utf-8')).get('attempted')!=len(work):
            p.error('Connection-only retry requires a fully scored first pass')
    if pending and a.method in ('mem0','pgmem','omem'):encoder('mem0' if a.method=='mem0' else 'pgmem')
    print('START',a.bench,a.method,len(saved),'/',len(work),'new',len(pending),flush=True)
    with data.open('a',encoding='utf-8') as stream,ThreadPoolExecutor(a.workers) as pool:
        futures=[pool.submit(predict,a.bench,a.method,c,out,**predict_options) for c in pending]
        for f in as_completed(futures):
            r=f.result()
            if a.deferred_phase:r['execution_workers']=a.workers
            saved[r['id']]=r;stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
            print(f"[{len(saved)}/{len(work)}] {r['id']} {r['status']} tokens={r['generated_tokens']} calls={len(r['calls'])} seconds={r['seconds']:.2f}",flush=True)
    if a.deferred_phase=='first':return
    if a.deferred_phase=='retry':
        print('DEFERRED_REMAINING',deferred_round(a.bench,a.method,work,out,saved,a.workers,predict_options),flush=True);return
    if a.deferred_phase=='score':
        final,attempts=deferred_history(out,saved)
        if any(retryable(r) for r in final.values()):p.error('Deferred predictions still unresolved')
        if attempts:
            first_score=out/'scores.json'
            if not first_score.exists() or json.loads(first_score.read_text(encoding='utf-8')).get('attempted')!=len(work):
                score(a.bench,list(saved.values()),out,len(work),a.workers)
            score(a.bench,list(final.values()),out,len(work),a.workers,attempts,list(saved.values()),repeat_failures=True)
        else:score(a.bench,list(final.values()),out,len(work),a.workers,repeat_failures=True)
        return
    if not a.retry_connections_only:
        score(a.bench,list(saved.values()),out,len(work),a.workers)
    # Preserve the first pass and its score. A retry is a fresh memory instance
    # with an independent answer budget, even when resuming a stopped job.
    retry_path=out/'predictions.retry.jsonl'
    retried={}
    if retry_path.exists():
        for line in retry_path.read_text(encoding='utf-8').splitlines():
            row=json.loads(line); key=row['id']
            if key in retried or key not in saved or not retryable(saved[key]):
                raise ValueError('Invalid saved retry ID')
            retried[key]=row
    selected=[c for c in work if c['id'] in saved and retryable(saved[c['id']]) and c['id'] not in retried
              and (not a.retry_connections_only or connection_failure(saved[c['id']]))]
    if selected and a.method in ('mem0','pgmem','omem'):encoder('mem0' if a.method=='mem0' else 'pgmem')
    if selected:
        print('RETRY',a.bench,a.method,'new',len(selected),flush=True)
        with retry_path.open('a',encoding='utf-8') as stream,ThreadPoolExecutor(a.workers) as pool:
            futures=[pool.submit(predict,a.bench,a.method,c,out,**predict_options) for c in selected]
            for f in as_completed(futures):
                row=f.result();key=row['id'];row.update(attempt=2,first_status=saved[key]['status'])
                retried[key]=row;stream.write(json.dumps(row,ensure_ascii=False)+'\n');stream.flush()
                print('RETRY_RESULT',key,row['status'],flush=True)
    if retried:
        final={**saved,**retried}
        score(a.bench,list(final.values()),out,len(work),a.workers,list(retried.values()),list(saved.values()))


if __name__=='__main__': main()
