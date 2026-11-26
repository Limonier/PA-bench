"""Offline assertions for the five initial pairings; no API calls."""
import sys
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_memory_pairing as runner
from fingertip import screenshot_prompt, ZERO_SCREENSHOT_NOTE, prompts


def complete(text='ok',n=100):
    return {'status':'complete','seconds':.01,'answer':text,'finish_reason':'stop','effort':'medium',
            'usage':{'completion_tokens':n,'prompt_tokens':10,'reasoning_tokens':max(0,n-1)}}


class FakeMemory:
    def __init__(self,chat,method=None):self.chat=chat;self.method=method
    def update(self,history):
        # Accounts for Mem0 worker-thread / O-Mem async callbacks, not just caller context.
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(lambda h:self.chat([{'role':'user','content':h.text}]),history[:2]))
    def retrieve(self,query):return json.dumps({'persona attributes':['Selected memory supported by past observations.']}) if self.method=='omem' else 'Selected memory supported by past observations.'


def main():
    token=runner.nex_tokenizer()
    source={'persona attributes':['first'], 'persona facts':['second'], 'retrieved context messages':[['third','20250420_095500'],['long '*1000,'20250420_095600']], 'keyword message':['last']}
    packed=runner.pack_omem(json.dumps(source), token)
    assert len(token.encode(packed,add_special_tokens=False).ids)<=runner.FINGER_AUGMENT_TOKENS
    assert json.loads(packed)=={'persona attributes':['first'],'persona facts':['second'],'retrieved context messages':[['third','20250420_095500']]}
    pairings=[('lamp2','mem0'),('lamp2','pgmem'),('fingertip','mem0'),('fingertip','pgmem'),('fingertip','omem')]
    datasets={b:list(runner.cases(b)) for b in ('lamp2','fingertip')}
    with TemporaryDirectory() as tmp, patch('nex_vita_transport.phase',return_value=complete('answer',20000)) as api:
        record=runner.predict('fingertip','raw',datasets['fingertip'][0],Path(tmp),answer_limit=None)
        assert record['status']=='complete' and record['answer_cost']['completion_tokens']==20000
        assert api.call_args.kwargs['max_tokens'] is None and api.call_args.args[1]=='medium'
    with TemporaryDirectory() as tmp, patch('nex_vita_transport.phase',side_effect=[complete(),complete(),complete('answer',20000)]) as api:
        c=datasets['fingertip'][0]
        record=runner.predict('fingertip','pgmem',c,Path(tmp),make_backend=lambda m,b,c,s:FakeMemory(c,m),answer_limit=None,answer_effort='none',concise_reasoning=True)
        original,_=runner.prompt_for('fingertip',c,'pgmem',record['retrieved_memory'])
        assert record['status']=='complete' and record['answer_cost']['completion_tokens']==20000
        assert [x.args[1] for x in api.call_args_list]==['none','none','none']
        assert all(x.kwargs['max_tokens'] is None for x in api.call_args_list)
        assert record['prompt'].count(runner.CONCISE_REASONING_NOTE)==1
        assert record['prompt'].replace('\n'+runner.CONCISE_REASONING_NOTE,'')==original
    assert len(datasets['lamp2'])==50 and len(datasets['fingertip'])==20
    original=prompts()('profile','time','scenario','[]')
    adapted=screenshot_prompt(original,0)
    assert adapted.count(ZERO_SCREENSHOT_NOTE)==1
    assert adapted.replace('\n'+ZERO_SCREENSHOT_NOTE,'')==original
    assert all(screenshot_prompt(original,k)==original for k in (1,2,3))
    for bench,method in pairings:
        for c in datasets[bench]:
            example=json.dumps({'persona attributes':['Example memory. '*150]}) if method=='omem' else 'Example memory. '*150
            prompt,meta=runner.prompt_for(bench,c,method,example)
            assert prompt and (bench!='lamp2' or meta['input_tokens_reference']<=512)
            if bench=='fingertip':assert prompt.count(ZERO_SCREENSHOT_NOTE)==1
        with TemporaryDirectory() as tmp, patch('nex_vita_transport.phase',side_effect=[complete(n=12000),complete(n=12000),complete('action',300)]) as api:
            r=runner.predict(bench,method,datasets[bench][0],Path(tmp),make_backend=lambda m,b,c,s:FakeMemory(c,m))
            assert r['status']=='complete' and r['generated_tokens']==24300 and len(r['calls'])==3,r
            assert r['preparation_cost']['completion_tokens']==24000 and r['answer_cost']['completion_tokens']==300
            assert r['preparation_cost']['reasoning_tokens']==23998 and r['preparation_cost']['prompt_tokens']==20
            assert [x.kwargs['max_tokens'] for x in api.call_args_list]==[None,None,16384]
            assert [x.args[1] for x in api.call_args_list]==['none','none','medium']
            assert [c['reasoning_effort'] for c in r['calls']]==['none','none','medium']
            assert all(json.loads(p.read_text(encoding='utf8'))['finished'] for p in (Path(tmp)/'progress').glob('*.json'))
        print(bench,method,'frozen inputs and threaded lifecycle OK',flush=True)
    exhausted={'status':'incomplete','finish_reason':'length','seconds':1,'answer':'partial','reasoning':'actual upstream text\n第二行','effort':'medium','usage':{'completion_tokens':16384}}
    with TemporaryDirectory() as tmp, patch('nex_vita_transport.phase',side_effect=[complete(n=10000),complete(n=10000),exhausted]) as api:
        r=runner.predict('lamp2','mem0',datasets['lamp2'][0],Path(tmp),make_backend=lambda m,b,c,s:FakeMemory(c))
        assert r['status']=='budget_exceeded' and r['generated_tokens']==36384 and api.call_count==3,r
        assert r['failure_stage']=='answer' and r['diagnostic']['category']=='model_token_limit'
        assert r['budget_trace']['reasoning_text']==exhausted['reasoning'] and r['budget_trace']['partial_answer']=='partial'
        assert not r['answer'] and r['budget_trace']['reasoning_text_available']
    stalled={'status':'TimeoutError','seconds':121,'request_stage':'stream','request_sent':True,'reasoning_chars':100,'answer_chars':0,'last_event_seconds':1}
    with TemporaryDirectory() as tmp, patch('nex_vita_transport.phase',side_effect=[complete(n=100),stalled]) as api:
        r=runner.predict('lamp2','mem0',datasets['lamp2'][0],Path(tmp),make_backend=lambda m,b,c,s:FakeMemory(c))
        assert r['status']=='TimeoutError' and r['failure_stage']=='memory_write' and api.call_count==2
        assert r['diagnostic']['category']=='reasoning_stream_stalled' and r['preparation_cost']['missing_usage_calls']==1
        assert r['answer_cost']['calls']==0
        assert 'budget_trace' not in r
    print('Independent preparation and answer costs, hard answer cap and failed preparation termination OK',flush=True)

    from run_lamp_gpg512 import cohort,predict
    task,rows=next(cohort())
    q,gold=rows[0]
    with patch('nex_vita_transport.phase',side_effect=[complete('digest',12000),complete('profile',12000),complete('answer',300)]) as api:
        r=predict(task,q,gold)
        assert r['status']=='complete' and r['preparation_cost']['completion_tokens']==24000
        assert r['answer_cost']['completion_tokens']==300
        assert [c.kwargs['max_tokens'] for c in api.call_args_list]==[None,None,16384]
        assert [c.args[1] for c in api.call_args_list]==['none','none','medium']
    print('GPG-512 independent budgets OK',flush=True)

    # Exercise the real runner's persistence/selection logic without API calls.
    from track_retry import retryable, connection_failure
    assert not retryable({'status':'incomplete','raw':{'phases':[{'finish_reason':'length'}]}})
    assert not retryable({'status':'manual_cancelled'})
    assert not retryable({'status':'TimeoutError'},attempts=2)
    assert connection_failure({'diagnostic':{'category':'reasoning_stream_stalled'}})
    from collections import Counter
    ids=['ok','budget','network','method']
    fake_cases=[{'id':k} for k in ids]
    counts=Counter()
    def fake_predict(bench,method,case,out):
        k=case['id'];counts[k]+=1
        status={'ok':'complete','budget':'budget_exceeded','network':'TimeoutError','method':'method_error'}[k] if counts[k]==1 else ('complete' if k=='network' else 'method_error')
        return {'id':k,'status':status,'generated_tokens':0,'calls':[],'seconds':0}
    def fake_score(bench,items,out,expected,workers,retries=None,first_pass=None,**kwargs):
        path=out/('scores.after_retry.json' if retries is not None else 'scores.json')
        path.write_text(json.dumps({'attempted':len(items),'statuses':dict(Counter(r['status'] for r in items))}),encoding='utf8')
    with TemporaryDirectory() as tmp, patch.object(runner,'cases',return_value=fake_cases), patch.object(runner,'predict',side_effect=fake_predict), patch.object(runner,'score',side_effect=fake_score), patch.object(runner,'encoder'), patch.object(sys,'argv',['run_memory_pairing.py','--bench','lamp2','--method','mem0','--output',tmp,'--workers','1']):
        # First invocation performs only one retry per eligible track.
        runner.main()
        runner.main()
        assert counts=={'ok':1,'budget':1,'network':2,'method':2},counts
        first=[json.loads(l) for l in (Path(tmp)/'predictions.jsonl').read_text().splitlines()]
        retry=[json.loads(l) for l in (Path(tmp)/'predictions.retry.jsonl').read_text().splitlines()]
        assert len(first)==4 and len(retry)==2 and all(r['attempt']==2 for r in retry)
        assert json.loads((Path(tmp)/'scores.json').read_text())['statuses'].get('TimeoutError')==1
        assert json.loads((Path(tmp)/'scores.after_retry.json').read_text())['statuses']['complete']==2
    with TemporaryDirectory() as tmp, patch.object(runner,'cases',return_value=fake_cases), patch.object(runner,'predict',side_effect=fake_predict), patch.object(runner,'score',side_effect=fake_score), patch.object(runner,'encoder'), patch.object(sys,'argv',['run_memory_pairing.py','--bench','lamp2','--method','mem0','--output',tmp,'--workers','1']):
        counts.clear()
        # Emulate a pre-retry-protocol complete first pass without launching retries.
        with patch.object(runner,'retryable',return_value=False):
            runner.main()
        path=Path(tmp)
        cfg=json.loads((path/'experiment.json').read_text());cfg.pop('track_retry_limit')
        (path/'experiment.json').write_text(json.dumps(cfg),encoding='utf8')
        first=[json.loads(l) for l in (path/'predictions.jsonl').read_text().splitlines()]
        for r in first:
            if r['id']=='network':r['diagnostic']={'category':'reasoning_stream_stalled'}
        (path/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in first),encoding='utf8')
        with patch.object(sys,'argv',sys.argv+['--retry-connections-only']):
            runner.main();runner.main()
        assert counts=={'ok':1,'budget':1,'network':2,'method':1},counts
        assert len((path/'predictions.retry.jsonl').read_text().splitlines())==1
    print('Exactly one retry, original records retained, resume does not retry twice',flush=True)
    with TemporaryDirectory() as tmp, patch.object(runner,'cases',return_value=fake_cases[:3]), patch.object(runner,'score',side_effect=fake_score), patch.object(runner,'encoder'), patch.object(sys,'argv',['run_memory_pairing.py','--bench','lamp2','--method','mem0','--output',tmp,'--workers','1']):
        counts.clear()
        def deferred_predict(bench,method,case,out):
            key=case['id'];counts[key]+=1
            status='budget_exceeded' if key=='budget' else 'TimeoutError' if key=='network' and counts[key]<4 else 'complete'
            return {'id':key,'status':status,'generated_tokens':0,'calls':[],'seconds':0}
        with patch.object(runner,'predict',side_effect=deferred_predict):
            for phase in ('first','first','retry','retry','retry','retry','score'):
                with patch.object(sys,'argv',sys.argv+['--deferred-phase',phase]):runner.main()
            with patch.object(sys,'argv',sys.argv+['--workers','10','--deferred-phase','first']):runner.main()
        out=Path(tmp); first={r['id']:r for r in map(json.loads,(out/'predictions.jsonl').read_text().splitlines())}
        latest,attempts=runner.deferred_history(out,first)
        assert counts=={'ok':1,'budget':1,'network':4} and [r['attempt'] for r in attempts]==[2,3,4]
        assert latest['network']['status']=='complete' and latest['budget']['status']=='budget_exceeded'
        assert first['network']['status']=='TimeoutError'
        cfg=json.loads((out/'experiment.json').read_text());assert cfg['workers']==10 and cfg['worker_changes'][0]['from']==1 and cfg['worker_changes'][0]['first_pass_rows']==3
    print('Deferred rounds preserve all attempts, never repeat successful or budget-terminal tracks',flush=True)


if __name__=='__main__':main()
