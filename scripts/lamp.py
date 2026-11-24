"""LaMP time split no-profile zero-shot baseline; official evaluation is separate."""
import argparse
import json
import os
from pathlib import Path
from run_context import RESULTS

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task',choices=['1','2','3','4','5','7'],required=True)
    parser.add_argument('--split',choices=['dev','test'],default='test')
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--limit',type=int)
    parser.add_argument('--history',choices=['none','bm25'],default='none')
    parser.add_argument('--num-retrieved',type=int,default=4)
    a=parser.parse_args()
    src=ROOT/f'data/lamp/LaMP_{a.task}/{a.split}/{a.split}_questions.json'
    questions=json.loads(src.read_text(encoding='utf-8'))
    if a.limit: questions=questions[:a.limit]
    if a.dry_run:
        assert len({q['id'] for q in questions})==len(questions)
        print(f'{len(questions)} unique questions; no-profile baseline')
        return
    from timed_nex import answer, MODEL
    from track_retry import retryable
    if os.getenv('SII_REASONING_EFFORT','medium') != 'medium':
        raise ValueError('New experiments require medium-task16384')
    suffix='' if a.history=='none' else f'-bm25-k{a.num_retrieved}'
    dst=RESULTS/f'lamp-{a.task}-{a.split}{suffix}.json'
    result={'task':f'LaMP_{a.task}','golds':[]}
    if dst.exists(): result=json.loads(dst.read_text(encoding='utf-8'))
    result['run_config']={'model':MODEL,'reasoning_effort':'medium','temperature':0,'baseline':'no_profile','split':'time_test','task_token_budget':16384,'fallback':None}
    result['run_config'].update(baseline=a.history,num_retrieved=a.num_retrieved if a.history=='bm25' else 0,track_retry_limit=1)
    attempts=result.setdefault('attempts',{})
    done={p['id'] for p in result['golds']}
    for q in questions:
        previous=attempts.get(q['id'])
        if q['id'] in done or (previous and not retryable(previous,previous.get('attempt',1))): continue
        prompt=q['input']
        if a.history=='bm25':
            from lamp_bm25 import augment
            prompt,_=augment(int(a.task),q,a.num_retrieved)
        while not previous or retryable(previous,previous.get('attempt',1)):
            response=answer(prompt)
            current={'status':response['status'],'generated_tokens':response['generated_tokens'],'raw':response,'attempt':2 if previous else 1}
            if previous:current['first_attempt']=previous
            attempts[q['id']]=current
            if response['status']=='complete':
                result['golds'].append({'id':q['id'],'output':response['answer']})
            tmp=dst.with_suffix('.tmp')
            tmp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8'); tmp.replace(dst)
            previous=current

if __name__=='__main__': main()
