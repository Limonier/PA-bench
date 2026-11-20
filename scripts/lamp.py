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
    from sii import safe_chat
    suffix='' if a.history=='none' else f'-bm25-k{a.num_retrieved}'
    dst=RESULTS/f'lamp-{a.task}-{a.split}{suffix}.json'
    result={'task':f'LaMP_{a.task}','golds':[]}
    if dst.exists(): result=json.loads(dst.read_text(encoding='utf-8'))
    result['run_config']={'model':os.environ['SII_MODEL'],'reasoning_effort':os.getenv('SII_REASONING_EFFORT','none'),'temperature':0,'baseline':'no_profile','split':'time_test'}
    result['run_config'].update(baseline=a.history,num_retrieved=a.num_retrieved if a.history=='bm25' else 0)
    done={p['id'] for p in result['golds']}
    for q in questions:
        if q['id'] in done: continue
        prompt=q['input']
        if a.history=='bm25':
            from lamp_bm25 import augment
            prompt,_=augment(int(a.task),q,a.num_retrieved)
        response=safe_chat(os.environ['SII_MODEL'], prompt, sample_id=q['id'],benchmark='lamp-'+a.task,max_tokens=16384)
        if response is None: continue
        if response['choices'][0].get('finish_reason')=='length': raise RuntimeError('Truncated model response')
        result['golds'].append({'id':q['id'],'output':response['choices'][0]['message']['content']})
        tmp=dst.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,ensure_ascii=False),encoding='utf-8'); tmp.replace(dst)

if __name__=='__main__': main()
