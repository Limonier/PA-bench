"""Track 1, zero screenshots. Reuses the official prompt verbatim via AST."""
import argparse
import ast
import csv
import json
import os
from pathlib import Path
from run_context import RESULTS

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'upstream/FingerTip-20K'

def prompts():
    tree = ast.parse((SOURCE/'proactive_suggestion.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='get_prompt')
    scope = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<official prompt>', 'exec'), scope)
    return scope['get_prompt']

def rows(name):
    with (SOURCE/name).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--limit',type=int)
    args=p.parse_args()
    tasks=rows('test_suggestion.csv')
    history=rows('total.csv')
    profiles={r['user_id']:r for r in rows('user_profile.csv')}
    if args.limit: tasks=tasks[:args.limit]
    client=None
    if not args.dry_run:
        from sii import safe_chat
    prompt_fn=prompts()
    dst=RESULTS/('fingertip-prompts.jsonl' if args.dry_run else 'fingertip-predictions.jsonl')
    done=set()
    if dst.exists() and not args.dry_run:
        done={json.loads(line)['id'] for line in dst.read_text(encoding='utf-8').splitlines()}
    with dst.open('w' if args.dry_run else 'a',encoding='utf-8') as out:
        for t in tasks:
            if t['user_id']+'/'+t['time'] in done: continue
            u=profiles[t['user_id']]
            profile=f"{u['sex']}，{u['age']}岁，职业为{u['occupation']}，现居{u['address']}，{u['family']}，使用{u['phone_brand']}手机"
            h=sorted([r for r in history if r['user_id']==t['user_id'] and r['time']<t['time']],key=lambda r:r['time'])[-20:]
            assert all(r['time']<t['time'] for r in h)
            previous=[f"时间：{r['time']}，地点：{r['scenario']}，意图：{r['intentDescription']}" for r in h]
            prompt=prompt_fn(profile,t['time'],t['scenario'],str(previous))
            result={'id':t['user_id']+'/'+t['time']}
            if args.dry_run: result['prompt']=prompt
            else:
                response=safe_chat(os.environ['SII_MODEL'],prompt,sample_id=result['id'],benchmark='fingertip',max_tokens=16384)
                if response is None: continue
                if response['choices'][0].get('finish_reason')=='length': raise RuntimeError('Truncated model response')
                result['prediction']=response['choices'][0]['message']['content']
                result['usage']=response.get('usage')
                result['model']=os.environ['SII_MODEL']
                result['reasoning_effort']=os.getenv('SII_REASONING_EFFORT','none')
            out.write(json.dumps(result,ensure_ascii=False)+'\n'); out.flush()

if __name__=='__main__': main()
