"""Track 1, official prompt with 0-3 original screenshots and fixed Nex policy."""
import argparse
import ast
import csv
import json
import hashlib
from pathlib import Path
from fingertip_data import tasks as load_tasks
from fingertip_images import screenshots, image_metadata, request_content

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'upstream/FingerTip-20K'
ZERO_SCREENSHOT_NOTE = "No screenshots are provided; infer the user's intent from the supplementary information only."


def screenshot_prompt(prompt, count):
    if count != 0:
        return prompt
    head, separator, rest = prompt.partition('\n\n## Input')
    if not separator:
        raise ValueError('Official prompt Input marker not found')
    return head + '\n' + ZERO_SCREENSHOT_NOTE + separator + rest

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
    p.add_argument('--screenshots',type=int,choices=[0,1,2,3],default=0)
    p.add_argument('--screenshot-root',type=Path,default=SOURCE)
    p.add_argument('--sampled',action='store_true',help='Official 100-row subset with full-set row identities')
    p.add_argument('--check-inputs',action='store_true',help='Validate selected images offline; write nothing')
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    tasks=load_tasks(sampled=args.sampled)
    if args.limit is not None and args.limit < 1: p.error('--limit must be positive')
    history=rows('total.csv')
    profiles={r['user_id']:r for r in rows('user_profile.csv')}
    if args.limit: tasks=tasks[:args.limit]
    # Validate the entire selected subset before spending any model calls.
    images={t['id']:screenshots(args.screenshot_root,t,args.screenshots) for t in tasks}
    metadata={key:image_metadata(paths) for key,paths in images.items()}
    if args.check_inputs:
        print(json.dumps({'samples':len(tasks),'screenshots_per_sample':args.screenshots,'valid':True}))
        return
    if not args.dry_run:
        from timed_nex import answer, MODEL
        from track_retry import retryable
    prompt_fn=prompts()
    if args.output is None:
        suffix='-zero-notice-v1' if args.screenshots==0 else ''
        args.output=ROOT/f'results/fingertip-nex-medium-task16384-retry1-images{args.screenshots}{suffix}'
    args.output.mkdir(parents=True,exist_ok=True)
    config={'model':'nex-n2.5-pro','primary':'medium','fallback':None,'temperature':0,'task_token_budget':16384,'budget_scope':'per_task_completion',
            'track_retry_limit':1,'screenshot_count':args.screenshots,'task_ids':[t['id'] for t in tasks],
            'test_sha256':hashlib.sha256((SOURCE/'test_suggestion.csv').read_bytes()).hexdigest(),
            'official_code_sha256':hashlib.sha256((SOURCE/'proactive_suggestion.py').read_bytes()).hexdigest(),
            'history_sha256':hashlib.sha256((SOURCE/'total.csv').read_bytes()).hexdigest(),
            'profile_sha256':hashlib.sha256((SOURCE/'user_profile.csv').read_bytes()).hexdigest(),'images':metadata}
    if args.screenshots==0:
        config['zero_screenshot_note']=ZERO_SCREENSHOT_NOTE
    config_path=args.output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=config:
        raise ValueError('Inputs or screenshot level changed; select a new experiment directory')
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    dst=args.output/('fingertip-prompts.jsonl' if args.dry_run else 'fingertip-predictions.jsonl')
    attempts={}
    if dst.exists() and not args.dry_run:
        for line in dst.read_text(encoding='utf-8').splitlines():
            r=json.loads(line);attempts.setdefault(r['id'],[]).append(r)
    with dst.open('w' if args.dry_run else 'a',encoding='utf-8') as out:
        for t in tasks:
            prior=attempts.get(t['id'],[])
            if prior and not retryable(prior[-1],len(prior)): continue
            u=profiles[t['user_id']]
            profile=f"{u['sex']}，{u['age']}岁，职业为{u['occupation']}，现居{u['address']}，{u['family']}，使用{u['phone_brand']}手机"
            h=sorted([r for r in history if r['user_id']==t['user_id'] and r['time']<t['time']],key=lambda r:r['time'])[-20:]
            assert all(r['time']<t['time'] for r in h)
            previous=[f"时间：{r['time']}，地点：{r['scenario']}，意图：{r['intentDescription']}" for r in h]
            prompt=screenshot_prompt(prompt_fn(profile,t['time'],t['scenario'],str(previous)),args.screenshots)
            result={'id':t['id'],'row_index':t['row_index'],'legacy_id':t['legacy_id'],
                    'screenshot_count':args.screenshots,'images':metadata[t['id']],
                    'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest()}
            if args.dry_run:
                result['prompt']=prompt;out.write(json.dumps(result,ensure_ascii=False)+'\n');out.flush()
            else:
                content=request_content(prompt,images[t['id']])
                while not prior or retryable(prior[-1],len(prior)):
                    response=answer(content,temperature=0,max_tokens=16384)
                    attempt=dict(result,prediction=response['answer'].strip(),raw=response,status=response['status'],model=MODEL,reasoning_effort='medium',attempt=len(prior)+1)
                    out.write(json.dumps(attempt,ensure_ascii=False)+'\n');out.flush();prior.append(attempt)

if __name__=='__main__': main()
