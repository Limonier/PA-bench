"""Freeze today's P0 cohort and resume compatible pilot records without replay."""
import argparse
import json
import msvcrt
import os
import random
import shutil
from datetime import datetime, timedelta

import run_memory_pairing as runner
import run_memory_pairing_queue as queue

OUTPUT = runner.ROOT / 'results/p0-expanded-20261123'
MANIFEST = OUTPUT / 'subset.json'
PAIRS = [('lamp2', m) for m in ('raw','no_history','gpg','mem0','pgmem')] + [
    ('fingertip', m) for m in ('raw','no_history','pgmem','omem','mem0')]
OLD = runner.ROOT / 'results/memory-pairings-prep-none-v4'
REPAIRED = OLD.with_name(OLD.name + '-zero-notice-pgmem-structured-v1')
SEEDS = {('lamp2','mem0'): OLD/'lamp2-mem0', ('lamp2','gpg'): REPAIRED/'lamp2-gpg',
         **{('fingertip',m):REPAIRED/f'fingertip-{m}' for m in ('pgmem','omem','mem0')}}


def freeze():
    old = json.loads(runner.SUBSET.read_text(encoding='utf-8'))
    parent = json.loads((runner.ROOT/old['parent_manifest']).read_text(encoding='utf-8'))
    spec = parent['lamp']['branches']['2']
    questions = {str(q['id']):str(q['user_id']) for q in runner.load_source(spec['source'])}
    tasks = {r['id']:r for r in runner.fingertip_tasks()}
    past = runner.finger_rows('total.csv')
    result = {**old, 'version':2, 'seed':20261123, 'previous_subset_sha256':runner.sha(runner.SUBSET)}
    for bench, pool, users, target in [('lamp2',spec['ids'],questions,100),
        ('fingertip',parent['fingertip']['ids'],{i:str(t['user_id']) for i,t in tasks.items()},60)]:
        ids = list(old[bench]['ids']); seen = {users[i] for i in ids}
        candidates = sorted(pool); random.Random(20261123).shuffle(candidates)
        for i in candidates:
            if len(ids)==target:break
            if users[i] in seen:continue
            if bench=='fingertip' and len(runner.fingertip_history(past,tasks[i]['user_id'],tasks[i]['time'],20))!=20:continue
            ids.append(i);seen.add(users[i])
        assert len(ids)==len(seen)==target
        result[bench] = {'sampling':'retain pilot IDs; seeded shuffle, one eligible row per new user','ids':ids}
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():
        assert json.loads(MANIFEST.read_text(encoding='utf-8'))==result
    else:MANIFEST.write_text(json.dumps(result,indent=2),encoding='utf-8')
    for bench in ('lamp2','fingertip'):
        work=list(runner.cases(bench,MANIFEST))
        for case in work:
            for method in runner.PAIRS[bench]:
                example=json.dumps({'persona attributes':['Example memory. '*100]}) if method=='omem' else 'Example memory. '*100
                prompt,meta=runner.prompt_for(bench,case,method,example)
                assert prompt and (bench!='lamp2' or meta['input_tokens_reference']<=512)
        print('INPUTS_CHECKED',bench,len(work),flush=True)


def seed(bench, method):
    src=SEEDS.get((bench,method));dst=OUTPUT/f'{bench}-{method}'
    if src is None or (dst/'predictions.jsonl').exists():return
    cfg=json.loads((src/'experiment.json').read_text(encoding='utf-8'))
    assert cfg['subset_sha256']==runner.sha(runner.SUBSET)
    assert cfg['model']=='nex-n2.5-pro' and cfg['preparation_effort']=='none' and cfg['reasoning_effort']=='medium' and cfg['task_token_budget']==16384 and cfg['fallback'] is None and cfg['track_retry_limit']==1
    if cfg['adapter_revision']!=runner.adapter_revision(bench,method):
        print('PILOT_INCOMPATIBLE',bench,method,flush=True);return
    if bench=='fingertip':assert cfg['zero_screenshot_note']==runner.ZERO_SCREENSHOT_NOTE
    assert (src/'scores.json').exists()
    dst.mkdir(parents=True,exist_ok=True)
    for name in ('predictions.jsonl','predictions.retry.jsonl','sr1-items.jsonl','sr1-retry-items.jsonl'):
        if (src/name).exists():shutil.copyfile(src/name,dst/name)
    (dst/'pilot_source.json').write_text(json.dumps({'source':str(src),'experiment':cfg},indent=2),encoding='utf-8')
    print('PILOT_REUSED',bench,method,cfg['expected'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--check-only',action='store_true');a=p.parse_args()
    freeze()
    if a.check_only:return
    if not (os.getenv('SII_API_KEY') or os.getenv('SII_GATEWAY_TOKEN')):raise RuntimeError('Gateway credential missing')
    os.environ.update(PYTHONUTF8='1',MEM0_TELEMETRY='False',TOKENIZERS_PARALLELISM='false')
    os.environ.pop('VITA_DIAGNOSTIC_2048',None)
    queue.OUTPUT=OUTPUT
    with (OUTPUT/'queue.lock').open('a+b') as lock:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        # Continue the authorized P0 queue if a pair extends beyond midnight.
        deadline=datetime.now(queue.SHANGHAI)+timedelta(hours=12)
        for bench,method in PAIRS:
            seed(bench,method)
            queue.run_pair(bench,method,deadline,subset=MANIFEST)
        print('P0_QUEUE_FINISHED',flush=True)


if __name__=='__main__':main()
