"""Gate the FingerTip none/concise experiment on two real tracks per method."""
import json
import msvcrt
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from timed_nex import _phase
from track_retry import retryable

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(os.environ.get('PA_BENCH_QUEUE_OUTPUT', str(ROOT / 'results/fingertip-methods-none-concise-20261124')))
SUBSET = ROOT / 'results/p0-expanded-20261123/subset.json'
METHODS = ('pgmem', 'omem', 'mem0')
WORKERS = 2
DEFERRED_WORKERS = 10
if os.getenv('PA_BENCH_ALT_MODEL'):
    SUBSET = ROOT / 'data/fingertip_model_comparison_two_v1.json'
    METHODS = ('pgmem',)
    DEFERRED_WORKERS = 2
EXPECTED = len(json.loads(SUBSET.read_text(encoding='utf-8'))['fingertip']['ids'])


def anomalies(out):
    for name in ('predictions.jsonl', 'predictions.retry.jsonl', 'sr1-items.jsonl', 'sr1-retry-items.jsonl'):
        path = out / name
        if not path.exists():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # A concurrently appended line may not be complete yet.
            if row['status'] != 'complete':
                raw = row.get('raw', {})
                yield (name, row['id'], row['status'], row.get('failure_stage'),
                       (row.get('diagnostic') or {}).get('category'), raw.get('status'), row.get('attempt', 1))


def inspect(out, known=()):
    return next((a for a in anomalies(out) if a not in known), None)


def run(method, limit=None, resume=False):
    out = OUTPUT / f'fingertip-{method}'
    out.mkdir(exist_ok=True)
    known = set(anomalies(out)) if resume else set()
    if known:
        print('PRESERVED_FAILURES', method, sorted(known), flush=True)
    args = [sys.executable, '-u', str(ROOT / 'scripts/run_memory_pairing.py'),
            '--bench', 'fingertip', '--method', method, '--output', str(out),
            '--workers', str(WORKERS), '--subset', str(SUBSET), '--uncapped-answer', '--direct-fingertip']
    if limit:
        args += ['--limit', str(limit)]
    print('BEGIN', 'PILOT' if limit else 'FULL', method, flush=True)
    with (out / 'formal.log').open('a', encoding='utf-8') as log, (out / 'formal.err.log').open('a', encoding='utf-8') as err:
        child = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=err)
        try:
            while child.poll() is None:
                if anomaly := inspect(out, known):
                    raise RuntimeError(f'{method}: {anomaly}')
                time.sleep(2)
            if child.returncode or (anomaly := inspect(out, known)):
                raise RuntimeError(f'{method}: exit={child.returncode}, anomaly={inspect(out, known)}')
            score_path = out / ('scores.after_retry.json' if (out / 'predictions.retry.jsonl').exists() else 'scores.json')
            scores = json.loads(score_path.read_text(encoding='utf-8'))
            expected = limit or scores['expected']
            if (scores['attempted'], scores['valid_predictions'], scores['SR1_judged']) != (expected,) * 3:
                raise RuntimeError(f'{method}: incomplete prediction or scoring: {scores["statuses"]}')
            print('PASS', method, expected, flush=True)
        finally:
            if child.poll() is None:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                child.wait()


def run_deferred(method, phase):
    out = OUTPUT / f'fingertip-{method}'
    out.mkdir(exist_ok=True)
    args = [sys.executable, '-u', str(ROOT / 'scripts/run_memory_pairing.py'),
            '--bench', 'fingertip', '--method', method, '--output', str(out),
            '--workers', str(DEFERRED_WORKERS), '--subset', str(SUBSET), '--uncapped-answer',
            '--direct-fingertip', '--deferred-phase', phase]
    print('BEGIN_DEFERRED', phase, method, 'workers', DEFERRED_WORKERS, flush=True)
    with (out/'formal.log').open('a',encoding='utf-8') as log, (out/'formal.err.log').open('a',encoding='utf-8') as err:
        subprocess.run(args,cwd=ROOT,stdout=log,stderr=err,check=True)
    print('FINISH_DEFERRED',phase,method,flush=True)


def unresolved(method):
    out=OUTPUT/f'fingertip-{method}'
    latest={}
    for name in ('predictions.jsonl','predictions.retry.jsonl'):
        path=out/name
        if path.exists():
            for line in path.read_text(encoding='utf-8').splitlines():
                row=json.loads(line);latest[row['id']]=row
    if len(latest)!=EXPECTED:raise RuntimeError(f'{method}: incomplete first pass: {len(latest)}/{EXPECTED}')
    return sum(retryable(r) for r in latest.values())


def deferred_queue():
    for method in METHODS:
        run_deferred(method,'first')
    round_no=0
    while True:
        pending={m:unresolved(m) for m in METHODS}
        print('DEFERRED_PENDING',pending,flush=True)
        if not any(pending.values()):break
        round_no+=1
        print('DEFERRED_ROUND',round_no,flush=True)
        for method,count in pending.items():
            if count:run_deferred(method,'retry')
        if any(unresolved(m) for m in METHODS):time.sleep(60)
    for method in METHODS:
        run_deferred(method,'score')
        out=OUTPUT/f'fingertip-{method}'
        score_path=out/('scores.after_retry.json' if (out/'predictions.retry.jsonl').exists() else 'scores.json')
        scores=json.loads(score_path.read_text(encoding='utf-8'))
        if scores['attempted']!=EXPECTED or scores['SR1_judged']!=scores['valid_predictions'] or unresolved(method):
            raise RuntimeError(f'{method}: final evaluation still incomplete')
        print('REAL_RESULTS_COMPLETE',method,scores['valid_predictions'],scores['SR1_judged'],flush=True)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / 'queue.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        print('QUEUE_READY', os.getpid(), flush=True)
        if '--deferred' in sys.argv:
            deferred_queue()
            print('QUEUE_FINISHED',flush=True)
            return
        resume = '--resume' in sys.argv
        if resume:
            # Repeated new TLS connections at the actual two-worker concurrency.
            with ThreadPoolExecutor(WORKERS) as pool:
                for i in range(12):
                    checks = [pool.submit(_phase, 'Reply only OK', effort, 25, max_tokens=64) for effort in ('none', 'medium')]
                    for f in checks:
                        probe = f.result()
                        print('STABILITY_PROBE', i+1, probe['effort'], probe['status'], probe.get('http_status'), round(probe['seconds'], 2), probe.get('request_stage'), flush=True)
                        if probe['status'] != 'complete' or probe.get('http_status') != 200 or probe['answer'].strip() != 'OK':
                            raise RuntimeError('SII stability probe failed')
                    if i < 11:
                        time.sleep(15)
            print('STABILITY_PASSED 24/24', flush=True)
        else:
            probe = _phase('Reply only OK', 'none', 20, max_tokens=16)
            print('SERVICE_PROBE', probe['status'], probe.get('http_status'), flush=True)
            if probe['status'] != 'complete' or probe['answer'].strip() != 'OK':
                raise RuntimeError('SII probe failed')
            for method in METHODS:
                run(method, limit=2)
            print('PILOT_PASSED', flush=True)
        for method in METHODS:
            run(method, resume=resume)
        print('QUEUE_FINISHED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('QUEUE_STOPPED', str(exc), flush=True)
        sys.exit(1)
