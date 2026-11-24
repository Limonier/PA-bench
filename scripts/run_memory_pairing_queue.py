"""Resume the agreed sequential queue after a successful IPv4/IPv6 model probe."""
import msvcrt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone

from timed_nex import phase

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'results/memory-pairings-prep-none-v4-zero-notice'
PAIRS = [('lamp2', 'pgmem'), ('fingertip', 'pgmem'),
         ('fingertip', 'omem'), ('fingertip', 'mem0')]
SHANGHAI = timezone(timedelta(hours=8))


def wait_for_service(deadline):
    while datetime.now(SHANGHAI) < deadline:
        # HTTPSConnection uses system getaddrinfo, accepting both A and AAAA.
        result = phase('Reply only OK', 'none', 30, max_tokens=16)
        print('SERVICE_PROBE', result['status'], result.get('http_status'), flush=True)
        if result['status'] == 'complete':
            return
        if result.get('http_status') in (401, 403):
            raise RuntimeError('Gateway credential rejected; no evaluation submitted')
        print('WAITING_FOR_SII', datetime.now(SHANGHAI).isoformat(), flush=True)
        time.sleep(60)
    raise RuntimeError('Service unavailable before midnight; queue stopped')


def run_pair(bench, method, deadline, limit=None, subset=None, uncapped_answer=False, direct_fingertip=False):
    wait_for_service(deadline)
    out = OUTPUT / f'{bench}-{method}'
    out.mkdir(parents=True, exist_ok=True)
    print('BEGIN', bench, method, datetime.now(SHANGHAI).isoformat(), flush=True)
    with (out / 'formal.log').open('a', encoding='utf-8') as log, \
            (out / 'formal.err.log').open('a', encoding='utf-8') as err:
        subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/run_memory_pairing.py'),
                        '--bench', bench, '--method', method, '--output', str(out),
                        '--workers', '10'] + (['--limit', str(limit)] if limit else []) + (['--subset', str(subset)] if subset else []) + (['--uncapped-answer'] if uncapped_answer else []) + (['--direct-fingertip'] if direct_fingertip else []), cwd=ROOT, stdout=log, stderr=err, check=True)
    print('FINISH', bench, method, datetime.now(SHANGHAI).isoformat(), flush=True)


def main():
    global OUTPUT, PAIRS
    if '--pgmem-repair' in sys.argv:
        OUTPUT = OUTPUT.with_name(OUTPUT.name + '-pgmem-structured-v1')
        PAIRS = PAIRS[1:]
    if not (os.getenv('SII_API_KEY') or os.getenv('SII_GATEWAY_TOKEN')):
        raise RuntimeError('Configure an authorized Nex gateway credential before starting')
    os.environ.update(PYTHONUTF8='1', MEM0_TELEMETRY='False', TOKENIZERS_PARALLELISM='false')
    os.environ.pop('VITA_DIAGNOSTIC_2048', None)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Keep the OS lock open throughout; a crashed process releases it automatically.
    with (OUTPUT / 'queue.lock').open('a+b') as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        deadline = datetime.now(SHANGHAI).replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        print('QUEUE_READY', os.getpid(), datetime.now(SHANGHAI).isoformat(), flush=True)
        if '--pgmem-repair' in sys.argv:
            run_pair('fingertip', 'pgmem', deadline, limit=2)
            out = OUTPUT / 'fingertip-pgmem'
            rows = [json.loads(line) for line in (out / 'predictions.jsonl').read_text(encoding='utf-8').splitlines()]
            retry = out / 'predictions.retry.jsonl'
            if retry.exists():
                rows += [json.loads(line) for line in retry.read_text(encoding='utf-8').splitlines()]
            final = {r['id']:r for r in rows}
            if len(final) != 2 or any(r['status'] != 'complete' for r in final.values()):
                raise RuntimeError('PGMem repair smoke test did not complete both tasks; queue stopped')
            print('PGMEM_REPAIR_VALIDATED', flush=True)
        for bench, method in PAIRS:
            run_pair(bench, method, deadline)
        if datetime.now(SHANGHAI) + timedelta(minutes=30) < deadline:
            run_pair('lamp2', 'gpg', deadline)
        else:
            print('SKIP_OPTIONAL_GPG_INSUFFICIENT_TIME', flush=True)
        print('QUEUE_FINISHED', datetime.now(SHANGHAI).isoformat(), flush=True)


if __name__ == '__main__':
    main()
