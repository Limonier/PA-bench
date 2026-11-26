"""Run a fixed 6-user cohort after the two DNS-retry validations pass."""
import json
import random
import shutil
import subprocess
import time
from pathlib import Path

from run_vita_after_fingertip import ROOT, VITA, BACKENDS, read

OUT = ROOT / 'results/vita-tenth-concurrency10-20261121'
RUN = 'tenth-concurrency10-20261121'
VALIDATION = VITA / 'medium60-none30-dns-retry-U010122'
LOGS = ROOT / 'results/vita-full-medium60-none30-20261121'
FAILURES = ('Error evaluating subtask', 'failed after 4 attempts', 'Error generating RAG summary')


def status(state, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'queue-status.json').write_text(json.dumps({'state': state, **details}, indent=2), encoding='utf-8')
    print(state, details, flush=True)


def main():
    tasks = json.loads((ROOT / 'data/vita/tasks.json').read_text(encoding='utf-8'))
    sampled = random.Random(20261121).sample(tasks, 6)
    expected = {t['id']: len(t['subtasks']) for t in sampled}
    details = dict(seed=20261121, users=expected, trials=4, concurrency=10,
                   backends=BACKENDS, subtasks_per_backend=4*sum(expected.values()))
    status('waiting_for_validation', **details)
    deadline = time.monotonic() + 3600
    while True:
        pending = []
        for name in ('null', 'rewrite'):
            log = '\n'.join((LOGS / f'dns-retry-{name}{suffix}').read_text(encoding='utf-8', errors='replace')
                            for suffix in ('.log', '.err.log'))
            if any(marker in log for marker in FAILURES):
                raise RuntimeError(f'{name} validation contains unresolved evaluation errors')
            sims = (read(VALIDATION / (name + '.json')) or {}).get('simulations', [])
            if not sims:
                pending.append(name)
                continue
            info = (sims[0].get('reward_info') or {}).get('info', {})
            if len(sims) != 1 or info.get('num_subtasks') != 10 or info.get('num_evaluated') != 10:
                raise RuntimeError(f'{name} validation coverage mismatch')
        if not pending:
            break
        if time.monotonic() >= deadline:
            raise RuntimeError(f'Validation timeout: {pending}')
        time.sleep(20)

    shell = shutil.which('pwsh') or shutil.which('powershell')
    if not shell:
        raise RuntimeError('PowerShell not found')
    completed = []
    for name in BACKENDS:
        memory = 'rag' if name.startswith('rag-') else name
        status('running', **details, active=name, completed=completed)
        # PowerShell -File does not reliably bind array arguments; use a fixed, locally generated ID array.
        ids = ','.join("'"+task_id+"'" for task_id in expected)
        command = f"& './scripts/run_vita.ps1' -Memory {memory} -TaskIds @({ids}) -Trials 4 -Concurrency 10 -RunName {RUN}"
        if memory == 'rag':
            command += ' -LocalEmbeddings'
        path = OUT / (name + '.log')
        with path.open('w', encoding='utf-8') as log:
            code = subprocess.run([shell, '-NoProfile', '-Command', command], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
        sims = (read(VITA / RUN / (name + '.json')) or {}).get('simulations', [])
        pairs = {(s.get('task_id'), s.get('trial')) for s in sims}
        required = {(task_id, trial) for task_id in expected for trial in range(4)}
        valid = all((s.get('reward_info') or {}).get('info', {}).get('num_evaluated') == expected.get(s.get('task_id')) for s in sims)
        log_text = path.read_text(encoding='utf-8', errors='replace')
        if code or pairs != required or not valid or any(marker in log_text for marker in FAILURES):
            raise RuntimeError(f'{name}: incomplete or degraded experiment ({len(pairs)}/{len(required)}, exit={code})')
        completed.append(name)
    status('finished', **details, completed=completed)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('blocked', error=str(exc))
        raise
