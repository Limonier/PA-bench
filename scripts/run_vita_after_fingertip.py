"""User-requested dependency queue; do not start Vita before FingerTip finishes."""
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINGER = ROOT / 'results/fingertip-zero100-medium-max16384-20261121/scores/scores.json'
VITA = ROOT / 'upstream/VitaBench-2.0/data/simulations/verification'
VALIDATION_RUN = 'medium60-none30-rewrite16k-U010122'
CHECK = VITA / VALIDATION_RUN
OUT = ROOT / 'results/vita-full-medium60-none30-20261121'
BACKENDS = ['null', 'full_context', 'rewrite', 'rag-local-minilm']


def read(path):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def status(state, **details):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'queue-status.json').write_text(json.dumps({'state': state, **details}, indent=2), encoding='utf-8')
    print(state, flush=True)


def ready_finger():
    result = read(FINGER) or {}
    return result.get('n') == 100 and result.get('SR1_judged') == 100 and result.get('SR1_pending_or_failed') == 0


def ready_vita():
    pending = []
    for name in BACKENDS:
        data = read(CHECK / (name + '.json')) or {}
        simulations = data.get('simulations', [])
        if not simulations:
            pending.append(name)
            continue
        log_name = 'rag' if name.startswith('rag-') else name
        log_path = OUT / ('validation-' + log_name + '.log')
        log = log_path.read_text(encoding='utf-8', errors='replace') if log_path.exists() else ''
        if 'Error evaluating subtask' in log or 'Traceback' in log:
            raise RuntimeError(f'{name}: validation log contains unresolved execution/evaluator errors')
        reward = simulations[0].get('reward_info') or {}
        info = reward.get('info', {})
        if info.get('num_subtasks') != 10 or info.get('num_evaluated') != 10:
            raise RuntimeError(f'{name}: full-user validation has missing subtask evaluations')
    return pending


def main():
    os.chdir(ROOT)
    status('waiting_for_fingertip', expected_samples=100)
    while not ready_finger():
        time.sleep(20)
    # The local RAG replacement needs the existing loopback embedding service.
    import urllib.request
    body = json.dumps({'model': 'paraphrase-multilingual-MiniLM-L12-v2', 'input': ['health check']}).encode()
    request = urllib.request.Request('http://127.0.0.1:8766/v1/embeddings', data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=30) as response:
        embedding = json.load(response)['data'][0]['embedding']
        if len(embedding) != 384:
            raise RuntimeError('Local embedding service/model mismatch')
    shell = shutil.which('pwsh') or shutil.which('powershell')
    if not shell:
        raise RuntimeError('PowerShell not found')
    status('validating', user='U010122', backends=BACKENDS)
    validation = []
    for name in BACKENDS:
        memory = 'rag' if name.startswith('rag-') else name
        command = [shell, '-NoProfile', '-File', str(ROOT / 'scripts/run_vita.ps1'),
                   '-Memory', memory, '-TaskIds', 'U010122', '-Trials', '1',
                   '-Concurrency', '1', '-RunName', VALIDATION_RUN]
        if memory == 'rag':
            command.append('-LocalEmbeddings')
        log = (OUT / ('validation-' + memory + '.log')).open('w', encoding='utf-8')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        validation.append((name, proc, log))
    validation_codes = {}
    for name, proc, log in validation:
        validation_codes[name] = proc.wait()
        log.close()
    if any(validation_codes.values()) or ready_vita():
        raise RuntimeError(f'Full-user validation did not pass: {validation_codes}')
    status('running', users_per_backend=56, trials=4, backends=BACKENDS, rag_variant='local MiniLM replacement')
    processes = []
    for name in BACKENDS:
        command = [shell, '-NoProfile', '-File', str(ROOT / 'scripts/run_vita.ps1'),
                   '-Memory', 'rag' if name.startswith('rag-') else name, '-Tasks', '56', '-Trials', '4',
                   '-Concurrency', '2', '-RunName', 'full-medium60-none30-20261121']
        if name.startswith('rag-'):
            command.append('-LocalEmbeddings')
        log = (OUT / (name + '.log')).open('w', encoding='utf-8')
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        processes.append((name, proc, log))
    codes = {}
    for name, proc, log in processes:
        codes[name] = proc.wait()
        log.close()
        result = read(VITA / 'full-medium60-none30-20261121' / (name + '.json')) or {}
        simulations = result.get('simulations', [])
        pairs = {(s.get('task_id'), s.get('trial')) for s in simulations}
        if len(pairs) != 224:
            codes[name] = codes[name] or 1
    status('finished' if all(code == 0 for code in codes.values()) else 'failed', exit_codes=codes)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('blocked', error=str(exc))
        raise
