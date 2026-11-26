"""GPG-512 on the immutable initial LaMP evaluation cohort."""
import os
if __name__ == '__main__':
    os.environ['USE_TORCH'] = '0'
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import argparse
import hashlib
import json
import time
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rank_bm25 import BM25Okapi
from official_lamp_inputs import ROOT, tokenizer, INPUT_LENGTH, scope, names
from lamp_bm25 import template_names
from nex_vita_transport import chat, task_budget, TaskBudget, NexDiagnosticStop, TASK_TOKEN_BUDGET, PREPARATION_EFFORT, token_cost
from track_retry import retryable
sys.path.insert(0, str(ROOT))
from digital_twin_methods import create, History

MANIFEST = ROOT / 'data/initial_eval_set_v1.json'
FIELDS = {1: ('title',), 2: ('description','tag'), 3: ('text','score'),
          4: ('text','title'), 5: ('abstract','title'), 7: ('text',)}
K = 4
PROFILE_BUDGET = 96


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source(spec):
    p = ROOT / spec['path']
    if sha(p) != spec['sha256']:
        raise ValueError(f'Frozen source hash changed: {p.name}')
    return json.loads(p.read_text(encoding='utf-8'))


def cohort():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for branch, spec in manifest['lamp']['branches'].items():
        questions = {str(q['id']): q for q in load_source(spec['source'])}
        golds = {str(g['id']): g['output'] for g in load_source(spec['gold_source'])['golds']}
        assert len(spec['ids']) == len(set(spec['ids'])) == 200
        assert set(spec['ids']) <= questions.keys() & golds.keys()
        yield int(branch), [(questions[i], golds[i]) for i in spec['ids']]


def retrieve(task, q):
    profile = q['profile']
    if not profile:
        return []
    corpus, query = scope[names[task]](q['input'], profile)
    # Exact official BM25 terms and rank ordering; fixed k=4, not test-tuned.
    return BM25Okapi([s.split() for s in corpus]).get_top_n(query.split(), profile, n=min(K,len(profile)))


def history_for(task, selected):
    # Only fields observable in the upstream task-specific history formatter.
    return [History(text='\n'.join(f'{key}: {entry[key]}' for key in FIELDS[task])) for entry in selected]


def trim(tok, text, cap):
    ids = tok(text, add_special_tokens=False)['input_ids']
    return text if len(ids) <= cap else tok.decode(ids[:cap], skip_special_tokens=True)


def pack(task, q, selected, profile, digest, memory_label='Personal profile'):
    tok = tokenizer()
    size = lambda t: len(tok(t)['input_ids'])
    # LaMP-7 GPG includes both the selected writing-style guidance and profile.
    profile_text = memory_label + ': ' + profile.strip()
    if task == 7:
        block = trim(tok, 'Guided analysis: '+digest.strip(), 24) + '\n' + trim(tok,profile_text,PROFILE_BUDGET-24)
    else:
        block = trim(tok,profile_text,PROFILE_BUDGET)
    original = q['input']
    question = original
    # The upstream allocation reserves up to 60% of the input for the query.
    # Only shorten a long query if the profile and basic raw-history space cannot fit.
    if selected and size(question) + size(block) + 64 > INPUT_LENGTH:
        question = trim(tok, question, int(INPUT_LENGTH * 0.6)-1)
    allowance = INPUT_LENGTH - size(question) - size(block) - 3
    formatter = scope[template_names[task]]
    for k in range(len(selected), 0, -1):
        budget = allowance
        while budget >= 16:
            try:
                raw_prompt = formatter(question, selected[:k], budget, tok)
                final = block + '\n\n' + raw_prompt
                length = size(final)
            except (ValueError, OverflowError, TypeError):
                break
            if length <= INPUT_LENGTH and raw_prompt != question:
                return final, {'input_tokens_reference': length, 'raw_history_count': k,
                               'question_truncated': question != original, 'profile_tokens_reference': size(block)}
            budget -= max(1, length-INPUT_LENGTH)
    if selected:
        raise ValueError('Unable to fit any official raw history within GPG-512 budget')
    raise ValueError('No official history available for GPG')


def predict(task, q, gold, request=chat):
    start = time.monotonic()
    answer_started = None
    r = {'id': str(q['id']), 'task': task, 'gold': gold, 'status': 'running', 'answer': '', 'stages': []}
    try:
        selected = retrieve(task,q)
        r['history_ids'] = [str(h.get('id','')) for h in selected]
        r['selected_history_count'] = len(selected)
        if not selected:
            raise ValueError('No historical records')
        def stage(messages, **kwargs):
            name = ('digestion','profile','answer')[len(r['stages'])]
            entry = {'stage':name, 'prompt_sha256':hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()}
            r['stages'].append(entry)
            try:
                response = request(messages, temperature=0)
                phases=response.get('nex_phases',[])
                entry.update(status='complete', usage=phases[-1].get('usage') if phases else response['usage'], phases=phases)
                text = response['choices'][0]['message']['content']
                if not text or not text.strip():
                    raise ValueError(f'Empty {name} output')
                return text
            except NexDiagnosticStop as exc:
                entry.update(status=exc.reason, phase=exc.phase_result, diagnostic=exc.diagnostic)
                raise
        with task_budget(TaskBudget(limit=None,effort=PREPARATION_EFFORT)):
            gpg = create('gpg', task=f'lamp{task}', chat=stage)
            gpg.update(history_for(task,selected))
            r.update(digest=gpg.digest, generated_profile=gpg.profile)
            prompt, metadata = pack(task,q,selected,gpg.profile,gpg.digest)
            r.update(metadata, prompt=prompt)
        with task_budget():
            answer_started = time.monotonic()
            r['answer'] = stage([{'role':'user','content':prompt}]).strip()
            r['status'] = 'complete'
    except NexDiagnosticStop as e:
        r.update(status=e.reason, error=str(e), failure_stage=r['stages'][-1]['stage'] if r['stages'] else 'setup',diagnostic=e.diagnostic)
    except Exception as e:
        failed_stage=r['stages'][-1]['stage'] if r['stages'] else 'setup'
        r.update(status='client_error', error=f'{type(e).__name__}: {str(e)[:250]}',failure_stage=failed_stage,
                 diagnostic={'category':'method_or_input_error','phase':failed_stage,'exception_type':type(e).__name__})
    r['preparation_cost'] = token_cost([s for s in r['stages'] if s['stage']!='answer'])
    r['answer_cost'] = token_cost([s for s in r['stages'] if s['stage']=='answer'])
    r['generated_tokens'] = r['preparation_cost']['completion_tokens']+r['answer_cost']['completion_tokens']
    end=time.monotonic()
    r['seconds'] = end-start
    r['answer_cost']['wall_seconds']=end-answer_started if answer_started is not None else 0
    r['preparation_cost']['wall_seconds']=max(0,r['seconds']-r['answer_cost']['wall_seconds'])
    return r


def verify_inputs():
    count = 0
    summary = Counter()
    for task, rows in cohort():
        for q, _ in rows:
            selected = retrieve(task,q)
            prompt, meta = pack(task,q,selected,'A consistent preference for concise and evidence-backed answers. '*50,'A distinctive writing pattern. '*50)
            assert meta['input_tokens_reference'] <= INPUT_LENGTH and meta['raw_history_count'] >= 1 and prompt
            count += 1
            summary[(task,meta['question_truncated'],meta['raw_history_count'])] += 1
    assert count == 1200
    print('Input preflight:', count, 'valid; (task, shortened_query, raw_history_count):', dict(summary), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--check-inputs', action='store_true')
    args = parser.parse_args()
    if args.check_inputs:
        verify_inputs(); return
    if not args.output or not 1 <= args.workers <= 10:
        parser.error('Supply --output and 1–10 workers')
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    cfg = {'method':'gpg-512','manifest_sha256':sha(MANIFEST),'split':'time/test','retriever':'official BM25','k':K,
           'model':'nex-n2.5-pro','reasoning_effort':'medium','preparation_effort':PREPARATION_EFFORT,'temperature':0,'fallback':None,'workers':args.workers,
           'task_token_budget':TASK_TOKEN_BUDGET,'input_budget':INPUT_LENGTH,'profile_budget':PROFILE_BUDGET,'budget_scope':'answer_only','preparation_token_budget':None,
           'scoring_token_cap':None,'track_retry_limit':1,'question_policy':'upstream 60% query budget only when necessary; question never silently removed','count':1200}
    config_path = out/'experiment.json'
    if config_path.exists():
        if json.loads(config_path.read_text(encoding='utf-8')) != cfg:
            raise ValueError('Experiment configuration mismatch')
    else:
        config_path.write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    work = [(t,q,g) for t, rows in cohort() for q,g in rows]
    expected = {(t,str(q['id'])) for t,q,_ in work}
    path = out/'predictions.jsonl'
    saved = {}
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            r = json.loads(line); key = (r['task'],r['id'])
            if key in saved or key not in expected:
                raise ValueError('Duplicate or unmapped saved prediction')
            saved[key] = r
    print('Starting GPG-512', len(saved), '/',len(expected),flush=True)
    with path.open('a', encoding='utf-8') as stream, ThreadPoolExecutor(args.workers) as pool:
        futures = [pool.submit(predict,t,q,g) for t,q,g in work if (t,str(q['id'])) not in saved]
        for f in as_completed(futures):
            r = f.result(); saved[(r['task'],r['id'])] = r
            stream.write(json.dumps(r,ensure_ascii=False)+'\n'); stream.flush()
            print(f"[{len(saved)}/1200] task={r['task']} id={r['id']} {r['status']} tokens={r['generated_tokens']} seconds={r['seconds']:.2f}",flush=True)
    assert set(saved) == expected
    retry_path = out/'predictions.retry.jsonl'
    retried = {}
    if retry_path.exists():
        for line in retry_path.read_text(encoding='utf-8').splitlines():
            r=json.loads(line); key=(r['task'],r['id'])
            if key in retried or key not in saved or not retryable(saved[key]):
                raise ValueError('Invalid saved retry')
            retried[key]=r
    pending=[(t,q,g) for t,q,g in work if (t,str(q['id'])) not in retried and retryable(saved[(t,str(q['id']))])]
    if pending:
        with retry_path.open('a',encoding='utf-8') as stream,ThreadPoolExecutor(args.workers) as pool:
            futures=[pool.submit(predict,t,q,g) for t,q,g in pending]
            for f in as_completed(futures):
                r=f.result();key=(r['task'],r['id'])
                r.update(attempt=2,first_status=saved[key]['status'])
                retried[key]=r
                stream.write(json.dumps(r,ensure_ascii=False)+'\n');stream.flush()
    final={**saved,**retried}
    from run_lamp_formal import metrics, strict_metrics
    results = {}
    for task in sorted({t for t,_ in expected}):
        rows = [r for (t,_),r in final.items() if t==task]
        try:
            results[str(task)] = metrics(task,rows)
        except Exception as exc:
            results[str(task)] = dict(strict_metrics(task,rows), scoring_error=f'{type(exc).__name__}: {str(exc)[:200]}')
        stages=[s for r in rows for s in r['stages']]
        results[str(task)].update(preparation_cost=token_cost([s for s in stages if s['stage']!='answer']),
                                  answer_cost=token_cost([s for s in stages if s['stage']=='answer']),
                                  failure_attribution=dict(Counter(r['diagnostic']['category'] for r in rows if r.get('diagnostic'))))
        for key in ('preparation_cost','answer_cost'):
            results[str(task)][key]['wall_seconds_sum']=sum(r[key]['wall_seconds'] for r in rows)
        results[str(task)]['retry_attempts']=sum(t==task for t,_ in retried)
        results[str(task)]['retry_recovered']=sum(t==task and r['status']=='complete' for (t,_),r in retried.items())
    (out/'scores.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('FINISHED',json.dumps(dict(Counter(r['status'] for r in final.values()))),flush=True)


if __name__ == '__main__':
    main()
