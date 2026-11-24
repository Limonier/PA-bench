"""Freeze the shared initial evaluation cohort without making model calls."""
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'data/initial_eval_set_v1.json'
SEED = 20261122
TASKS = (1, 2, 3, 4, 5, 7)


def source(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sample_groups(groups, count, rng):
    """Proportional allocation with deterministic largest remainders."""
    total = sum(map(len, groups.values()))
    sizes = {k: len(v) * count // total for k, v in groups.items()}
    remainder = count - sum(sizes.values())
    # Equal remainders use a seeded random tie break, not alphabetical treatment bias.
    ties = {k: rng.random() for k in sorted(groups)}
    for k in sorted(groups, key=lambda k: (-(len(groups[k])*count % total), ties[k]))[:remainder]:
        sizes[k] += 1
    picked = sorted(i for k in sorted(groups) for i in rng.sample(sorted(groups[k]), sizes[k]))
    assert len(picked) == len(set(picked)) == count
    return picked, len(groups), sum(n > 0 for n in sizes.values())


def lamp():
    branches = {}
    for task in TASKS:
        path = ROOT / f'data/lamp/LaMP_{task}/test/test_questions.json'
        rows = load(path)
        ids = [str(r['id']) for r in rows]
        assert len(ids) == len(set(ids))
        selected = sorted(random.Random(SEED + task).sample(ids, 200))
        branches[str(task)] = {'source': source(path), 'gold_source': source(path.with_name('test_outputs.json')),
                               'population': len(ids), 'ids': selected}
    return {'split': 'time/test', 'sampling': 'uniform per branch; independent of dev tuning', 'branches': branches}


# Only study-assigned conditions; never use the response or outcome fields.
MEGA_STRATA = {
    'accuracy_nudges': ('Condition', 'RealFake'), 'affective_priming': ('emotion', 'cond'),
    'consumer_minimalism': ('order',), 'digital_certification': ('item',),
    'idea_evaluation': ('condition', 'source'), 'junk_fees': ('apartment', 'Hotel', 'CreditCard', 'Rental.Car', 'Tickets', 'Food'),
    'obedient_twins': ('absurd_1', 'absurd_2', 'absurd_3', 's_1', 's_2', 's_3'),
    'privacy': ('Group',), 'targeting_fairness': ('Segment', 'target'),
}


def mega():
    studies = {}
    for j, path in enumerate(sorted((ROOT/'data/mega/data').glob('*.parquet'))):
        name = path.name.split('-00000')[0]
        rows = pq.read_table(path, columns=['PID', 'survey_text', 'survey_json_with_human_response']).to_pylist()
        fields = MEGA_STRATA.get(name, ())
        groups = defaultdict(list)
        variants = len({r['survey_text'] for r in rows})
        for r in rows:
            if fields:
                embedded = json.loads(r['survey_json_with_human_response'])['EmbeddedData']
                key = json.dumps([embedded.get(k) for k in fields], ensure_ascii=False, sort_keys=True)
            elif 1 < variants <= 100:
                key = hashlib.sha256(r['survey_text'].encode()).hexdigest()
            else:
                key = 'all'
            groups[key].append(str(r['PID']))
        ids, n_strata, represented = sample_groups(groups, 100, random.Random(SEED + 100 + j))
        studies[name] = {'source': source(path), 'population': len(rows), 'ids': ids,
                         'stratification': list(fields) if fields else ('survey_variant' if 1 < variants <= 100 else 'uniform'),
                         'population_strata': n_strata, 'represented_strata': represented}
    assert len(studies) == 19
    return {'sampling': '100 per study; proportional allocation using design conditions when available, survey variants otherwise, uniform for unique-input designs', 'studies': studies}


def fingertip():
    directory = ROOT/'upstream/FingerTip-20K'
    path = directory/'test_suggestion.csv'
    with path.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1000
    return {'track': 1, 'screenshots': 0, 'history_limit': 20, 'sampling': 'all official rows in original order',
            'source': source(path), 'history_source': source(directory/'total.csv'), 'profile_source': source(directory/'user_profile.csv'),
            'users': len({r['user_id'] for r in rows}), 'population': len(rows), 'ids': [f'suggestion:{i:06d}' for i in range(len(rows))]}


def features(user):
    out = Counter(total=len(user['subtasks']))
    for task in user['subtasks']:
        out[f"domain:{task['domain']}"] += 1
        for skill in task['skill_tested']:
            if skill:
                out[f'skill:{skill}'] += 1
    return out


def vita():
    path = ROOT/'data/vita/tasks.json'
    users = sorted(load(path), key=lambda u: u['id'])
    counts = [features(u) for u in users]
    overall = sum(counts, Counter())
    target = {k: v*12/len(users) for k, v in overall.items()}
    rng = random.Random(SEED + 200)
    # Choose the closest population mix among seeded candidate cohorts, without rubric/gold access.
    def error(indices):
        observed = sum((counts[i] for i in indices), Counter())
        return sum(abs(observed[k]-v)/max(v, 1) for k, v in target.items())
    selected = min((tuple(sorted(rng.sample(range(len(users)), 12))) for _ in range(5000)), key=error)
    cohort = [users[i] for i in selected]
    observed = sum((counts[i] for i in selected), Counter())
    return {'source': source(path), 'sampling': '12 complete users; minimize domain and skill-count deviation across 5000 deterministic random cohorts',
            'population_users': len(users), 'population_subtasks': overall['total'], 'trials': 4,
            'population_features': dict(overall), 'selected_features': dict(observed),
            'users': [{'id': u['id'], 'subtasks': [t['subtask_id'] for t in u['subtasks']]} for u in cohort]}


def build():
    result = {'version': 1, 'seed': SEED, 'purpose': 'shared initial method comparison; disjoint from LaMP dev tuning',
              'policy': {'model': 'nex-n2.5-pro', 'reasoning_effort': 'medium', 'task_completion_tokens': 16384, 'scoring_token_cap': None, 'fallback': None},
              'lamp': lamp(), 'mega_study': mega(), 'fingertip': fingertip(), 'vita': vita()}
    assert sum(len(x['ids']) for x in result['lamp']['branches'].values()) == 1200
    assert sum(len(x['ids']) for x in result['mega_study']['studies'].values()) == 1900
    assert len(result['fingertip']['ids']) == 1000 and len(result['vita']['users']) == 12
    return result


if __name__ == '__main__':
    data = build()
    text = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True) + '\n'
    if OUTPUT.exists():
        if OUTPUT.read_text(encoding='utf-8') != text:
            raise SystemExit('Frozen manifest differs; create a new version instead of overwriting.')
    else:
        with OUTPUT.open('x', encoding='utf-8') as f:
            f.write(text)
    print(OUTPUT)
    print('LaMP=1200 Mega=1900 FingerTip=1000 Vita=', data['vita']['selected_features'])
