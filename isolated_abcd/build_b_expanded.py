"""Freeze an independent, answer-blind replication cohort for experiment B."""
import json
from collections import defaultdict

from dataio import BASE, digest, split, twin


def main():
    seed = 'abcd-b-expanded-20261125'
    old = json.loads((BASE / 'b_min_validation_v1.json').read_text(encoding='utf8'))
    excluded = {key.split('/')[0] for keys in old['samples'].values() for key in keys}
    candidates = defaultdict(dict)
    for pid, _, q, _, _ in twin(BASE.parent / 'data', True):
        if pid not in excluded:
            key = f'{pid}/{q["qid"]}/' + digest(q['text'])[:12]
            if key in candidates[pid]:
                raise ValueError(f'Duplicate paired sample: {key}')
            candidates[pid][key] = len(q['options'])
    rank = lambda key: digest(seed + '|' + key)
    samples = {}
    for fold, n in [('dev', 72), ('cal', 48), ('test', 72)]:
        people = sorted((p for p, keys in candidates.items() if split(p) == fold and len(keys) >= 6), key=rank)[:n]
        assert len(people) == n and not set(people) & excluded
        selected = []
        for p in people:
            keys = candidates[p]
            order = sorted(keys, key=rank)
            chosen = []
            for group in (lambda k: k == 2, lambda k: 3 <= k <= 4, lambda k: k >= 5):
                chosen.extend([k for k in order if group(keys[k])][:2])
            chosen.extend(k for k in order if k not in chosen)
            selected.extend(chosen[:6])
        assert len(set(selected)) == n * 6
        samples[fold] = selected
    path = BASE / 'b_expanded_validation_v2.json'
    with path.open('x', encoding='utf8') as f:
        json.dump({'version': 2, 'seed': seed, 'selection': 'New PIDs only; six pairs per user; hash-ranked, two each from binary, 3–4 and 5+ options where feasible; no answer labels used', 'samples': samples}, f, ensure_ascii=False, indent=2)
    print({fold: len(keys) for fold, keys in samples.items()})


if __name__ == '__main__':
    main()
