"""Row identities preserve all official test rows, including exact duplicates."""
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'upstream/FingerTip-20K'


def rows(name):
    with (SOURCE / name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def tasks(sampled=False):
    full = [dict(row, id=f'suggestion:{index:06d}', row_index=index,
                 legacy_id=row['user_id'] + '/' + row['time'])
            for index, row in enumerate(rows('test_suggestion.csv'))]
    if not sampled:
        return full
    buckets = collections.defaultdict(collections.deque)
    for row, task in zip(rows('test_suggestion.csv'), full):
        buckets[json.dumps(row, sort_keys=True)].append(task)
    selected = []
    fields = rows('test_suggestion.csv')[0].keys()
    for row in rows('sampled_test_suggestion.csv'):
        # Sample file adds a level column; CLI screenshot count controls level.
        group = buckets[json.dumps({k: row[k] for k in fields}, sort_keys=True)]
        if not group:
            raise ValueError('Official sampled row is absent from full test set')
        selected.append(group.popleft())
    return selected


def align_predictions(records, task_rows, *, partial=False):
    expected = {row['id']: row for row in task_rows}
    if all(row.get('id') in expected for row in records):
        if all('status' in row for row in records):
            # Runner resumes append attempts. Keep the latest successful
            # attempt per CSV row, retaining the source JSONL unchanged.
            selected = {}
            for record in records:
                if record['status'] == 'complete' or record['id'] not in selected:
                    selected[record['id']] = record
            aligned = list(selected.values())
        else:
            aligned = records
    else:
        # A legacy occurrence can only be assigned safely when every duplicate
        # refers to an identical CSV row. Never deduplicate or copy a prediction.
        buckets = collections.defaultdict(collections.deque)
        for task in task_rows:
            buckets[task['legacy_id']].append(task)
        for group in buckets.values():
            canonical = [{k: v for k, v in task.items() if k not in ('id', 'row_index')} for task in group]
            if any(row != canonical[0] for row in canonical):
                raise ValueError('Ambiguous legacy identities: duplicate source rows differ')
        aligned = []
        for record in records:
            group = buckets.get(record.get('id'))
            if not group:
                raise ValueError('Unknown or excess legacy prediction: ' + str(record.get('id')))
            task = group.popleft()
            aligned.append(dict(record, id=task['id'], legacy_id=record['id'], row_index=task['row_index']))
    ids = [row['id'] for row in aligned]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate row-level prediction IDs')
    if not set(ids) <= set(expected) or (not partial and set(ids) != set(expected)):
        raise ValueError('Prediction coverage does not match the official CSV rows')
    if any(not isinstance(row.get('prediction'), str) or not row['prediction'].strip() for row in aligned):
        raise ValueError('Blank or invalid predictions cannot be silently scored')
    # Official llm_client strips surrounding whitespace before computing metrics.
    aligned = [dict(row, prediction=row['prediction'].strip()) for row in aligned]
    return sorted(aligned, key=lambda row: expected[row['id']]['row_index'])
