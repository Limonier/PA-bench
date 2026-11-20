"""Require complete prediction IDs, then call the official LaMP evaluator."""
import argparse
import json
from pathlib import Path
from run_context import RESULTS
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--task',choices=['1','2','3','4','5','7'],required=True)
a=p.parse_args()
gold=ROOT/f'data/lamp/LaMP_{a.task}/test/test_outputs.json'
pred=RESULTS/f'lamp-{a.task}-test.json'
g=json.loads(gold.read_text(encoding='utf-8'))
r=json.loads(pred.read_text(encoding='utf-8'))
ids=[x['id'] for x in r['golds']]
assert len(ids)==len(set(ids)), 'Duplicate predictions'
assert set(ids)=={x['id'] for x in g['golds']}, 'Incomplete predictions; refusing final scoring'
subprocess.run([sys.executable,'-X','utf8',str(ROOT/'upstream/LaMP/eval/eval_task.py'),
    '--golds_json',str(gold),'--preds_json',str(pred),'--task_name',f'LaMP_{a.task}',
    '--output_file',str(RESULTS/f'lamp-{a.task}-score.json')],check=True)
