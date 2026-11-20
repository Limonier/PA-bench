"""Official SR2/Sim1 formula; SR1 judge is not supplied by the public release."""
import argparse
import csv
import json
from pathlib import Path
from run_context import RESULTS
import numpy as np
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--partial',action='store_true')
a=p.parse_args()
with (ROOT/'upstream/FingerTip-20K/test_suggestion.csv').open(encoding='utf-8-sig') as f:
    gold={r['user_id']+'/'+r['time']:r['intentDescription'] for r in csv.DictReader(f)}
pred=[json.loads(s) for s in (ROOT/'results/fingertip-predictions.jsonl').read_text(encoding='utf-8').splitlines()]
ids=[r['id'] for r in pred]
assert len(ids)==len(set(ids)), 'Duplicate predictions'
assert set(ids)<=set(gold), 'Unknown prediction IDs'
if not a.partial: assert set(ids)==set(gold), 'Incomplete predictions; use --partial only for diagnostics'
model=SentenceTransformer(str(ROOT/'data/models/paraphrase-multilingual-MiniLM-L12-v2'))
scores=[]
for r in pred:
    texts=[gold[r['id']],r['prediction']]
    emb=model.encode(texts)
    cosine=np.dot(emb[0],emb[1])/(np.linalg.norm(emb[0])*np.linalg.norm(emb[1]))
    scores.append(round((fuzz.ratio(*texts)/100+float(cosine))/2,2))
result={'complete':set(ids)==set(gold),'n':len(ids),'total':len(gold),'SR2_Sim1':float(np.mean(scores)),
    'SR1':None,'SR1_note':'Public release lacks complete judge implementation; not fabricated.'}
print(json.dumps(result))
(ROOT/'results/fingertip-score.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
