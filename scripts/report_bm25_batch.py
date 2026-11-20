"""Render scores beside the paper reference, with comparability qualifications."""
import json
from pathlib import Path
P=Path(__file__).resolve().parents[1]/'results/bm25-official-six-branch-10'
scores=json.loads((P/'scores.json').read_text(encoding='utf-8'));refs=json.loads((P/'official-reference.json').read_text(encoding='utf-8'))
lines=['# Nex BM25 reasoning-mode comparison','','Design: time-based test split, 10 fixed random items per each of tasks 1/2/3/4/5/7; BM25 k=4; same history/prompt per mode; temperature=0; no max_tokens, max_completion_tokens, reasoning budget, stop or client timeout. Seed 20261120.','', 'These are sample diagnostics, not full-benchmark reproduction or evidence of superiority.','','|Task|Mode|Completed|Accuracy|Macro F1|MAE|RMSE|ROUGE-1|ROUGE-L|Mean seconds|Reasoning tokens|','|---|---|---|---|---|---|---|---|---|---|---|']
def v(r,k):return f'{r[k]:.4f}' if k in r else '—'
for r in scores:
 lines.append('|'+ '|'.join([str(r['task']),r['effort'],f'{r["completed"]}/10']+[v(r,k) for k in ['accuracy','macro_f1','MAE','RMSE','rouge1','rougeL','mean_seconds']]+[str(r.get('reasoning_tokens','—'))])+'|')
lines+=['','## Official full-test reference (Table 4, time split)','','|Task|Model|Metrics|','|---|---|---|']
for model in ['GPT-3.5','FlanT5-XXL']:
 for t,r in refs[model].items():lines.append(f'|{t}|{model}|'+', '.join(f'{k}={val}' for k,val in r.items())+'|')
lines+=['','Source: '+refs['source'],'','## Differences']+['- '+x for x in refs['comparison_caveats']]
(P/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print('Report saved; completed calls:',sum(r['completed'] for r in scores),'/180')
