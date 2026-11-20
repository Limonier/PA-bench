"""Report coverage from saved results, without treating partial runs as complete."""
import json
from pathlib import Path
from run_context import RESULTS
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for t,n in [(1,1500),(2,1557),(3,2500),(4,1800),(5,1500),(7,1500)]:
    f=RESULTS/f'lamp-{t}-test.json'
    done=len(json.loads(f.read_text(encoding='utf-8'))['golds']) if f.exists() else 0
    rows.append({'benchmark':f'LaMP-{t}','done':done,'total':n})
f=ROOT/'results/fingertip-predictions.jsonl'
rows.append({'benchmark':'FingerTip','done':len(f.read_text(encoding='utf-8').splitlines()) if f.exists() else 0,'total':1000})
for f in sorted(RESULTS.glob('mega-*.jsonl')):
    rows.append({'benchmark':f.stem,'done':len(f.read_text(encoding='utf-8').splitlines())})
print(json.dumps(rows,ensure_ascii=False,indent=2))
(ROOT/'results/progress.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
