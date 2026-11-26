"""Score the paired sample using LaMP metric definitions and underlying metric libraries."""
import json,math,collections
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,mean_squared_error
from rouge_score import rouge_scorer,scoring
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/bm25-official-six-branch-10'
manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
labels=['sci-fi','based on a book','comedy','action','twist ending','dystopia','dark comedy','classic','psychology','fantasy','romance','thought-provoking','social commentary','violence','true story']
report=[]
for task in [1,2,3,4,5,7]:
 for effort in ['none','medium','high']:
  rows=[];missing=[]
  for item in manifest['items']:
   if item['task']!=task:continue
   p=OUT/f'task{task}-{item["id"]}-{effort}.json'
   if not p.exists():missing.append(item['id']);continue
   row=json.loads(p.read_text(encoding='utf-8'))
   if row['status']!='complete':missing.append(item['id']);continue
   rows.append(row)
  result={'task':task,'effort':effort,'completed':len(rows),'expected':10,'missing':missing}
  if rows:
   pred=[r['answer'].strip() for r in rows];gold=[str(r['gold']).strip() for r in rows]
   if task in [1,2]:
    allowed=['[1]','[2]'] if task==1 else labels
    result['invalid_labels']=sum(x not in allowed for x in pred)
    result['accuracy']=accuracy_score(gold,pred)
    if task==2:result['macro_f1']=f1_score(gold,pred,labels=labels,average='macro',zero_division=0)
   elif task==3:
    valid=[]
    for a,b in zip(pred,gold):
     try:valid.append((float(a),float(b)))
     except ValueError:pass
    result['invalid_ratings']=len(pred)-len(valid)
    if valid:
     a,b=zip(*valid);result['MAE']=mean_absolute_error(b,a);result['RMSE']=math.sqrt(mean_squared_error(b,a))
   else:
    scorer=rouge_scorer.RougeScorer(['rouge1','rougeL'],use_stemmer=False);agg=scoring.BootstrapAggregator()
    for a,b in zip(pred,gold):agg.add_scores(scorer.score(b,a))
    np.random.seed(20261120);v=agg.aggregate();result.update(rouge1=v['rouge1'].mid.fmeasure,rougeL=v['rougeL'].mid.fmeasure)
   result['mean_seconds']=sum(r['seconds'] for r in rows)/len(rows)
   result['reasoning_tokens']=sum((r.get('usage') or {}).get('reasoning_tokens') or 0 for r in rows)
  report.append(result)
(OUT/'scores.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
