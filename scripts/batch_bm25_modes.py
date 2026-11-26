"""Six-branch fixed random subset, paired modes, no client/token generation limits."""
import os,json,random,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
os.environ['MODE_COMPARISON_DIR']=str(Path(__file__).resolve().parents[1]/'results/bm25-official-six-branch-10')
from compare_bm25_modes import run,OUT,ROOT
SEED=20261120
jobs=[];manifest=[]
for task in [1,2,3,4,5,7]:
 questions=json.loads((ROOT/f'data/lamp/LaMP_{task}/test/test_questions.json').read_text(encoding='utf-8'))
 indices=sorted(random.Random(SEED+task).sample(range(len(questions)),10))
 for i in indices:
  q=questions[i];manifest.append({'task':task,'index':i,'id':q['id']})
  for effort in ['none','medium','high']:jobs.append((task,effort,q))
(OUT/'manifest.json').write_text(json.dumps({'seed':SEED,'split':'time/test','k':4,'items':manifest,'token_limit':None,'client_timeout':None},indent=2),encoding='utf-8')
rows=[]
with ThreadPoolExecutor(max_workers=4) as pool:
 futures=[]
 for t,e,q in jobs:
  dst=OUT/f'task{t}-{q["id"]}-{e}.json'
  if dst.exists() and json.loads(dst.read_text(encoding='utf-8')).get('status')=='complete':continue
  futures.append(pool.submit(run,t,e,q))
 for f in as_completed(futures):
  row=f.result();rows.append(row);print(json.dumps(row,ensure_ascii=True),flush=True)
  (OUT/'completed-this-run.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print('Batch finished',flush=True)
