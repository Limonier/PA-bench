"""Official LaMP templates, six branches x20, Nex medium, no generation limits."""
import os,json,random
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/('lamp-official-bm25-medium-20-'+datetime.now().strftime('%Y%m%dT%H%M%S'))
os.environ['MODE_COMPARISON_DIR']=str(OUT)
os.environ['SII_REASONING_EFFORT']='medium'
from compare_bm25_modes import run
SEED=20261120
TASKS=[1,2,3,4,5,7]

def main():
 selected={};manifest=[]
 for task in TASKS:
  all_q=json.loads((ROOT/f'data/lamp/LaMP_{task}/test/test_questions.json').read_text(encoding='utf-8'))
  indices=sorted(random.Random(SEED+task).sample(range(len(all_q)),20))
  selected[task]=[all_q[i] for i in indices]
  manifest.extend({'task':task,'index':i,'id':all_q[i]['id']} for i in indices)
 assert len(manifest)==120 and len({(x['task'],x['id']) for x in manifest})==120
 config={'seed':SEED,'split':'time/test','model':'nex-n2.5-pro','reasoning_effort':'medium','retriever':'official BM25','k':4,'template':'official task-specific formatter; selected history preserved','generation_token_limit':None,'client_timeout':None,'items':manifest}
 (OUT/'experiment.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
 print('Experiment directory: '+str(OUT),flush=True)
 rows=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  fs=[pool.submit(run,t,'medium',selected[t][i]) for i in range(20) for t in TASKS]
  for f in as_completed(fs):
   row=f.result();rows.append(row)
   (OUT/'summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
   print(json.dumps({'finished':len(rows),'total':120,**row},ensure_ascii=True),flush=True)
 print('All 120 requests returned; inspect completion status before scoring.',flush=True)

if __name__=='__main__':main()
