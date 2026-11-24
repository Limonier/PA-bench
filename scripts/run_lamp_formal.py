"""Full time-test paired experiment; dev-only retrieval selection; fixed Nex policy."""
import os
os.environ['USE_TORCH']='0' # Tokenizer-only environment; no PyTorch model is used for BM25/Recency.
import json,sqlite3,random,time,math,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime
from official_lamp_inputs import ROOT,build
from timed_nex import answer,MODEL,PRIMARY_EFFORT,FALLBACK_EFFORT,THINK_SECONDS,ANSWER_SECONDS
from track_retry import retryable
from sklearn.metrics import accuracy_score,f1_score,mean_absolute_error,mean_squared_error
from rouge_score import rouge_scorer,scoring
import numpy as np
TASKS=[1,2,3,4,5,7];LABELS=['sci-fi','based on a book','comedy','action','twist ending','dystopia','dark comedy','classic','psychology','fantasy','romance','thought-provoking','social commentary','violence','true story']
CANDIDATES=[(r,k) for r in ['bm25','recency','contriever'] for k in [1,2,4]]

def strict_metrics(task,rows):
 good=[r for r in rows if r.get('status')=='complete'];a=[r['answer'].strip() for r in good];b=[str(r['gold']).strip() for r in good]
 abandoned=sum(r.get('status')=='budget_exceeded' for r in rows)
 result={'total':len(rows),'completed':len(good),'budget_abandoned':abandoned,'failed':len(rows)-len(good)-abandoned,'fallback':sum(r.get('fallback',False) for r in rows)}
 if not good:return result
 if task in [1,2]:
  labels=['[1]','[2]'] if task==1 else LABELS
  result.update(accuracy=sum(x==y for x,y in zip(a,b))/len(rows),invalid=sum(x not in labels for x in a))
  if task==2:result['macro_f1']=f1_score(b,a,labels=labels,average='macro',zero_division=0)
 elif task==3:
  pairs=[]
  for x,y in zip(a,b):
   try:pairs.append((float(x),float(y)))
   except ValueError:pass
  result['invalid']=len(a)-len(pairs)
  errors=[]
  for x,y in zip(a,b):
   try:errors.append(abs(float(x)-float(y)))
   except ValueError:errors.append(max(abs(1-float(y)),abs(5-float(y))))
  errors.extend([4.0]*(len(rows)-len(good)))
  result.update(MAE=sum(errors)/len(rows),RMSE=math.sqrt(sum(e*e for e in errors)/len(rows)))
 else:
  scorer=rouge_scorer.RougeScorer(['rouge1','rougeL'],use_stemmer=False);agg=scoring.BootstrapAggregator()
  for x,y in zip(a,b):agg.add_scores(scorer.score(y,x))
  for _ in range(len(rows)-len(good)):agg.add_scores(scorer.score('missing',''))
  np.random.seed(20261120);v=agg.aggregate();result.update(rouge1=v['rouge1'].mid.fmeasure,rougeL=v['rougeL'].mid.fmeasure)
 return result

def metrics(task,rows):
 raw=strict_metrics(task,rows)
 if task not in [1,2,3]:return raw
 from retrieval_client import map_label
 allowed=['[1]','[2]'] if task==1 else (LABELS if task==2 else ['1','2','3','4','5'])
 mapped=[];count=0
 for row in rows:
  item=dict(row)
  if item.get('status')=='complete':
   mapping=map_label(item['answer'],allowed);item['answer']=mapping['label'];count+=bool(mapping['mapped'])
  mapped.append(item)
 result=strict_metrics(task,mapped);result.update(raw_strict=raw,mapped_count=count,mapping_config={'model':'roberta-large','layer':17,'idf':False,'rescale':False,'tokenizer':'slow','similarity':'F1','paper_exact_parameters_known':False})
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--smoke',action='store_true');p.add_argument('--validation-only',action='store_true');p.add_argument('--retrievers',nargs='+',choices=['bm25','recency','contriever'],default=['bm25','recency','contriever']);args=p.parse_args();candidates_to_run=[(r,k) for r in args.retrievers for k in [1,2,4]]
 out=Path(args.output) if args.output else ROOT/'results'/('lamp-formal-paired-'+datetime.now().strftime('%Y%m%dT%H%M%S'));out.mkdir(parents=True,exist_ok=True)
 db=sqlite3.connect(out/'predictions.sqlite');db.execute('create table if not exists predictions(task integer,split text,arm text,id text,result text,primary key(task,split,arm,id))');db.execute('create table if not exists settings(task integer primary key,config text)');db.commit()
 config={'model':MODEL,'primary_effort':PRIMARY_EFFORT,'fallback_effort':None,'split':'time','tasks':TASKS,'input_budget':512,'tokenizer':'google/flan-t5-base','input_builder':'exact upstream create_prompt_generator + tokenizer truncation','validation_per_task':100,'validation_seed':20261120,'candidates':candidates_to_run,'validation_only':args.validation_only,'retrieval_checkpoint':'facebook/contriever','generation_token_cap':16384,'budget_scope':'per_task_completion','track_retry_limit':1,'smoke':args.smoke,'selection_metric':{'1':'accuracy','2':'accuracy','3':'negative MAE','4':'ROUGE-1','5':'ROUGE-1','7':'ROUGE-1'},'tie_break':'fixed candidate order: bm25, recency, contriever; each k1/k2/k4','postprocessing':'raw strict scores plus BERTScore nearest-label mapping; roberta-large L17 no-idf no-rescale slow tokenizer, an explicit reproducible choice not verified as author settings'}
 path=out/'experiment.json'
 if path.exists() and json.loads(path.read_text(encoding='utf-8'))!=json.loads(json.dumps(config)):raise RuntimeError('Experiment config mismatch')
 path.write_text(json.dumps(config,indent=2),encoding='utf-8');print('Experiment: '+str(out),flush=True)
 def execute(task,split,arm,q,gold,ret=None,k=0):
  try:
   prompt,n=build(task,q,ret,k);r=answer(prompt);r.update(prompt=prompt,input_tokens_reference=n,retriever=ret,k=k)
  except Exception as exc:r={'status':'client_error','error':type(exc).__name__,'answer':''}
  r.update(task=task,id=q['id'],split=split,arm=arm,gold=gold)
  return r
 def batch(task,split,arm,qs,golds,ret=None,k=0):
  existing={x[0]:json.loads(x[1]) for x in db.execute('select id,result from predictions where task=? and split=? and arm=?',(task,split,arm))}
  (out/'progress.json').write_text(json.dumps({'task':task,'split':split,'arm':arm,'saved':len(existing),'expected':len(qs),'stage':'starting','updated':time.time()}),encoding='utf-8')
  for retry_round in range(2):
   with ThreadPoolExecutor(4) as pool:
    fs=[pool.submit(execute,task,split,arm,q,golds[str(q['id'])],ret,k) for q in qs if str(q['id']) not in existing or retryable(existing[str(q['id'])],existing[str(q['id'])].get('attempt',1))]
    for f in as_completed(fs):
     r=f.result();key=str(r['id'])
     if key in existing:r.update(attempt=2,first_attempt=existing[key])
     existing[key]=r;db.execute('insert or replace into predictions values(?,?,?,?,?)',(task,split,arm,key,json.dumps(r,ensure_ascii=False)));db.commit()
     (out/'progress.json').write_text(json.dumps({'task':task,'split':split,'arm':arm,'saved':len(existing),'expected':len(qs),'updated':time.time()}),encoding='utf-8')
   if all(str(q['id']) in existing and existing[str(q['id'])]['status'] in ('complete','budget_exceeded') for q in qs):break
  return [existing[str(q['id'])] for q in qs]
 for task in TASKS:
  qs=json.loads((ROOT/f'data/lamp/LaMP_{task}/dev/dev_questions.json').read_text(encoding='utf-8'));g=json.loads((ROOT/f'data/lamp/LaMP_{task}/dev/dev_outputs.json').read_text(encoding='utf-8'))['golds'];g={str(x['id']):x['output'] for x in g}
  qs=random.Random(20261120+task).sample(qs,2 if args.smoke else 100)
  saved=db.execute('select config from settings where task=?',(task,)).fetchone()
  if saved:chosen=json.loads(saved[0])
  else:
   candidates=[]
   for ret,k in (candidates_to_run[:1] if args.smoke else candidates_to_run):
    rows=batch(task,'dev',f'{ret}-k{k}',qs,g,ret,k);m=metrics(task,rows);print(json.dumps({'task':task,'candidate':[ret,k],'dev_metrics':m}),flush=True);(out/f'task{task}-{ret}-k{k}-dev-scores.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
    if m['completed']!=len(qs):continue
    score=m.get('accuracy') if task in [1,2] else (-m['MAE'] if task==3 else m['rouge1'])
    candidates.append({'retriever':ret,'k':k,'score':score,'metrics':m})
   if not candidates:raise RuntimeError(f'No fully valid validation configuration for task {task}; fix failed records and resume')
   chosen=max(candidates,key=lambda x:x['score']);db.execute('insert or replace into settings values(?,?)',(task,json.dumps(chosen)));db.commit()
  if args.validation_only:continue
  qs=json.loads((ROOT/f'data/lamp/LaMP_{task}/test/test_questions.json').read_text(encoding='utf-8'));g=json.loads((ROOT/f'data/lamp/LaMP_{task}/test/test_outputs.json').read_text(encoding='utf-8'))['golds'];g={str(x['id']):x['output'] for x in g}
  if args.smoke:qs=qs[:1]
  arms=[('no_history',None,0),('history',chosen['retriever'],chosen['k'])]
  existing={(a,i):json.loads(v) for a,i,v in db.execute('select arm,id,result from predictions where task=? and split=?',(task,'test'))}
  for retry_round in range(2):
   with ThreadPoolExecutor(4) as pool:
    fs=[pool.submit(execute,task,'test',arm,q,g[str(q['id'])],ret,k) for q in qs for arm,ret,k in arms if (arm,str(q['id'])) not in existing or retryable(existing[(arm,str(q['id']))],existing[(arm,str(q['id']))].get('attempt',1))]
    for f in as_completed(fs):
     r=f.result();key=(r['arm'],str(r['id']))
     if key in existing:r.update(attempt=2,first_attempt=existing[key])
     existing[key]=r
     db.execute('insert or replace into predictions values(?,?,?,?,?)',(task,'test',r['arm'],str(r['id']),json.dumps(r,ensure_ascii=False)));db.commit()
     (out/'progress.json').write_text(json.dumps({'task':task,'split':'test','arm':'paired','saved':len(existing),'expected':2*len(qs),'updated':time.time()}),encoding='utf-8')
   if all(existing.get((arm,str(q['id'])),{}).get('status')=='complete' for q in qs for arm,_,_ in arms):break
  for arm,_,_ in arms:
   rows=[existing[(arm,str(q['id']))] for q in qs];m=metrics(task,rows)
   if m['total']!=len(qs):raise RuntimeError(f'Incomplete test coverage for task {task} arm {arm}; resume required')
   (out/f'task{task}-{arm}-scores.json').write_text(json.dumps(m,indent=2),encoding='utf-8');print(json.dumps({'task':task,'arm':arm,'metrics':m}),flush=True)
 print('All stages finished; verify coverage and invalid counts before accepting scores.',flush=True)
if __name__=='__main__':main()
