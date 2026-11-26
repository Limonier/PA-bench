"""Isolated GPU worker for the official Contriever algorithm and BERTScore label mapping."""
import sys,os,json,hashlib,sqlite3
from pathlib import Path
WIRE=sys.stdout;sys.stdout=sys.stderr
os.environ['USE_TORCH']='1'
import torch
from transformers import AutoModel,AutoTokenizer
ROOT=Path(__file__).resolve().parents[1]
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
models={}
cache=sqlite3.connect(ROOT/'data/retrieval-rank-cache.sqlite');cache.execute('create table if not exists ranks(key text primary key,indices text)')
def rank(query,corpus):
 key=hashlib.sha256(json.dumps([query,corpus],ensure_ascii=False).encode()).hexdigest()
 old=cache.execute('select indices from ranks where key=?',(key,)).fetchone()
 if old:return json.loads(old[0])
 if 'contriever' not in models:
  tok=AutoTokenizer.from_pretrained(str(ROOT/'data/models/contriever'),local_files_only=True);model=AutoModel.from_pretrained(str(ROOT/'data/models/contriever'),local_files_only=True).to(DEVICE).eval();models['contriever']=(tok,model)
 tok,model=models['contriever']
 def embed(texts):
  x=tok(texts,padding=True,truncation=True,return_tensors='pt').to(DEVICE)
  with torch.no_grad():h=model(**x).last_hidden_state
  return h.masked_fill(~x['attention_mask'][...,None].bool(),0.0).sum(1)/x['attention_mask'].sum(1)[...,None]
 q=embed([query]);scores=[]
 for i in range(0,len(corpus),4):scores.extend((q@embed(corpus[i:i+4]).T).squeeze(0).tolist())
 indices=torch.topk(torch.tensor(scores),min(4,len(scores))).indices.tolist()
 cache.execute('insert or replace into ranks values(?,?)',(key,json.dumps(indices)));cache.commit();return indices

def map_label(text,labels):
 if text.strip() in labels:return {'label':text.strip(),'mapped':False}
 if 'bertscore' not in models:
  from bert_score import BERTScorer
  models['bertscore']=BERTScorer(model_type=str(ROOT/'data/models/roberta-large'),num_layers=17,idf=False,rescale_with_baseline=False,use_fast_tokenizer=False,device=DEVICE)
 _,_,scores=models['bertscore'].score([text]*len(labels),labels)
 values=scores.tolist();return {'label':labels[max(range(len(labels)),key=lambda i:values[i])],'mapped':True,'F1':values,'model':'roberta-large','layer':17,'idf':False,'rescale':False,'tokenizer':'slow','paper_exact_parameters_known':False}
for line in sys.stdin:
 try:
  x=json.loads(line)
  if x['op']=='rank':result={'indices':rank(x['query'],x['corpus'])}
  elif x['op']=='label':result=map_label(x['text'],x['labels'])
  elif x['op']=='health':result={'device':DEVICE,'torch':torch.__version__,'cuda':torch.cuda.is_available()}
  else:raise ValueError('Unknown operation')
  WIRE.write(json.dumps({'ok':True,'result':result})+'\n');WIRE.flush()
 except Exception as e:WIRE.write(json.dumps({'ok':False,'error':type(e).__name__,'message':str(e)[:500]})+'\n');WIRE.flush()
