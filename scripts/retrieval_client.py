"""Keep retrieval/scoring auxiliary models out of the Nex tokenizer process."""
import os,json,subprocess,threading,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];lock=threading.RLock();worker=None;ranks={};labels_cache={}
def call(body):
 global worker
 with lock:
  if worker is None or worker.poll() is not None:
   env=dict(os.environ,USE_TORCH='1',PYTHONUTF8='1',HTTP_PROXY='http://127.0.0.1:7890',HTTPS_PROXY='http://127.0.0.1:7890')
   for key in ['SII_API_KEY','SII_GATEWAY_TOKEN','OPENAI_API_KEY','DEEPSEEK_API_KEY']:env.pop(key,None)
   worker=subprocess.Popen([str(ROOT/'.retrieval-venv/Scripts/python.exe'),str(ROOT/'scripts/retrieval_worker.py')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf-8',env=env)
  worker.stdin.write(json.dumps(body)+'\n');worker.stdin.flush();line=worker.stdout.readline()
  if not line:raise RuntimeError('Auxiliary worker exited')
  result=json.loads(line)
  if not result['ok']:raise RuntimeError(result['error']+': '+result['message'])
  return result['result']
def rank(query,corpus):
 key=hashlib.sha256(json.dumps([query,corpus],ensure_ascii=False).encode()).hexdigest()
 with lock:
  if key not in ranks:ranks[key]=call({'op':'rank','query':query,'corpus':corpus})['indices']
  return ranks[key]
def map_label(text,labels):
 if text.strip() in labels:return {'label':text.strip(),'mapped':False}
 key=(text,tuple(labels))
 with lock:
  if key not in labels_cache:labels_cache[key]=call({'op':'label','text':text,'labels':labels})
  return labels_cache[key]
