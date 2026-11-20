import json,sys,urllib.request
from pathlib import Path
sys.path.insert(0,'scripts');from sii import BASE,load_key
req=urllib.request.Request(BASE.rsplit('/v1',1)[0]+'/openapi.json',headers={'Authorization':'Bearer '+load_key()})
try:
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=15) as r:j=json.load(r)
 for k,v in j.get('components',{}).get('schemas',{}).items():
  if k in ['TokenizeReqInput','ChatCompletionRequest']:
   fields=v.get('properties',{});print(k,[(a,fields[a]) for a in fields if 'template' in a or 'message' in a or 'token' in a or 'prompt' in a][:12])
 print('tokenize_paths',[k for k in j.get('paths',{}) if 'tokeniz' in k])
except Exception as e:print(type(e).__name__)
