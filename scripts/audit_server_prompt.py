import json,urllib.request,urllib.error
from pathlib import Path
import sys
sys.path.insert(0,'scripts')
from sii import BASE,load_key
op=urllib.request.build_opener(urllib.request.ProxyHandler({}))
out=[]
for path in ['/get_model_info','/get_server_info','/v1/models']:
 url=BASE.rsplit('/v1',1)[0]+path
 try:
  req=urllib.request.Request(url,headers={'Authorization':'Bearer '+load_key()})
  with op.open(req,timeout=15) as r:
   j=json.load(r)
  keys=['model_path','tokenizer_path','chat_template','reasoning_parser','version','served_model_name','is_generation']
  row={'path':path,'status':200,'available_keys':list(j) if isinstance(j,dict) else [],'metadata':{k:j[k] for k in keys if isinstance(j,dict) and k in j}}
 except Exception as e:row={'path':path,'status':getattr(e,'code',type(e).__name__)}
 out.append(row)
Path('results/nex-server-prompt-audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=True))
