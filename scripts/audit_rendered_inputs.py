"""Inspect server-rendered input tokens; no text generation and no credentials saved."""
import json,sys,urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0,'scripts');from sii import BASE,load_key
from jinja2.sandbox import SandboxedEnvironment
examples=json.loads(Path('results/official-prompt-examples.json').read_text(encoding='utf-8'))
env=SandboxedEnvironment();env.globals['raise_exception']=lambda s:(_ for _ in ()).throw(ValueError(s));template=env.from_string(Path('results/nex-public-chat-template.txt').read_text(encoding='utf-8'))
def post(path,body):
 req=urllib.request.Request(BASE+path,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+load_key(),'Content-Type':'application/json'})
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=30) as r:return json.load(r)
def audit(pair):
 ex,effort=pair;messages=[{'role':'user','content':ex['prompt']}]
 try:
  tok=post('/tokenize',{'model':'nex-n2.5-pro','messages':messages,'reasoning_effort':effort})
  tokens=tok.get('tokens',tok.get('token_ids'))
  dec=post('/detokenize',{'model':'nex-n2.5-pro','tokens':tokens,'skip_special_tokens':False})
  text=dec.get('text',dec.get('prompt',dec.get('content')))
  expected=template.render(messages=messages,tools=None,add_generation_prompt=True,reasoning_effort=effort)
  return {'task':ex['task'],'effort':effort,'status':'checked','tokens':tokens,'rendered':text,'matches_public_template':text==expected,'system_role_present':'<|im_start|>system' in text,'expected':expected if text!=expected else None}
 except Exception as e:return {'task':ex['task'],'effort':effort,'status':type(e).__name__,'http_status':getattr(e,'code',None),'error':str(e)[:150]}
with ThreadPoolExecutor(3) as pool:rows=list(pool.map(audit,[(e,m) for e in examples for m in ['none','medium','high']]))
Path('results/nex-rendered-input-audit.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps([{k:v for k,v in r.items() if k not in ['tokens','rendered','expected']} for r in rows]))
