import json,sys,urllib.request
from pathlib import Path
sys.path.insert(0,'scripts');from sii import BASE,load_key
prompt=json.loads(Path('results/official-prompt-examples.json').read_text(encoding='utf-8'))[0]['prompt']
body={'model':'nex-n2.5-pro','reasoning_effort':'none','temperature':0,'messages':[{'role':'user','content':prompt}],'return_prompt_token_ids':True,'stream':False}
req=urllib.request.Request(BASE+'/chat/completions',data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+load_key(),'Content-Type':'application/json'})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=None) as r:response=json.load(r)
audit=json.loads(Path('results/nex-rendered-input-audit.json').read_text(encoding='utf-8'))
expected=next(x['tokens'] for x in audit if x['task']==1 and x['effort']=='none')
actual=response.get('prompt_token_ids')
result={'status':'checked' if actual is not None else 'prompt_ids_not_returned','response_keys':list(response),'actual_completion_prompt_ids_match_tokenize':actual==expected if actual is not None else None,'request':body,'response':response}
Path('results/nex-actual-completion-input-audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k not in ['request','response']}))
