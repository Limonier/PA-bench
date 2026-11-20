"""Bounded synthetic/real-prompt probes; no credentials in reports."""
import json
from pathlib import Path
import socket
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.request
from sii import BASE, load_key

ROOT=Path(__file__).resolve().parents[1]

def probe(case):
    name,body=case
    row={'case':name,'budget':body.get('max_tokens'),'effort':body.get('reasoning_effort')}
    start=time.monotonic()
    text=[]; reasoning=0;tools=[]
    try:
        req=urllib.request.Request(BASE+'/chat/completions',data=json.dumps({**body,'stream':True,'stream_options':{'include_usage':True}}).encode(),headers={'Authorization':'Bearer '+load_key(),'Content-Type':'application/json'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req,timeout=90) as r:
            row['headers_seconds']=round(time.monotonic()-start,3)
            for line in r:
                if not line.startswith(b'data:'):continue
                raw=line[5:].strip()
                if raw==b'[DONE]':row['done']=True;break
                if not raw:continue
                obj=json.loads(raw)
                if obj.get('usage'):row['usage']=obj['usage']
                for c in obj.get('choices',[]):
                    delta=c.get('delta',{})
                    if delta.get('content'):
                        row.setdefault('first_text_seconds',round(time.monotonic()-start,3));text.append(delta['content'])
                    if delta.get('reasoning_content'):
                        row.setdefault('first_reasoning_seconds',round(time.monotonic()-start,3));reasoning+=len(delta['reasoning_content'])
                    if delta.get('tool_calls'):tools.extend(delta['tool_calls'])
                    if c.get('finish_reason'):row['finish_reason']=c['finish_reason']
        row.update(status='received',answer=''.join(text)[:300],reasoning_chars=reasoning,tool_call_chunks=len(tools))
    except Exception as exc:
        row.update(status=type(exc).__name__,http_status=getattr(exc,'code',None),reason=str(getattr(exc,'reason',''))[:120])
    row['seconds']=round(time.monotonic()-start,3)
    return row

def main():
    common={'model':'nex-n2.5-pro','temperature':0,'max_tokens':1024}
    prompt=json.loads((ROOT/'results/fingertip-prompts.jsonl').read_text(encoding='utf-8').splitlines()[0])['prompt']
    cases=[]
    for effort in ['none','high']:
        lamp=json.loads((ROOT/'data/lamp/LaMP_2/test/test_questions.json').read_text(encoding='utf-8'))[10]['input']
        for name,txt in [('short','Reply exactly OK.'),('fingertip',prompt),('lamp2_stalled_sample',lamp)]:
            cases.append((name+'_'+effort,{**common,'reasoning_effort':effort,'messages':[{'role':'user','content':txt}]}))
    cases.append(('tool_high',{**common,'reasoning_effort':'high','messages':[{'role':'user','content':'Call ping with value OK.'}],'tools':[{'type':'function','function':{'name':'ping','description':'Return a value','parameters':{'type':'object','properties':{'value':{'type':'string'}},'required':['value']}}}],'tool_choice':'required'}))
    out=ROOT/'results/diagnostics';out.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows=list(pool.map(probe,cases))
    (out/'sii.json').write_text(json.dumps(rows,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(rows,ensure_ascii=True))

if __name__=='__main__':main()
