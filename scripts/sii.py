"""SII gateway transport following CDCA's cached-token authentication."""
import json
import os
from pathlib import Path
import urllib.request
import urllib.error
import time
import socket

LOG = Path(__file__).resolve().parents[1]/'results/diagnostics/requests.jsonl'

def record(row):
    LOG.parent.mkdir(parents=True,exist_ok=True)
    with LOG.open('a',encoding='utf-8') as f:
        f.write(json.dumps({'time':time.time(),**row},ensure_ascii=False)+'\n')

def read_response(response,stream):
    if not stream: return json.load(response)
    result={'choices':[{'index':0,'message':{'role':'assistant','content':'','reasoning_content':''},'finish_reason':None}]}
    complete=False
    started=time.monotonic()
    first=True
    for line in response:
        if time.monotonic()-started>120:
            record({'stage':'generation_deadline','reasoning_chars':len(result['choices'][0]['message']['reasoning_content']),'answer_chars':len(result['choices'][0]['message']['content'])})
            raise TimeoutError('Streaming generation exceeded 120 seconds')
        if not line.startswith(b'data:'): continue
        if first: record({'stage':'first_stream_event','seconds':time.monotonic()-started});first=False
        payload=line[5:].strip()
        if payload==b'[DONE]': complete=True;break
        if not payload:continue
        event=json.loads(payload)
        if event.get('error'): raise RuntimeError('Upstream streaming error')
        for key in ['id','model','usage']:
            if event.get(key) is not None:result[key]=event[key]
        for choice in event.get('choices',[]):
            for key in ['content','reasoning_content']:
                result['choices'][0]['message'][key]+=choice.get('delta',{}).get(key) or ''
            if choice.get('finish_reason'):result['choices'][0]['finish_reason']=choice['finish_reason']
    if not complete:raise ConnectionError('Streaming connection ended before DONE')
    return result

BASE = os.getenv('SII_BASE_URL', 'https://nex-n2-5-pro-long-context-fast.openapi-qb-ai.sii.edu.cn/v1').rstrip('/')

def load_key():
    value = os.getenv('SII_API_KEY') or os.getenv('SII_GATEWAY_TOKEN')
    if value:
        return value
    cache = Path(os.getenv('CDCA_SII_TOKEN_FILE', 'C:/npx/token.json'))
    if cache.exists():
        return json.loads(cache.read_text(encoding='utf-8'))['token']
    raise RuntimeError('SII token unavailable; set SII_API_KEY or SII_GATEWAY_TOKEN')

def request(path, body=None, timeout=120):
    if not BASE.startswith('https://'):
        raise ValueError('SII endpoint requires HTTPS')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    req = urllib.request.Request(BASE + path,
        data=None if body is None else json.dumps(body).encode('utf-8'),
        headers={'Authorization':'Bearer '+load_key(), 'Content-Type':'application/json'})
    for attempt in range(4):
        start=time.monotonic()
        record({'stage':'request_started','attempt':attempt+1,'max_tokens':(body or {}).get('max_tokens'),'reasoning_effort':(body or {}).get('reasoning_effort')})
        try:
            with opener.open(req, timeout=timeout) as response:
                headers=time.monotonic()
                record({'stage':'headers_received','attempt':attempt+1,'seconds':headers-start})
                result=read_response(response,(body or {}).get('stream',False))
            record({'stage':'response','attempt':attempt+1,'headers_seconds':headers-start,'total_seconds':time.monotonic()-start,'usage':result.get('usage'),'finish_reason':result.get('choices',[{}])[0].get('finish_reason')})
            return result
        except urllib.error.HTTPError as exc:
            record({'stage':'http','attempt':attempt+1,'http_status':exc.code})
            if exc.code not in (429,500,502,503,504) or attempt==3: raise
        except (urllib.error.URLError,TimeoutError,ConnectionError) as exc:
            reason=getattr(exc,'reason',exc)
            record({'stage':'dns' if isinstance(reason,socket.gaierror) else 'transport','attempt':attempt+1,'error':type(reason).__name__,'seconds':time.monotonic()-start})
            if attempt==3: raise
        time.sleep(2**attempt)

def chat(model, prompt, **settings):
    settings.setdefault('reasoning_effort', os.getenv('SII_REASONING_EFFORT','none'))
    body=dict(model=model,temperature=0,stream=True,stream_options={'include_usage':True},messages=[{'role':'user','content':prompt}])
    body.update(settings)
    result=request('/chat/completions',body)
    # High reasoning shares the output budget; retry a truncated response with room for the answer.
    while result.get('choices',[{}])[0].get('finish_reason')=='length' and body.get('max_tokens',65536)<65536:
        body['max_tokens']=min(65536,body['max_tokens']*2)
        result=request('/chat/completions',body)
    return result

def safe_chat(model,prompt,*,sample_id,benchmark,**settings):
    try:
        response=chat(model,prompt,**settings)
        c=response.get('choices',[{}])[0]
        if c.get('finish_reason')=='length' or not c.get('message',{}).get('content','').strip():
            record({'stage':'invalid_output','benchmark':benchmark,'id':sample_id,'finish_reason':c.get('finish_reason'),'usage':response.get('usage')})
            return None
        return response
    except (urllib.error.URLError,TimeoutError,ConnectionError,json.JSONDecodeError) as exc:
        record({'stage':'sample_failed','benchmark':benchmark,'id':sample_id,'error':type(exc).__name__})
        return None
