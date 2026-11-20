"""Medium with wall-clock deadline; fall back to none with the same clean prompt."""
import http.client,json,os,socket,threading,time
from urllib.parse import urlsplit
from sii import BASE,load_key

def phase(prompt,effort,seconds):
 url=urlsplit(BASE);conn=http.client.HTTPSConnection(url.hostname,url.port or 443,timeout=seconds)
 start=time.monotonic();row={'effort':effort,'deadline_seconds':seconds,'answer':'','reasoning':''};expired=threading.Event();sock=None
 def abort():
  expired.set()
  if sock:
   try:sock.shutdown(socket.SHUT_RDWR)
   except OSError:pass
   sock.close()
 timer=threading.Timer(seconds,abort);timer.daemon=True;timer.start()
 try:
  conn.connect();sock=conn.sock
  if expired.is_set():raise TimeoutError()
  body={'model':'nex-n2.5-pro','reasoning_effort':effort,'temperature':0,'stream':True,'stream_options':{'include_usage':True},'messages':[{'role':'user','content':prompt}]}
  conn.request('POST',url.path+'/chat/completions',json.dumps(body).encode(),{'Authorization':'Bearer '+load_key(),'Content-Type':'application/json'})
  response=conn.getresponse();row['http_status']=response.status
  if response.status!=200:raise RuntimeError('HTTP '+str(response.status))
  done=False
  while True:
   if expired.is_set():raise TimeoutError()
   line=response.readline()
   if not line:break
   if not line.startswith(b'data:'):continue
   data=line[5:].strip()
   if data==b'[DONE]':done=True;break
   if not data:continue
   event=json.loads(data)
   if event.get('error'):raise RuntimeError('Upstream error')
   if event.get('usage'):row['usage']=event['usage']
   for c in event.get('choices',[]):
    delta=c.get('delta',{});row['answer']+=delta.get('content') or '';row['reasoning']+=delta.get('reasoning_content') or ''
    if c.get('finish_reason'):row['finish_reason']=c['finish_reason']
  row['status']='deadline' if expired.is_set() else ('complete' if done and row.get('finish_reason')=='stop' and row['answer'].strip() else 'incomplete')
 except Exception as exc:row['status']='deadline' if expired.is_set() else type(exc).__name__
 finally:timer.cancel();conn.close()
 row['seconds']=time.monotonic()-start
 return row

def answer(prompt,think_seconds=60,answer_seconds=30):
 phases=[phase(prompt,'medium',think_seconds)]
 if phases[0]['status']!='complete':phases.append(phase(prompt,'none',answer_seconds))
 final=phases[-1]
 return {'status':final['status'],'answer':final['answer'] if final['status']=='complete' else '', 'phases':phases,'fallback':len(phases)>1,'final_effort':final['effort'],'seconds':sum(p['seconds'] for p in phases)}
