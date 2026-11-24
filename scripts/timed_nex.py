"""Nex streaming transport, passive failure attribution, and legacy diagnostics."""
import http.client,json,os,socket,threading,time,sys
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from sii import BASE,load_key

# User-locked experiment policy: callers cannot override the model, effort sequence or deadlines.
MODEL='nex-n2.5-pro'
ALTERNATE_NONE_ENDPOINTS={
 'Nex-N2.6-W8A8':'https://8bjahbmodog5ce5kj99k8koqm9hcg9kc.openapi-sj.sii.edu.cn/v1',
 'Qwen3.8-27B':'https://cqhbod8bjjjbcoakk8pmeebgkaq9akcq.openapi-sj.sii.edu.cn/v1',
}

def request_model(effort):
 model=os.getenv('PA_BENCH_ALT_MODEL') if effort=='none' else None
 if model and model not in ALTERNATE_NONE_ENDPOINTS:raise ValueError('Unsupported alternate model')
 return model or MODEL
PRIMARY_EFFORT='medium'
PREPARATION_EFFORT='none'
FALLBACK_EFFORT='none'
THINK_SECONDS=60
ANSWER_SECONDS=30
TASK_TOKEN_BUDGET=16384

@lru_cache(maxsize=1)
def diagnostic_tokenizer():
 sys.path.append(str(Path(__file__).resolve().parents[1]/'.retrieval-venv/Lib/site-packages'))
 from tokenizers import Tokenizer
 return Tokenizer.from_file(os.environ['VITA_DIAGNOSTIC_TOKENIZER'])

def failure_diagnostic(row, status=None):
 """Passive attribution: never replay a generation or infer an unseen cause."""
 status=status or row.get('status')
 stage=row.get('request_stage','unknown')
 if status=='complete':return None
 if status in ('budget_exceeded','thinking_budget') or row.get('finish_reason')=='length':
  category='model_token_limit'
 elif status=='usage_missing':
  category='usage_missing'
 elif status=='deadline':
  category='client_deadline'
 elif stage=='connect' and status in ('gaierror','TimeoutError','ConnectionRefusedError','ConnectionResetError','SSLError','SSLEOFError','SSLCertVerificationError','OSError'):
  category='network_before_submission'
 elif status in ('SSLError','SSLEOFError','SSLCertVerificationError','ConnectionResetError','BrokenPipeError','RemoteDisconnected','ConnectionAbortedError','IncompleteRead'):
  category='transport_interrupted'
 elif status=='TimeoutError':
  category=('submission_timeout_unknown' if stage=='submit'
            else 'reasoning_stream_stalled' if row.get('reasoning_chars') and not row.get('answer_chars') and 'first_tool_seconds' not in row
            else 'generation_stream_stalled' if row.get('answer_chars') or 'first_tool_seconds' in row
            else 'upstream_no_output_timeout')
 elif row.get('http_status',200)!=200 or row.get('upstream_error'):
  category='upstream_error'
 elif row.get('incomplete_reason')=='missing_done':
  category='stream_ended_without_done'
 else:category='response_or_client_error'
 return {'category':category,'status':status,'phase':stage,
         'submission_state':'sent' if row.get('request_sent') else 'not_sent' if stage=='connect' else 'unknown',
         'root_cause_confirmed':category in ('network_before_submission','model_token_limit','client_deadline','usage_missing'),
         'seconds_since_last_event':round(row['seconds']-row['last_event_seconds'],3) if 'seconds' in row and 'last_event_seconds' in row else None,
         'evidence':{k:row[k] for k in ('connect_seconds','request_seconds','headers_seconds','first_event_seconds',
                     'first_reasoning_seconds','first_answer_seconds','first_tool_seconds','last_event_seconds','last_output_seconds','seconds',
                     'reasoning_chars','answer_chars','finish_reason','http_status','done_received','error','network_retries','network_wait_seconds') if k in row}}


def budget_trace(row, status=None):
 """Retain the upstream text verbatim only for a token-limit termination."""
 status=status or row.get('status')
 if status not in ('budget_exceeded','thinking_budget') and row.get('finish_reason')!='length':
  return None
 return {'source':'terminating_request','reasoning_text':row.get('reasoning'),
         'partial_answer':row.get('answer'),'tool_calls':row.get('tool_calls'),
         'usage':row.get('usage'),'finish_reason':row.get('finish_reason'),
         'done_received':row.get('done_received'),
         'reasoning_text_available':bool(row.get('reasoning'))}


def _phase(prompt,effort,seconds,*,temperature=0,max_tokens=None,messages=None,tools=None,tool_choice=None,thinking_budget=None,on_progress=None,response_format=None):
 model=request_model(effort);alternate=model!=MODEL
 url=urlsplit(ALTERNATE_NONE_ENDPOINTS[model] if alternate else BASE);conn=http.client.HTTPSConnection(url.hostname,url.port or 443,timeout=seconds if seconds is not None else 120)
 start=time.monotonic();row={'effort':effort,'deadline_seconds':seconds,'answer':'','reasoning':'','done_received':False,'request_stage':'connect','request_sent':False};expired=threading.Event();sock=None
 last_report=None
 def report(force=False):
  nonlocal last_report
  now=time.monotonic()
  if not on_progress or (not force and last_report is not None and now-last_report<15):return
  last_report=now
  on_progress({'effort':effort,'request_stage':row['request_stage'],'elapsed_seconds':now-start,
               'reasoning_chars':len(row['reasoning']),'answer_chars':len(row['answer']),
               'request_sent':row['request_sent'],'last_event_seconds':row.get('last_event_seconds'),
               'first_reasoning_seconds':row.get('first_reasoning_seconds'),'first_answer_seconds':row.get('first_answer_seconds')})
 def abort():
  expired.set()
  if sock:
   try:sock.shutdown(socket.SHUT_RDWR)
   except OSError:pass
   sock.close()
 timer=threading.Timer(seconds,abort) if seconds is not None else None
 if timer:timer.daemon=True;timer.start()
 try:
  report(True)
  conn.connect();sock=conn.sock;row['connected']=True;row['connect_seconds']=time.monotonic()-start
  if expired.is_set():raise TimeoutError()
  body={'model':MODEL,'reasoning_effort':effort,'temperature':temperature,'stream':True,'stream_options':{'include_usage':True},'messages':messages if messages is not None else [{'role':'user','content':prompt}]}
  if alternate:
   body['model']=model;body.pop('reasoning_effort');body['chat_template_kwargs']={'enable_thinking':False}
  if tools:body.update(tools=tools,tool_choice=tool_choice or 'auto')
  if max_tokens is not None:body['max_tokens']=max_tokens
  if response_format is not None:body['response_format']=response_format
  row['request_stage']='submit'
  report(True)
  conn.request('POST',url.path+'/chat/completions',json.dumps(body).encode(),{'Authorization':'Bearer '+(os.environ['PA_BENCH_ALT_KEY'] if alternate else load_key()),'Content-Type':'application/json'})
  row['request_sent']=True;row['request_seconds']=time.monotonic()-start;row['request_stage']='headers'
  report(True)
  response=conn.getresponse();row['http_status']=response.status;row['headers_seconds']=time.monotonic()-start
  if response.status!=200:raise RuntimeError('HTTP '+str(response.status))
  row['request_stage']='stream'
  report(True)
  done=False;tool_parts={}
  while True:
   if expired.is_set():raise TimeoutError()
   line=response.readline()
   if not line:break
   if not line.startswith(b'data:'):continue
   data=line[5:].strip()
   if data==b'[DONE]':done=True;row['done_received']=True;break
   if not data:continue
   row.setdefault('first_event_seconds',time.monotonic()-start);row['last_event_seconds']=time.monotonic()-start
   event=json.loads(data)
   if event.get('error'):row['upstream_error']=True;raise RuntimeError('Upstream error')
   if event.get('usage'):row['usage']=event['usage']
   for c in event.get('choices',[]):
    delta=c.get('delta',{})
    for field,key in [('content','first_answer_seconds'),('reasoning_content','first_reasoning_seconds'),('tool_calls','first_tool_seconds')]:
     if delta.get(field):
      row.setdefault(key,time.monotonic()-start);row['last_output_seconds']=time.monotonic()-start
    row['answer']+=delta.get('content') or '';row['reasoning']+=delta.get('reasoning_content') or ''
    for tc in delta.get('tool_calls') or []:
     target=tool_parts.setdefault(tc['index'],{'id':'','type':'function','function':{'name':'','arguments':''}})
     if tc.get('id'):target['id']=tc['id']
     for key in ['name','arguments']:target['function'][key]+=tc.get('function',{}).get(key) or ''
    if c.get('finish_reason'):row['finish_reason']=c['finish_reason']
   report()
   if thinking_budget is not None and row['reasoning'] and not (row['answer'].strip() or tool_parts or row.get('finish_reason')):
    row['observed_reasoning_tokens']=len(diagnostic_tokenizer().encode(row['reasoning'],add_special_tokens=False).ids)
    if row['observed_reasoning_tokens']>=thinking_budget:
     row['status']='thinking_budget';abort();break
  if row.get('status')=='thinking_budget':return _finish_budget(row,start)
  if tool_parts:
   row['tool_calls']=[tool_parts[k] for k in sorted(tool_parts)]
   for tc in row['tool_calls']:
    if not tc['id'] or not tc['function']['name'] or not isinstance(json.loads(tc['function']['arguments']),dict):raise ValueError('Invalid tool call')
  valid=(row.get('finish_reason')=='stop' and row['answer'].strip()) or (row.get('finish_reason')=='tool_calls' and row.get('tool_calls'))
  row['status']='deadline' if expired.is_set() else ('complete' if done and valid else 'incomplete')
 except Exception as exc:
  row['status']='deadline' if expired.is_set() else type(exc).__name__
  row['error']=str(exc)
 finally:
  if timer:timer.cancel()
  conn.close()
 if row['status']=='incomplete':
  row['incomplete_reason']='token_limit' if row.get('finish_reason')=='length' else ('missing_done' if not row['done_received'] else ('empty_final' if row.get('finish_reason')=='stop' else 'unexpected_finish_reason'))
 row['answer_chars']=len(row['answer']);row['reasoning_chars']=len(row['reasoning'])
 row['seconds']=time.monotonic()-start
 return row

def _finish_budget(row,start):
 row['seconds']=time.monotonic()-start
 row['answer_chars']=len(row['answer']);row['reasoning_chars']=len(row['reasoning'])
 return row

def phase(prompt,effort,seconds,**options):
 # Only a failed pre-request connection is safe to retry without replaying generation.
 attempts=[]
 for delay in (2,4,8,16,32,None):
  row=_phase(prompt,effort,seconds,**options)
  if row['status']!='gaierror' or row.get('connected') or delay is None:break
  attempts.append({'status':row['status'],'seconds':row['seconds'],'wait_seconds':delay,'error':row.get('error')})
  print(f'Nex DNS lookup failed; retrying {effort} in {delay}s',file=sys.stderr,flush=True)
  time.sleep(delay)
 if attempts:
  row['network_retries']=attempts
  row['network_wait_seconds']=sum(a['seconds']+a['wait_seconds'] for a in attempts)
 if row['status']!='complete':row['diagnostic']=failure_diagnostic(row)
 return row

def legacy_answer(prompt,*,temperature=0,max_tokens=None):
 options={'temperature':temperature} if temperature!=0 else {}
 if max_tokens is not None:options['max_tokens']=max_tokens
 phases=[phase(prompt,PRIMARY_EFFORT,THINK_SECONDS,**options)]
 if phases[0]['status']!='complete':phases.append(phase(prompt,FALLBACK_EFFORT,ANSWER_SECONDS,**options))
 final=phases[-1]
 return {'status':final['status'],'answer':final['answer'] if final['status']=='complete' else '', 'phases':phases,'fallback':len(phases)>1,'final_effort':final['effort'],'seconds':sum(p['seconds'] for p in phases)}

def answer(prompt,*,temperature=0,max_tokens=None,scoring=False):
 # One request is the entire task for LaMP, Mega-Study and FingerTip.
 limit=None if scoring else TASK_TOKEN_BUDGET
 p=phase(prompt,PRIMARY_EFFORT,None,temperature=temperature,max_tokens=limit)
 used=(p.get('usage') or {}).get('completion_tokens')
 status=p['status']
 if not scoring and ((p.get('finish_reason')=='length' and (used is None or used>=limit)) or (used is not None and used>limit)):
  status='budget_exceeded'
 elif status=='complete' and not scoring and used is None:
  status='usage_missing'
 if status!='complete':p['diagnostic']=failure_diagnostic(p,status)
 return {'status':status,'answer':p['answer'] if status=='complete' else '', 'phases':[p],'fallback':False,'final_effort':PRIMARY_EFFORT,'seconds':p['seconds'],'task_token_budget':limit,'generated_tokens':used}
