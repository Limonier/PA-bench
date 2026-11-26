from unittest.mock import patch
import threading,time
import timed_nex
class FakeSocket:
 def __init__(self):self.stopped=threading.Event()
 def shutdown(self,*args):self.stopped.set()
 def close(self):self.stopped.set()
class FakeConn:
 def __init__(self,*args,**kwargs):self.sock=FakeSocket()
 def connect(self):pass
 def request(self,*args,**kwargs):pass
 def getresponse(self):return self
 status=200
 def readline(self):self.sock.stopped.wait(1);return b''
 def close(self):self.sock.close()
with patch.object(timed_nex.http.client,'HTTPSConnection',FakeConn),patch.object(timed_nex,'load_key',return_value='test'):
 start=time.monotonic();r=timed_nex.phase('test','medium',.03)
 assert r['status']=='deadline' and time.monotonic()-start<.5,r
calls=[]
def fake_phase(prompt,effort,seconds):
 calls.append((prompt,effort));return {'status':'deadline' if effort=='medium' else 'complete','answer':'' if effort=='medium' else '[1]','effort':effort,'seconds':.1}
with patch.object(timed_nex,'phase',fake_phase):
 r=timed_nex.legacy_answer('OFFICIAL PROMPT');assert r['fallback'] and r['answer']=='[1]' and calls==[('OFFICIAL PROMPT','medium'),('OFFICIAL PROMPT','none')]
print('Deadline cancellation and clean-prompt fallback checks passed')

# DNS failures happen before submission: retain medium and its full deadline.
failed={'status':'gaierror','seconds':.1,'error':'temporary DNS failure'}
success={'status':'complete','seconds':1,'connected':True,'answer':'ok'}
with patch.object(timed_nex,'_phase',side_effect=[failed.copy(),failed.copy(),success.copy()]) as calls, patch.object(timed_nex.time,'sleep') as sleep:
 r=timed_nex.phase('OFFICIAL PROMPT','medium',60,max_tokens=16384)
 assert r['status']=='complete' and len(r['network_retries'])==2
 assert [c.args for c in sleep.call_args_list]==[(2,),(4,)]
 assert all(c.args==('OFFICIAL PROMPT','medium',60) and c.kwargs=={'max_tokens':16384} for c in calls.call_args_list)
with patch.object(timed_nex,'_phase',return_value={'status':'deadline','connected':True,'seconds':60}) as calls, patch.object(timed_nex.time,'sleep') as sleep:
 assert timed_nex.phase('test','medium',60)['status']=='deadline'
 assert calls.call_count==1 and not sleep.called
print('Pre-request DNS backoff and unchanged inference deadlines passed')

done={'status':'complete','seconds':1,'answer':'ok','usage':{'completion_tokens':12},'finish_reason':'stop'}
length={'status':'incomplete','seconds':1,'answer':'','usage':{'completion_tokens':16384},'finish_reason':'length'}
with patch.object(timed_nex,'phase',side_effect=[length,done]) as calls:
 assert timed_nex.answer('task')['status']=='budget_exceeded'
 assert timed_nex.answer('judge',scoring=True)['status']=='complete'
 assert [c.kwargs['max_tokens'] for c in calls.call_args_list]==[16384,None]
 assert all(c.args[1:]==('medium',None) for c in calls.call_args_list)
print('New single-task budget and unlimited judge passed without fallback')

# Probe the actual streaming transport without contacting a server.
import json, ssl
class ConnectTimeout(FakeConn):
 def connect(self):raise TimeoutError('connect timed out')
class HeaderTimeout(FakeConn):
 def getresponse(self):raise TimeoutError('headers timed out')
class ReasoningTimeout(FakeConn):
 def getresponse(self):
  self.lines=iter([b'data: '+json.dumps({'choices':[{'delta':{'reasoning_content':'thinking'}}]}).encode()+b'\n'])
  return self
 def readline(self):
  try:return next(self.lines)
  except StopIteration:raise TimeoutError('stream stalled')
class SSLInterrupted(ReasoningTimeout):
 def readline(self):
  try:return next(self.lines)
  except StopIteration:raise ssl.SSLEOFError('connection closed')
for conn,category,confirmed in [
 (ConnectTimeout,'network_before_submission',True),
 (HeaderTimeout,'upstream_no_output_timeout',False),
 (ReasoningTimeout,'reasoning_stream_stalled',False),
 (SSLInterrupted,'transport_interrupted',False)]:
 with patch.object(timed_nex.http.client,'HTTPSConnection',conn),patch.object(timed_nex,'load_key',return_value='test'):
  result=timed_nex.phase('test','medium',None)
  probe=result['diagnostic']
  assert probe['category']==category and probe['root_cause_confirmed']==confirmed,probe
  if conn in (ReasoningTimeout,SSLInterrupted):
   assert probe['evidence']['reasoning_chars']==8 and 'last_event_seconds' in probe['evidence']
   assert probe['submission_state']=='sent'
print('Passive attribution separates connect timeout, no output, reasoning stall, and TLS interruption')

progress=[]
with patch.object(timed_nex.http.client,'HTTPSConnection',ReasoningTimeout),patch.object(timed_nex,'load_key',return_value='test'):
 result=timed_nex.phase('test','none',None,on_progress=progress.append)
 assert [p['request_stage'] for p in progress[:4]]==['connect','submit','headers','stream']
 assert all(p['effort']=='none' and 'reasoning' not in p and 'answer' not in p for p in progress)
print('Live progress records connection stages without raw content')
