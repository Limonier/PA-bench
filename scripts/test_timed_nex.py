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
 r=timed_nex.answer('OFFICIAL PROMPT');assert r['fallback'] and r['answer']=='[1]' and calls==[('OFFICIAL PROMPT','medium'),('OFFICIAL PROMPT','none')]
print('Deadline cancellation and clean-prompt fallback checks passed')
