"""Offline regression checks for observed truncation and survey-format failures."""
from unittest.mock import patch
import sii
from mega import parse_prediction
from io import BytesIO

calls=[]
def truncated_then_complete(path,body):
    calls.append(body['max_tokens'])
    return {'choices':[{'finish_reason':'length' if len(calls)==1 else 'stop','message':{'content':'OK'}}]}

with patch.object(sii,'request',side_effect=truncated_then_complete):
    assert sii.chat('test','prompt',max_tokens=1024)['choices'][0]['message']['content']=='OK'
assert calls==[1024,2048]
stream=BytesIO(b'data: {"choices":[{"delta":{"content":"OK"},"finish_reason":"stop"}]}\n\ndata: [DONE]\n')
assert sii.read_response(stream,True)['choices'][0]['message']['content']=='OK'
try: sii.read_response(BytesIO(b'data: {"choices":[]}\n'),True)
except ConnectionError: pass
else: raise AssertionError('Incomplete stream accepted')
assert parse_prediction('```json\n{"Q1":{"Answers":{}}}\n```',{'Q1'})=={'Q1':{'Answers':{}}}
for invalid in ['{"Q2":{"Answers":{}}}','{"Q1":{}}','not json']:
    try: parse_prediction(invalid,{'Q1'})
    except ValueError: pass
    else: raise AssertionError('Malformed response accepted')
print('Adaptation regression checks passed')
