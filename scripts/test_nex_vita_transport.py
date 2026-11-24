"""Split tool deltas and exact message/tool preservation on fallback, offline."""
import json
from unittest.mock import patch
import timed_nex
from nex_vita_transport import chat, NexDiagnosticStop, task_budget

requests = []
class Connection:
    def __init__(self, *a, **k): self.sock = None
    def connect(self): pass
    def request(self, method, path, body, headers): requests.append(json.loads(body))
    def getresponse(self):
        self.status = 200
        events = [{'choices': [{'delta': {'tool_calls': [{'index': 0, 'id': 'call1', 'function': {'name': 'search', 'arguments': '{"query":'}}]}}]},
                  {'choices': [{'delta': {'tool_calls': [{'index': 0, 'function': {'arguments': '"清淡"}'}}]}, 'finish_reason': 'tool_calls'}]},
                  {'usage': {'prompt_tokens': 10, 'completion_tokens': 5}}]
        self.lines = iter([b'data: ' + json.dumps(e).encode() + b'\n' for e in events] + [b'data: [DONE]\n'])
        return self
    def readline(self): return next(self.lines, b'')
    def close(self): pass
messages = [{'role': 'system', 'content': 'OFFICIAL SYSTEM'}, {'role': 'user', 'content': 'OFFICIAL USER'}]
tools = [{'type': 'function', 'function': {'name': 'search', 'parameters': {'type': 'object'}}}]
with patch.object(timed_nex.http.client, 'HTTPSConnection', Connection), patch.object(timed_nex, 'load_key', return_value='test'):
    with task_budget():
        response = chat(messages, tools, max_tokens=16384)
        chat(messages, tools, max_tokens=16384)
assert requests[0]['messages'] == requests[1]['messages'] == messages
assert requests[0]['tools'] == requests[1]['tools'] == tools
assert [r['reasoning_effort'] for r in requests] == ['medium', 'medium']
assert [r['max_tokens'] for r in requests] == [16384, 16379]
assert json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments']) == {'query': '清淡'}
assert response['usage_complete'] is True
print('Multi-role messages, streamed tool arguments and cumulative task budget passed')

class BudgetConnection(Connection):
    def getresponse(self):
        self.status = 200
        event = {'choices': [{'delta': {'reasoning_content': 'test reasoning'}}]}
        self.lines = iter([b'data: ' + json.dumps(event).encode() + b'\n'])
        return self

from types import SimpleNamespace
with patch.dict('os.environ', {'VITA_DIAGNOSTIC_2048': '1'}), patch.object(timed_nex.http.client, 'HTTPSConnection', BudgetConnection), patch.object(timed_nex, 'load_key', return_value='test'), patch.object(timed_nex, 'diagnostic_tokenizer', return_value=SimpleNamespace(encode=lambda *a, **kw: SimpleNamespace(ids=range(2048)))):
    before = len(requests)
    try:
        chat(messages, tools)
        raise AssertionError('budget was not enforced')
    except NexDiagnosticStop as exc:
        assert exc.reason == 'thinking_budget' and exc.tokens == 2048
    assert len(requests) == before + 1 and requests[-1]['reasoning_effort'] == 'medium'
print('Diagnostic budget stops without fallback')

completed = {'status': 'complete', 'effort': 'medium', 'seconds': 1, 'answer': 'ok', 'finish_reason': 'stop', 'usage': {'reasoning_tokens': 3000}}
with patch.dict('os.environ', {'VITA_DIAGNOSTIC_2048': '1'}), patch('nex_vita_transport.phase', return_value=completed) as phase_call:
    chat(messages)
    assert phase_call.call_args.args == (None, 'medium', None)
    assert 'thinking_budget' not in phase_call.call_args.kwargs
    chat(messages, tools)
    assert phase_call.call_args.kwargs['thinking_budget'] == 2048
print('Diagnostic scoring and user simulation exempt from budget; no generation deadline')

truncated = {'status':'incomplete','effort':'medium','seconds':1,'answer':'','finish_reason':'length','usage':{'completion_tokens':8384}}
ok = {'status':'complete','effort':'medium','seconds':1,'answer':'ok','finish_reason':'stop','usage':{'completion_tokens':8000}}
with patch('nex_vita_transport.phase', side_effect=[ok.copy(), truncated.copy(), ok.copy()]) as called:
    with task_budget():
        chat(messages, tools)
        try:
            chat(messages, tools)
            raise AssertionError('expected budget abandonment')
        except NexDiagnosticStop as exc:
            assert exc.reason == 'budget_exceeded' and exc.tokens == 16384
    chat(messages)
    assert [c.kwargs['max_tokens'] for c in called.call_args_list] == [16384,8384,None]
print('Cumulative exhaustion abandons task; grading is uncapped')

from nex_vita_transport import TaskBudget, token_cost
without_usage={'status':'complete','seconds':1,'answer':'ok','finish_reason':'stop'}
with patch('nex_vita_transport.phase',side_effect=[ok.copy(), ok.copy(), ok.copy(), without_usage.copy()]) as called:
    with task_budget(TaskBudget(limit=None)) as preparation:
        for _ in range(4):chat(messages)
    cost=token_cost(preparation.usage_rows)
    assert preparation.remaining is None and cost['completion_tokens']==24000 and cost['missing_usage_calls']==1
    assert cost['missing_fields']['reasoning_tokens']==4
    assert [c.kwargs['max_tokens'] for c in called.call_args_list]==[None]*4
with patch('nex_vita_transport.phase',return_value=without_usage.copy()):
    with task_budget():
        try:chat(messages)
        except NexDiagnosticStop as exc:assert exc.reason=='usage_missing'
        else:raise AssertionError('Unknown answer usage accepted under hard cap')
print('Uncapped preparation ledger records exact known usage and missing fields')

# Exactly consuming the budget on a complete turn may prevent the next turn
# from being submitted. Preserve the last real response, not fictional text.
exact={'status':'complete','seconds':1,'answer':'done','reasoning':'earlier actual reasoning',
       'finish_reason':'stop','usage':{'completion_tokens':16384},'done_received':True}
with patch('nex_vita_transport.phase',return_value=exact.copy()) as called:
    with task_budget():
        chat(messages)
        try:chat(messages)
        except NexDiagnosticStop as exc:
            assert exc.budget_trace['source']=='previous_completed_request'
            assert exc.budget_trace['reasoning_text']==exact['reasoning']
        else:raise AssertionError('Expected exhausted budget')
    assert called.call_count==1
print('Exact-budget completion retains the last actual response when the next turn is blocked')
