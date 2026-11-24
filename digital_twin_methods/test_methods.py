"""Offline contract checks: no external requests, no benchmark answers.

Run using .fingertip-venv and the isolated .deps directory for all five paths.
"""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

from digital_twin_methods import History, create
from digital_twin_methods.bench_inputs import lamp_history, mega_history, fingertip_history, vita_history


def test_profiles():
    calls = []
    def chat(messages, **kwargs):
        calls.append(messages)
        return 'style evidence' if len(calls)==1 else 'This user writes in uppercase.'
    m = create('gpg', task='lamp7', chat=chat)
    m.update([History('I LOVE walking!')])
    assert len(calls)==2 and 'I LOVE walking!' in m.retrieve('unseen task') and 'style evidence' in m.retrieve()
    assert 'unseen task' not in str(calls)

    captured = []
    def trained(messages):
        captured.append(messages)
        return 'reasoning</think> A preference summary.'
    m = create('alignxplore_plus', generator=trained)
    m.update([History('Observed tea purchase',query='Drink',chosen='tea')])
    m.update([History('Observed coffee purchase',query='Drink',chosen='coffee')])
    assert 'A preference summary.' in captured[1][1]['content']
    assert m.retrieve('question') == 'A preference summary.'
    assert 'Not provided' in captured[0][1]['content']


def test_history():
    assert lamp_history(3, {'input':'HELD_OUT', 'profile':[{'text':'old review','score':'4','date':'2025-01-01'}]})[0].chosen == '4'
    assert mega_history([{'Questions':[{'QuestionText':'A?', 'Answers':{'Text':'B'}}]}])[0].chosen == '{"Text": "B"}'
    rows = [{'user_id':'x','time':'20250101','scenario':'home','intentDescription':'old'}, {'user_id':'x','time':'20250102','scenario':'home','intentDescription':'FUTURE'}]
    assert [r.chosen for r in fingertip_history(rows, 'x', '20250102')] == ['old']
    r = vita_history([{'date':'2025-01-01','behavior':[],'dialogue':[{'role':'user','content':'Tea please'},{'role':'assistant','content':'OK'}]}])
    assert len(r)==1 and r[0].assistant=='OK'


class Embedding:
    """Deterministic test fixture, not a real embedding model."""
    def encode(self, texts, **kwargs):
        import numpy as np
        scalar = isinstance(texts,str)
        texts = [texts] if scalar else texts
        out = np.zeros((len(texts),384), dtype=np.float32)
        for i,text in enumerate(texts):
            out[i,0] = 1
            out[i,1] = int('tea' in text.lower())
        if kwargs.get('normalize_embeddings') and len(out):
            out /= np.linalg.norm(out,axis=1,keepdims=True)
        return out[0] if scalar else out


def test_pgmem():
    from digital_twin_methods.methods import PGClient
    schema = {'type':'object','properties':{'judgments':{'type':'array'}},'required':['judgments']}
    responses = iter(['{}','{"judgments":[]}','{"judgments":[]}'])
    def structured_chat(messages, **kwargs):
        assert kwargs['response_format']['json_schema']['schema'] == schema
        return next(responses)
    adapter = PGClient(structured_chat)
    try:
        adapter.generate('A', guided_json=schema)
    except Exception:
        assert adapter.failure is not None
    else:
        raise AssertionError('Invalid schema accepted')
    adapter.generate('B', guided_json=schema)
    assert adapter.failure is not None
    adapter.generate('A\nPrevious attempt returned empty judgments; retry', guided_json=schema)
    assert adapter.failure is None
    def chat(messages, **kwargs):
        schema = json.loads(messages[-1]['content'].rsplit('Return a JSON object matching this schema:\n',1)[1])
        properties = schema['properties']
        # Empty extractions are valid only when the author's schema allows them.
        values = {}
        for key,s in properties.items():
            if s.get('type')=='array': values[key]=[]
            elif s.get('type')=='object':
                sample = {'id':'1','content':'Tea at breakfast','keywords':['tea'],'domain_label':['food'],'scope':'narrow','recall_priority':'high'}
                values[key] = {k:sample[k] for k in s['properties']}
        return json.dumps(values)
    m = create('pgmem',chat=chat,embedding=Embedding(),nlp=lambda text:SimpleNamespace(noun_chunks=[]))
    m.update([History('I like tea for breakfast.')])
    assert m.graph.get_memory_stats()['num_memories']>=1
    assert isinstance(m.retrieve('What to drink?'),str)
    clock = create('pgmem',chat=chat,embedding=Embedding(),nlp=lambda text:SimpleNamespace(noun_chunks=[]),timestamped=True)
    clock.update([History('Tea at breakfast.',timestamp='20250420_095000'), History('Tea again.',timestamp='20250420_095200')])
    clock.set_prediction_time('20250420_095500')
    from graph_store import format_elapsed_str
    assert format_elapsed_str(clock.event_clock('20250420_095000'),0,clock.graph._current_conv_id,0,clock.graph._cfg.TIME_PER_CONV_ID_HOURS,10)=='5min ago'
    assert format_elapsed_str(clock.event_clock('20250420_095200'),0,clock.graph._current_conv_id,0,clock.graph._cfg.TIME_PER_CONV_ID_HOURS,10)=='3min ago'
    assert m.graph._cfg.TIME_PER_CONV_ID_HOURS==24
    assert isinstance(clock.retrieve('What to drink?'),str)


def test_omem(tmp):
    def chat(messages, **kwargs):
        if len(messages)==2:
            return json.dumps({'Action':'ADD','Target':'tea'})
        return json.dumps({'text':'Tea', 'tags':{'topic':['tea'],'attitude':['Positive'],'reason':['Likes tea'],'facts':['Likes tea'],'attributes':[]},'summary':'Likes tea','rationale':'Historical evidence'})
    m=create('omem',storage=tmp/'omem',chat=chat,embedding=Embedding())
    m.update([History('I like tea.',timestamp=f'2025-01-{i+1:02d} 10:00:00') for i in range(20)])
    assert m.memory.user_episodic_memory.episodic_memory_cache_list
    assert 'tea' in m.retrieve('What drink?').lower()


def test_mem0(tmp):
    operations = iter([('ADD','Likes tea'),('UPDATE','Likes coffee'),('NONE','Likes coffee'),('DELETE','Likes coffee')])
    current = [None]
    def chat(messages, **kwargs):
        if len(messages)==2:
            current[0]=next(operations)
            return json.dumps({'facts':[current[0][1]]})
        event,text=current[0]
        return json.dumps({'memory':[{'id':'0','text':text,'event':event}]})
    m=create('mem0',storage=tmp/'mem0',chat=chat)
    try:
        m.memory.embedding_model.embed=lambda text,*args,**kwargs: Embedding().encode(text).tolist()
        m.update([History('I like tea.')])
        assert 'Likes tea' in m.retrieve('drink')
        m.update([History('Now I prefer coffee.')])
        assert m.retrieve('drink') == 'Likes coffee'
        m.update([History('I still prefer coffee.')])
        assert m.retrieve('drink') == 'Likes coffee'
        m.update([History('Forget my drink preference.')])
        assert not m.retrieve('drink')
    finally:
        m.close()


def test_terminal_failures(tmp):
    from digital_twin_methods.methods import Mem0Client
    sys_messages = [{'role':'system','content':'extract'}, {'role':'user','content':'history'}]
    update_messages = [{'role':'user','content':'update'}]
    for messages, invalid in ((sys_messages, '{"facts":"tea"}'),
                              (update_messages, '{"memory":[{"event":"INVALID"}]}')):
        client = Mem0Client(lambda messages: invalid)
        try:
            client.generate_response(messages, response_format={'type':'json_object'})
        except ValueError as exc:
            assert client.failure is exc
        else:
            raise AssertionError('Malformed structured result was accepted')

    class TaskStop(RuntimeError):
        terminal_task_stop = True
    for method in ('mem0','pgmem','omem'):
        stop = TaskStop('budget_exceeded')
        calls = []
        def chat(messages, **kwargs):
            calls.append(messages)
            if method == 'mem0' and len(calls) == 1:
                return '{"facts":["Likes tea"]}'
            raise stop
        kwargs = {'chat':chat}
        if method != 'pgmem': kwargs['storage'] = tmp / ('stop-'+method)
        if method != 'mem0': kwargs['embedding'] = Embedding()
        if method == 'pgmem': kwargs['nlp'] = lambda _: SimpleNamespace(noun_chunks=[])
        m = create(method, **kwargs)
        if method == 'mem0':
            m.memory.embedding_model.embed = lambda text,*a,**kw: Embedding().encode(text).tolist()
        try:
            m.update([History('I like tea.')])
        except TaskStop as exc:
            assert exc is stop and len(calls) == (2 if method == 'mem0' else 1)
        else:
            raise AssertionError(method+' swallowed task termination')
        finally:
            if method == 'mem0': m.close()


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--core-only',action='store_true');a=p.parse_args()
    test_profiles();test_history();print('GPG, AlignXplore+ prompt/streaming contract and four benchmark input adapters passed')
    if not a.core_only:
        runtime = Path(__file__).resolve().parent/'runtime'
        runtime.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=runtime) as d:
            test_pgmem();print('PGMem official graph/retrieval passed')
            test_omem(Path(d));print('O-Mem official memory/retrieval passed')
            test_mem0(Path(d));print('Mem0 official vector-store update/retrieval passed')
            test_terminal_failures(Path(d));print('Strict Mem0 JSON and immediate native task termination passed')
