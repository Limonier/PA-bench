"""Adapters for the official algorithms; only the GPG workflow is reimplemented.

Use one instance per user and trial. No model requests are made on import.
"""
from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import site
import sys
from functools import lru_cache
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
UPSTREAM = HERE / 'upstream'
if (HERE / '.deps').is_dir():
    site.addsitedir(str(HERE / '.deps'))
METHODS = ('gpg', 'alignxplore_plus', 'pgmem', 'omem', 'mem0')
MODEL = 'nex-n2.5-pro'


@dataclass(frozen=True)
class History:
    text: str
    timestamp: str = ''
    query: str = ''
    chosen: str = ''
    rejected: str = ''
    role: str = 'user'
    assistant: str = ''

    def render(self):
        return (f'[{self.timestamp}] ' if self.timestamp else '') + self.text

    def transcript(self):
        text = f'{self.role}: {self.render()}'
        return text + ('\nassistant: '+self.assistant if self.assistant else '')


def nex_chat(messages, *, temperature=0, max_tokens=16384, raw=False, **kwargs):
    if str(ROOT / 'scripts') not in sys.path:
        sys.path.insert(0, str(ROOT / 'scripts'))
    from nex_vita_transport import chat, task_budget, TaskBudget, PREPARATION_EFFORT
    with task_budget(TaskBudget(limit=None,effort=PREPARATION_EFFORT)):
        response = chat(messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
    return response if raw else response['choices'][0]['message']['content']


def _json(text):
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.partition('\n')[2].rsplit('```', 1)[0]
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError('Expected a JSON object')
    return value


@lru_cache(maxsize=1)
def _local_embedding():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(HERE / 'models/all-MiniLM-L6-v2'), device='cpu', local_files_only=True)


GUIDES = {
    'lamp1': 'Identify recurring research topics, authors, fields and citation preferences.',
    'lamp2': 'Identify which labels this person assigns to films and how themes distinguish those labels.',
    'lamp3': 'Identify what drives high and low ratings, the rating scale, and recurring product preferences.',
    'lamp4': 'Identify headline length, vocabulary, structure, tone and recurring subject matter.',
    'lamp5': 'Identify research title structure, terminology, style and typical specificity.',
    'lamp7': 'Among capitalization, emoji, abbreviations and punctuation, identify the single most distinctive property of the historical tweets.',
    'mega': 'Identify stable attitudes, decision patterns, risk tolerance, values and conditional preferences supported by historical answers.',
    'fingertip': 'Identify recurring phone activities, applications, temporal patterns, contexts and concrete intentions.',
    'vita': 'Identify current preferences and constraints, their context, and changes over time.',
}


class GPG:
    """Paper-derived digestion -> profile -> raw history plus profile.

    New-domain questions are explicit adaptations, not claimed author prompts.
    """
    def __init__(self, task, chat=nex_chat):
        self.task, self.chat = task, chat
        self.history = []
        self.profile = self.digest = ''

    def update(self, records):
        self.history.extend(records)
        if not self.history:
            return
        history = '\n'.join(r.transcript() for r in self.history)
        guide = GUIDES[self.task]
        self.digest = self.chat([{'role':'user','content':f'Personal context:\n{history}\n\n{guide}'}])
        self.profile = self.chat([{'role':'user','content':f'Personal context:\n{history}\n\nGuided analysis:\n{self.digest}\n\nWrite a concise descriptive profile of this person grounded in these observations.'}])
        if not self.profile or not self.profile.strip():
            raise ValueError('Empty GPG profile')

    def retrieve(self, query=''):
        if not self.history:
            return ''
        sections = ['Personal history:\n'+'\n'.join(r.transcript() for r in self.history), 'Generated profile:\n'+self.profile]
        if self.task == 'lamp7':
            sections.insert(1, 'Guided analysis:\n'+self.digest)
        return '\n\n'.join(sections)


def _align_prompt(records, profile):
    # Execute ONLY the author's prompt function, avoiding the script's vLLM and dataset imports.
    path = UPSTREAM / 'AlignX-Family' / 'AlignXplorePlus' / 'eval' / 'streaming_gen_preference.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'build_prompt')
    from typing import List, Dict, Optional
    scope = dict(AutoTokenizer=object, List=List, Dict=Dict, Optional=Optional)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages)
    triples = [{'query': r.query or 'Observed historical activity', 'chosen': r.chosen or r.text,
                'rejected': r.rejected or 'Not provided'} for r in records if r.role == 'user']
    return scope['build_prompt'](tokenizer, triples, profile or None)


class AlignGenerator:
    """Load the actual AlignXplore+ checkpoint; no Nex substitution."""
    def __init__(self, checkpoint, quantize=False):
        if not Path(checkpoint).is_dir():
            raise FileNotFoundError('Provide a local copy of VanillaH1/AlignXplore-Plus')
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        options = {'torch_dtype': torch.bfloat16, 'device_map': 'auto', 'local_files_only': True}
        if quantize:
            from transformers import BitsAndBytesConfig
            options['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, **options).eval()

    def __call__(self, messages):
        import torch
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt').to(self.model.device)
        if ids.shape[-1] > 12288:
            raise ValueError('AlignXplore+ input exceeds 12288-token budget; chunk history before updating')
        with torch.no_grad():
            output = self.model.generate(ids, max_new_tokens=4096, do_sample=True, temperature=.6, top_k=20, top_p=.95)
        generated = output[0, ids.shape[-1]:].tolist()
        end_think = self.tokenizer.convert_tokens_to_ids('</think>')
        if end_think in generated:
            generated = generated[generated.index(end_think)+1:]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    def fits(self, messages):
        ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return len(ids) <= 12288


class AlignXplore:
    def __init__(self, generator):
        self.generator = generator
        self.profile = ''

    def update(self, records):
        if not records:
            return
        messages = _align_prompt(records, self.profile)
        if hasattr(self.generator,'fits') and not self.generator.fits(messages):
            if len(records)<2:
                raise ValueError('Individual AlignXplore+ history record exceeds its context budget')
            mid = len(records)//2
            self.update(records[:mid])
            self.update(records[mid:])
            return
        result = self.generator(messages)
        profile = result.strip().split('</think>')[-1].strip()
        if not profile:
            raise ValueError('Empty AlignXplore+ profile')
        self.profile = profile

    def retrieve(self, query=''):
        return self.profile


class PGClient:
    def __init__(self, chat):
        self.chat = chat
        self.failures = {}

    @property
    def failure(self):
        return next(iter(self.failures.values()), None)

    def generate(self, prompt, system_prompt='', guided_json=None, temperature=0, **kwargs):
        messages = [{'role':'system','content':system_prompt}, {'role':'user','content':prompt}]
        if guided_json:
            messages[-1]['content'] += '\nReturn a JSON object matching this schema:\n'+json.dumps(guided_json, ensure_ascii=False)
        options = {'response_format': {'type':'json_schema','json_schema':{'name':'pgmem_result','schema':guided_json,'strict':True}}} if guided_json else {}
        call_key = (system_prompt, prompt.split('\nPrevious attempt returned empty judgments;')[0])
        try:
            response = self.chat(messages, temperature=temperature, raw=True, **options) if self.chat is nex_chat else None
            result = response['choices'][0]['message']['content'] if response else self.chat(messages, temperature=temperature, **options)
            if guided_json:
                value = _json(result)
                import jsonschema
                jsonschema.validate(value, guided_json)
                self.failures.pop(call_key, None)
                if response:
                    value['_usage'] = response['usage']
                return value
            return result
        except Exception as exc:
            self.failures[call_key] = exc
            raise


class PGMem:
    def __init__(self, chat=nex_chat, embedding=None, nlp=None, timestamped=False):
        sys.path.insert(0, str(UPSTREAM / 'pgmem' / 'personamem' / 'pgmem'))
        import config_0
        from graphmem_module import GraphMemModule
        if nlp is None:
            import spacy
            nlp = spacy.load(config_0.SPACY_MODEL)
        self.client = PGClient(chat)
        # FingerTip events are individual chunks, but their gaps are real clock time.
        # Keep upstream chunk identity and relative-time formatting; one ID unit is 1 ms.
        config = SimpleNamespace(**{k:v for k,v in vars(config_0).items() if not k.startswith('__')}) if timestamped else config_0
        if timestamped:
            # Upstream floors minute buckets; avoid rounding exact boundaries down.
            config.TIME_PER_CONV_ID_HOURS = math.nextafter(1 / 3_600_000, math.inf)
        self.graph = GraphMemModule(self.client, model_path=MODEL, config=config, embed_model=embedding or _local_embedding(), nlp=nlp)
        self.turn = 0
        self.timestamped = timestamped
        self.last_conv_id = None

    @staticmethod
    def event_clock(timestamp):
        return int((datetime.strptime(timestamp, '%Y%m%d_%H%M%S') - datetime(1970,1,1)).total_seconds()) * 1000

    def set_prediction_time(self, timestamp):
        if self.timestamped:
            anchor = self.event_clock(timestamp)
            if self.last_conv_id is None or anchor <= self.last_conv_id:
                raise ValueError('Prediction cutoff must follow all historical events')
            self.graph._current_conv_id = anchor
            self.graph._current_turn_id = 0

    def update(self, records):
        for record in records:
            self.client.failures.clear()
            conv_id = self.turn
            if self.timestamped:
                conv_id = self.event_clock(record.timestamp)
                if self.last_conv_id is not None:
                    if conv_id < (self.last_conv_id // 1000)*1000:
                        raise ValueError('FingerTip history must be chronological')
                    conv_id = max(conv_id, self.last_conv_id+1)
                self.last_conv_id = conv_id
            self.graph.process_turn(record.render() if record.role == 'user' else '', record.assistant or (record.render() if record.role == 'assistant' else ''), conv_id=conv_id, turn_id=0, session_id=0)
            self.turn += 1
            if self.client.failure:
                raise RuntimeError('PGMem internal LLM call failed') from self.client.failure
        if records:
            self.graph.finalize_chunk(session_id=0)
            if self.client.failure:
                raise RuntimeError('PGMem finalization failed') from self.client.failure

    def retrieve(self, query=''):
        if not self.turn:
            return ''
        return self.graph.prepare_qa(query, options=[]).retrieval_result.serialized


class AsyncNex:
    def __init__(self, chat):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self._chat = chat
        self.calls = 0

    async def create(self, *, messages, temperature=0, **kwargs):
        self.calls += 1
        content = await asyncio.to_thread(self._chat, messages, temperature=temperature)
        # Some OpenAI-compatible gateways wrap an otherwise valid JSON value in
        # a Markdown fence. Normalize only that transport envelope; preserve all
        # values and leave malformed or natural-language content untouched.
        stripped = content.strip()
        if stripped.startswith('```') and stripped.endswith('```'):
            candidate = stripped.partition('\n')[2].rsplit('```', 1)[0].strip()
            try:
                json.loads(candidate)
            except ValueError:
                pass
            else:
                content = candidate
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class OMem:
    def __init__(self, storage, chat=nex_chat, embedding=None):
        sys.path.insert(0, str(UPSTREAM / 'O-Mem'))
        from memory_chain import MemoryChain, MemoryManager
        folder = Path(storage).resolve()
        if folder.exists() and any(folder.iterdir()):
            raise ValueError('O-Mem requires a new storage directory per user/trial')
        folder.mkdir(parents=True, exist_ok=True)
        self.client = AsyncNex(chat)
        self.config = SimpleNamespace(working_memory_max_size=20, episodic_memory_refresh_rate=5,
                                      number_of_retrieval_pieces=15, drop_threshold=.1, output_dir=str(folder))
        self.embedding = embedding or _local_embedding()
        args = {'model':{'llm_model':MODEL}}
        kwargs = dict(memory_index=0,llm_model=MODEL,llm_client=self.client,embedding_model=self.embedding,
                      user_name='User',agent_name='Assistant',cmd_args=self.config,args=args,memory_dir=str(folder))
        self.memory = MemoryChain(**kwargs)
        self.manager = MemoryManager(memory_system=self.memory, **kwargs)
        self.index = 0

    async def _update(self, records):
        for record in records:
            await self.manager.receive_message(record.render(), self.index, self.client, record.timestamp or '1970-01-01 00:00:00', record.role=='user')
            self.index += 1
            if record.assistant:
                await self.manager.receive_message(record.assistant, self.index, self.client, record.timestamp or '1970-01-01 00:00:00', False)
                self.index += 1
        for side in ('user','agent'):
            working = getattr(self.memory, side+'_working_memory')
            episodic = getattr(self.memory, side+'_episodic_memory')
            topics = getattr(self.memory, side+'_topic_message_dict')
            for message in list(working.working_memory_queue.queue)+episodic.episodic_memory_cache_list:
                topics[message['topics']] = [message['raw_message'], message['timestamp']]
        # Rebuild, rather than append duplicate keyword postings on each update.
        self.memory.user_detail_dict.clear()
        self.memory.agent_detail_dict.clear()
        self.memory.generate_memory_detail_map()
        await self.memory.user_persona_memory.update_preference_persona(self.memory.user_episodic_memory.topic_episodic_memory_list)
        await self.memory.user_persona_memory.update_attribute_persona(self.memory.user_episodic_memory.attribute_episodic_memory_dict)

    def update(self, records):
        if records:
            asyncio.run(self._update(records))

    def retrieve(self, query=''):
        if not self.index:
            return ''
        result, *_ = self.manager.retrieve_from_memory_soft_segmentation('User: '+query, self.config.number_of_retrieval_pieces, self.config.drop_threshold)
        # Three channels remain in upstream retrieval; exclude internal numeric/debug fields.
        return json.dumps({k:result.get(k,[]) for k in ('persona attributes','persona facts','retrieved context messages','keyword message')}, ensure_ascii=False, default=str)


class Mem0Client:
    def __init__(self, chat):
        self.chat = chat
        self.failure = None

    def generate_response(self, messages, response_format=None, **kwargs):
        try:
            text = self.chat(messages)
            if response_format:
                value = _json(text)
                # Mem0's official fact extractor uses system + user messages;
                # its update planner uses a single user message.
                if len(messages) == 2 and messages[0]['role'] == 'system':
                    facts = value.get('facts')
                    if not isinstance(facts, list) or any(not isinstance(f, str) or not f.strip() for f in facts):
                        raise ValueError('Mem0 expected facts: list of non-empty strings')
                else:
                    actions = value.get('memory')
                    if not isinstance(actions, list) or any(
                        not isinstance(a, dict) or a.get('event') not in ('ADD','UPDATE','DELETE','NONE')
                        or not isinstance(a.get('id'), str)
                        or not isinstance(a.get('text'), str) or not a['text'].strip()
                        for a in actions
                    ):
                        raise ValueError('Mem0 expected memory: list of valid actions')
                return json.dumps(value, ensure_ascii=False)
            return text
        except Exception as exc:
            self.failure = exc
            raise


class Mem0:
    def __init__(self, storage, chat=nex_chat, embedding_url='http://127.0.0.1:8766/v1'):
        path = Path(storage).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.environ['MEM0_TELEMETRY'] = 'False'
        os.environ.setdefault('MEM0_DIR', str(HERE / 'runtime' / 'mem0'))
        sys.path.insert(0, str(UPSTREAM / 'mem0'))
        from mem0 import Memory
        from qdrant_client import QdrantClient
        self.client = Mem0Client(chat)
        self.memory = Memory.from_config({'llm':{'provider':'openai','config':{'model':MODEL,'api_key':'unused-local-adapter'}},
            'embedder':{'provider':'openai','config':{'model':'paraphrase-multilingual-MiniLM-L12-v2','embedding_dims':384,'api_key':'local-only','openai_base_url':embedding_url}},
            'vector_store':{'provider':'qdrant','config':{'collection_name':'memory','path':str(path/'vectors'),'client':QdrantClient(path=str(path/'vectors')),'embedding_model_dims':384,'on_disk':True}},
            'history_db_path':str(path/'history.db')})
        self.memory.llm = self.client
        self.user_id = 'user'

    def update(self, records):
        for r in records:
            self.client.failure = None
            messages = [{'role':r.role,'content':r.render()}]
            if r.assistant:
                messages.append({'role':'assistant','content':r.assistant})
            self.memory.add(messages,user_id=self.user_id,infer=True)
            if self.client.failure:
                raise RuntimeError('Mem0 internal LLM call failed') from self.client.failure

    def retrieve(self, query='', limit=10):
        result = self.memory.search(query,user_id=self.user_id,limit=limit)
        return '\n'.join(r['memory'] for r in result['results'])

    def close(self):
        self.memory.db.connection.close()
        self.memory.vector_store.client.close()


def create(method, **kwargs):
    classes = dict(gpg=GPG, alignxplore_plus=AlignXplore, pgmem=PGMem, omem=OMem, mem0=Mem0)
    if method not in classes:
        raise ValueError(f'Unknown method: {method}; choose from {METHODS}')
    return classes[method](**kwargs)
