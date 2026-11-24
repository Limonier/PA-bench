"""Controlled replay of real Vita Rewrite input; raw experiment outputs only."""
import concurrent.futures, hashlib, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'upstream/VitaBench-2.0/src'))
from vita.memory.rewrite_memory import RewriteMemory
import vita.memory.rewrite_memory as rewrite
from timed_nex import legacy_answer as answer
OUT=ROOT/'results/vita-incomplete-probe-20261121'
OUT.mkdir(parents=True,exist_ok=True)
user=next(t for t in json.loads((ROOT/'data/vita/tasks.json').read_text(encoding='utf-8')) if t['id']=='U010122')
captured={}
def capture(**kwargs):
 captured['messages']=[{'role':m.role,'content':m.content} for m in kwargs['messages']]
 return SimpleNamespace(content='capture only')
rewrite.generate=capture
memory=RewriteMemory(language='chinese',max_tokens=4096)
memory.update(user['subtasks'][0]['interactions'],llm='nex-n2.5-pro')
messages=captured['messages']
(OUT/'input.json').write_text(json.dumps({'user_id':user['id'],'subtask_id':user['subtasks'][0]['subtask_id'],'interactions':58,'messages':messages},ensure_ascii=False,indent=2),encoding='utf-8')
from timed_nex import phase,PRIMARY_EFFORT,FALLBACK_EFFORT,THINK_SECONDS,ANSWER_SECONDS

def run(case):
 budget,repeat=case
 phases=[phase(None,PRIMARY_EFFORT,THINK_SECONDS,messages=messages,max_tokens=budget)]
 if phases[0]['status']!='complete':phases.append(phase(None,FALLBACK_EFFORT,ANSWER_SECONDS,messages=messages,max_tokens=budget))
 row={'budget':budget,'repeat':repeat,'phases':phases,'status':phases[-1]['status'],'input_sha256':hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()}
 (OUT/f'budget{budget}-repeat{repeat}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps({'budget':budget,'repeat':repeat,'phases':[{k:p.get(k) for k in ['effort','status','incomplete_reason','finish_reason','done_received','seconds','reasoning_chars','answer_chars','usage','error']} for p in phases]},ensure_ascii=False),flush=True)
 return row
cases=[(b,r) for r in (1,2) for b in (4096,16384)]
(OUT/'config.json').write_text(json.dumps({'cases':cases,'model':'nex-n2.5-pro','policy':'medium60-none30','concurrency':2,'prompt':'identical official Rewrite 4096-token prompt across all API budgets'},indent=2),encoding='utf-8')
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool: rows=list(pool.map(run,cases))
(OUT/'results.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
