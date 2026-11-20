"""Mega published survey prompts + matched full-persona history. JSON outputs only."""
import argparse
import json
import os
import re
from pathlib import Path
from run_context import RESULTS
import pandas as pd
from sii import safe_chat

ROOT=Path(__file__).resolve().parents[1]

def parse_prediction(text,expected):
    cleaned=text.strip()
    if cleaned.startswith('```'): cleaned=cleaned.split('\n',1)[1].rsplit('```',1)[0].strip()
    parsed=json.loads(cleaned)
    if not isinstance(parsed,dict) or set(parsed)!=expected:
        raise ValueError('Question keys do not match the official survey template')
    if any(not isinstance(v,dict) or 'Answers' not in v for v in parsed.values()):
        raise ValueError('Missing answer structure')
    return parsed

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--study',default='context_effects')
    p.add_argument('--limit',type=int)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    personas={}
    for f in sorted((ROOT/'data/twin/full_persona/chunks').glob('*.parquet')):
        for row in pd.read_parquet(f).to_dict('records'):
            personas['pid_'+str(row['pid'])]=row['persona_text']
    df=pd.read_parquet(ROOT/f'data/mega/data/{a.study}-00000-of-00001.parquet')
    if a.limit: df=df.head(a.limit)
    dst=RESULTS/f'mega-{a.study}.jsonl'
    done=set()
    if dst.exists():
        done={json.loads(line)['id'] for line in dst.read_text(encoding='utf-8').splitlines()}
    with dst.open('a',encoding='utf-8') as out:
        for row in df.to_dict('records'):
            pid=row['PID']
            if pid not in personas: raise RuntimeError(f'Missing persona: {pid}; finish Twin download')
            if pid in done: continue
            # survey_text is the official masked input; ground-truth JSON is never sent.
            prompt='Predict this specific person\'s answers to the new survey using their previous responses.\n\nPERSONAL HISTORY:\n'+personas[pid]+'\n\nNEW SURVEY:\n'+row['survey_text']
            if a.dry_run:
                print(pid,len(prompt)); continue
            expected=set(re.findall(r'"(Q\d+)"\s*:',row['survey_text']))
            if not expected: raise RuntimeError('No question keys found in published template')
            parsed=None
            for attempt in range(3):
                response=safe_chat(os.environ['SII_MODEL'],prompt,sample_id=pid,benchmark='mega-'+a.study,max_tokens=32768,temperature=0.7)
                if response is None: continue
                try:
                    parsed=parse_prediction(response['choices'][0]['message']['content'],expected)
                    break
                except (json.JSONDecodeError,ValueError) as exc:
                    with (RESULTS/f'mega-{a.study}-errors.jsonl').open('a',encoding='utf-8') as errors:
                        errors.write(json.dumps({'id':pid,'attempt':attempt+1,'error':str(exc),'raw':response},ensure_ascii=False)+'\n')
            if parsed is None: continue
            out.write(json.dumps({'id':pid,'model':os.environ['SII_MODEL'],'prediction':parsed,'raw':response},ensure_ascii=False)+'\n');out.flush()

if __name__=='__main__':main()
