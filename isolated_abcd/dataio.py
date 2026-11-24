"""Read-only benchmark adapters. No imports from the main experiment."""
import hashlib
import json
import re
from pathlib import Path
from collections import Counter
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent
AUDIT = Counter()


def digest(s):
    return hashlib.sha256(s.encode()).hexdigest()


def split(pid):
    n = int(digest(str(pid)), 16) % 100
    return 'dev' if n < 60 else 'cal' if n < 80 else 'test'


def rows(files, columns=None):
    if not files:
        raise FileNotFoundError("No input parquet shards")
    for path in sorted(files):
        for batch in pq.ParquetFile(path).iter_batches(batch_size=8, columns=columns):
            yield from batch.to_pylist()


def choices(document):
    """Walk realized branches only. Preserve displayed option order."""
    if isinstance(document, str):
        document = json.loads(document)
    if isinstance(document, list):
        for item in document:
            yield from choices(item)
        return
    for key in ('Elements', 'Questions'):
        yield from choices(document[key]) if key in document else ()
    if document.get('QuestionType')=='Matrix' and document.get('Settings',{}).get('SubSelector')=='SingleAnswer':
        labels=(document.get('Answers') or {}).get('SelectedText')
        rows=document.get('Rows',[])
        ids=document.get('RowsID',rows)
        opts=document.get('Columns',[])
        if not isinstance(labels,list) or len(labels)!=len(rows) or len(ids)!=len(rows) or len(set(rows))!=len(rows):
            AUDIT['ambiguous_matrix']+=1
            return
        for rid,row,ans in zip(ids,rows,labels):
            yield from choices({'QuestionID':document['QuestionID']+'#'+str(rid),'QuestionType':'MC',
                'Settings':{'Selector':'SAVR'},'QuestionText':document['QuestionText']+' '+row,
                'Options':opts,'Answers':{'SelectedText':ans}})
        return
    if document.get('QuestionType') != 'MC':
        return
    q = document
    opts = q.get('Options', [])
    ans = q.get('Answers') or {}
    if not isinstance(ans, dict) or not str(q.get('Settings', {}).get('Selector','')).startswith('SA'):
        AUDIT['not_single'] += 1
        return
    label = ans.get('SelectedText')
    if not all(isinstance(o,str) for o in opts) or not 2 <= len(opts) <= 26 or len(set(opts)) != len(opts) or label not in opts:
        AUDIT['unmapped'] += 1
        return
    yield dict(qid=q['QuestionID'], name=q.get('QuestionName',''), text=q['QuestionText'], options=opts), opts.index(label)


def history(doc):
    return {q['qid']+'|'+q['text']:q['options'][y] for q,y in choices(doc)}


def twin(root, wave=False, users=None):
    folder = 'wave_split' if wave else 'full_persona'
    cols = ['pid','wave1_3_persona_json','wave4_Q_wave1_3_A','wave4_Q_wave4_A'] if wave else ['pid','persona_json']
    for r in rows(list((root/'twin'/folder/'chunks').glob('*.parquet')), cols):
        pid = str(r['pid'])
        if users is not None and pid not in users:
            continue
        if not wave:
            yield pid,history(r[cols[1]])
            continue
        def paired(s):
            result = {}
            for q,y in choices(s):
                key=(q['qid'],q['text'],tuple(sorted(q['options'])))
                result[key] = None if key in result else (q,y)
            return result
        early, late = paired(r[cols[2]]),paired(r[cols[3]])
        hist=history(r[cols[1]])
        for key in early:
            if not early[key] or not late.get(key):
                AUDIT['unpaired'] += 1
                continue
            q,y=early[key]
            q2,y2=late[key]
            yield pid,hist,q,y,q['options'].index(q2['options'][y2])


def mega(root, studies):
    for study in studies:
        for r in rows(list((root/'mega'/'data').glob(study+'-*.parquet'))):
            pid=str(r['PID']).removeprefix('pid_')
            for q,y in choices(r['survey_json_with_human_response']):
                if study=='context_effects' and not re.match('^(Bin|Comp|Attr)[._]',q['name'],re.I):
                    continue
                yield pid,study,q,y


EVENT = re.compile(r'Chose (?:the )?<<([^>]+)>>(?: option)? and received ([+-]?[\d.]+) points?\.')

def trajectory(text):
    lines=[l.strip() for l in text.splitlines() if l.strip().startswith('Chose')]
    matches=[EVENT.fullmatch(l) for l in lines]
    if not lines or not all(matches):
        raise ValueError("Unknown bandit transcript format")
    return [(m[1],float(m[2])) for m in matches]


def bandits(root, source, subset):
    seen=set()
    for r in rows(list(root.glob(subset+'*.parquet'))):
        if r['prev_task_name']!=source or r['next_task_name']!='two_armed_bandit':
            continue
        msgs={m['role']:m['content'] for m in r['message']}
        src=msgs['user'].split('--- PREVIOUS EXPERIMENT ---')[1].split('--- END PREVIOUS EXPERIMENT ---')[0]
        target=trajectory(msgs['assistant'])
        key=(r['participant'],digest(str(target)))
        if key in seen:
            continue
        seen.add(key)
        yield r['participant'],trajectory(src),target

