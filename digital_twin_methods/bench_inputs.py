"""Convert only visible benchmark history to a common record format.

Current questions are passed to retrieve(), never to update(); current golds are not accepted.
"""
import json
from dataclasses import replace
from .methods import History


def _text(value):
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def lamp_history(task, question):
    """Use the official profile, retaining historical labels and scores."""
    fields = {1:('abstract','title'), 2:('description','tag'), 3:('text','score'),
              4:('text','title'), 5:('abstract','title'), 7:('text','text')}
    query, chosen = fields[task]
    return [History(text=_text({k:v for k,v in r.items() if k != 'id'}), timestamp=r.get('date',''),
                    query=_text(r[query]), chosen=_text(r[chosen])) for r in question['profile']]


def mega_history(persona_json):
    """Twin historical questionnaire only; do not pass Mega survey answers here."""
    blocks = json.loads(persona_json) if isinstance(persona_json,str) else persona_json
    records = []
    for block in blocks:
        for question in block.get('Questions',[]):
            if not question.get('Answers'):
                continue
            context = {k:question[k] for k in ('QuestionText','QuestionType','Options','Rows','Columns','Answers') if k in question}
            q = _text({k:v for k,v in context.items() if k!='Answers'})
            records.append(History(text=_text(context), query=q, chosen=_text(question['Answers'])))
    return records


def fingertip_history(history, user_id, cutoff, limit=20):
    """Mirror the existing official zero-screenshot historical cutoff."""
    rows = sorted((r for r in history if str(r['user_id']) == str(user_id) and r['time'] < cutoff), key=lambda r:r['time'])
    if limit is not None:
        rows = rows[-limit:]
    return [History(text=f"时间：{r['time']}，地点：{r['scenario']}，意图：{r['intentDescription']}",timestamp=r['time'],
                    query=f"时间：{r['time']}，地点：{r['scenario']}", chosen=r['intentDescription']) for r in rows]


def vita_history(interactions):
    """Preserve time and role; only the historical interactions field is accepted."""
    records = []
    for item in interactions:
        if hasattr(item, 'model_dump'):
            item = item.model_dump()
        if 'behavior' in item or 'dialogue' in item:
            date = str(item.get('date',''))
            for behavior in item.get('behavior',[]):
                records.append(History(text=_text(behavior),timestamp=date,query=str(behavior.get('behavior_type','Observed activity'))))
            pair = None
            for message in item.get('dialogue',[]):
                role = message.get('role','user')
                content = _text(message.get('content',''))
                if role in ('assistant','agent'):
                    if pair is not None:
                        r = records[pair]
                        records[pair] = replace(r, assistant=(r.assistant+'\n'+content).strip())
                    else:
                        records.append(History(content,date,role='assistant'))
                else:
                    records.append(History(content,date,role='user'))
                    pair = len(records)-1
        else:
            records.append(History(_text(item),str(item.get('timestamp',''))))
    return records
