"""Prepare two deterministic Mega users and the last 60 legal answers."""
import json
import sys
import argparse
from dataio import BASE, digest, rows, split, twin, mega
from run import filter_target_history


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['pilot','preliminary','formal'],default='pilot');args=parser.parse_args()
    root=BASE.parent/'data'; study='context_effects'
    sys.path.insert(0,str(BASE.parent/'scripts'))
    from mega import prompt_for,schema_for,routed_questions
    targets=list(mega(root,[study])); test=sorted({u for u,*_ in targets if split(u)=='test'},key=digest)
    held=sorted({u for u,*_ in targets if split(u) in ('cal','test')} - set(test[:2]),key=lambda u:digest('mega-native60-20261126|'+u))
    ids={'pilot':test[:2],'preliminary':held[:20],'formal':held[20:120]}[args.stage]
    assert len(ids)=={'pilot':2,'preliminary':20,'formal':100}[args.stage]
    history=filter_target_history(dict(twin(root,users=set(ids))),targets)
    surveys={str(r['PID']).removeprefix('pid_'):r for r in rows(list((root/'mega/data').glob(study+'-*.parquet'))) if str(r['PID']).removeprefix('pid_') in ids}
    result=[]
    for u in ids:
        h=list(history[u].items())[-60:]; assert len(h)==60 and len(dict(h))==60
        r=surveys[u]; schema=schema_for(r['survey_text']); routed_questions(json.loads(r['survey_json_with_human_response']),schema)
        result.append({'id':str(r['PID']),'user_id':u,'history':h,'survey_text':r['survey_text'],'schema':schema,'prompt_template':prompt_for(study,r['survey_text'],'__FROZEN_PERSONA__')})
    dest=BASE/{'pilot':'mega_cost_pilot_2x60_v2.json','preliminary':'mega_method_preliminary_20x60_v1.json','formal':'mega_method_formal_100x60_v1.json'}[args.stage]
    with dest.open('x',encoding='utf8') as f:json.dump({'study':study,'history_protocol':'Last 60 legal single-choice responses in published full_persona questionnaire order after target prompt overlap removal; no fabricated timestamps','users':result},f,ensure_ascii=False,indent=2)
    print('Frozen',len(result),'users, each with 60 legal history records')


if __name__=='__main__':main()
