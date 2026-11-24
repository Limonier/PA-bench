import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone
import numpy as np
from dataio import BASE, AUDIT, mega, twin, bandits, digest
from run import Model, Output, run_a, run_b, run_c, run_d, profile
from mechanisms import replay, menu_features
from summary import summarize


def smoke(args,model):
    u,_,q,y=next(mega(args.data,['context_effects']))
    _,h=next(twin(args.data,users={u}))
    history=dict(list(h.items())[:3])
    assert y in range(len(model.score(q,history,args.reader)))
    print('A OK: linked PID, 3 historical answers, one actual Mega menu -> frozen candidate scores')
    _,hb,qb,a,b=next(twin(args.data,True))
    p=model.score(qb,hb,args.reader)
    assert a in range(len(p)) and b in range(len(p))
    print('B OK: canonical options, early + retest labels, one prediction')
    sizes={k:menu_features(q,profile(h),k).shape for k in ('independent','shared','personal')}
    assert all(n==len(q['options']) for n,_ in sizes.values())
    print('C OK: actual menu -> three design matrices; shares A candidate-scoring route',sizes)
    for src in ('two_armed_bandit','restless_bandit'):
        _,s,t=next(bandits(args.cross,src,'test'))
        prefix=t[:1]
        gen=replay(t[:2],['safe','risky'],.3,1.,.4,20.)
        assert np.isclose(next(gen).sum(),1.) and np.isclose(next(gen).sum(),1.)
        qd={'text':'Predict the next bandit action','options':['safe','risky']}
        model.score(qd,{'previous action and reward':str(prefix)},args.reader)
        print('D OK:',src,'-> two_armed_bandit; one visible feedback step, one next-action call')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('command',choices=['smoke','run','select-reader'])
    ap.add_argument('experiment',nargs='?',choices=list('ABCD'))
    ap.add_argument('--data',type=Path,default=BASE.parent/'data')
    ap.add_argument('--cross',type=Path,default=BASE/'local/cross')
    ap.add_argument('--model-python',default=str(BASE.parent/'.retrieval-venv/Scripts/python.exe'))
    ap.add_argument('--model',default=str(BASE.parent/'data/models/roberta-large'))
    ap.add_argument('--arch',choices=['masked','causal'],default='masked')
    ap.add_argument('--reader',choices=['bm25','empty','gpg'],default='bm25')
    ap.add_argument('--studies',nargs='+',default=['context_effects','default_eric'])
    ap.add_argument('--holdout',default='default_eric')
    ap.add_argument('--source',choices=['two_armed_bandit','restless_bandit','correlated_bandit'],default='two_armed_bandit')
    ap.add_argument('--dev-users',type=int,default=32,help='Fixed dev cohort for reader selection')
    args=ap.parse_args()
    if args.command=='run' and not args.experiment: ap.error('Select A, B, C or D')
    if args.reader=='gpg' and args.arch!='causal':ap.error('GPG needs a local generative checkpoint')
    if args.experiment=='A' and (args.holdout not in args.studies or len(set(args.studies))<2):ap.error('A requires a held-out study and development study')
    args.model=str(Path(args.model).resolve())
    with ExitStack() as stack:
        model=Model(args); stack.callback(model.close)
        if args.command=='smoke':
            smoke(args,model)
            return
        if args.command=='select-reader':
            from dataio import split
            cases=[(u,q,y) for u,_,q,y in mega(args.data,['context_effects']) if split(u)=='dev']
            ids=set(sorted({u for u,_,_ in cases},key=digest)[:args.dev_users])
            hist=dict(twin(args.data,users=ids))
            scores={}
            for mode in (['empty','bm25','gpg'] if args.arch=='causal' else ['empty','bm25']):
                by={u:[] for u in ids}
                for u,q,y in cases:
                    if u in ids:by[u].append(int(model.score(q,hist[u],mode).argmax()==y))
                scores[mode]=float(np.mean([np.mean(v) for v in by.values()]))
            print(json.dumps({'dev_users':sorted(ids),'agreement':scores,'reader':max(scores,key=scores.get)}))
            return
        path=BASE/'runs'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-'+args.experiment)
        path.mkdir(parents=True,exist_ok=False)
        def save(name,obj):
            (path/name).write_text(json.dumps(obj,default=str,indent=2,ensure_ascii=False),encoding='utf8')
        import hashlib
        def sha(p):
            with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
        files=list(BASE.glob('*.py'))+list(Path(args.model).glob('config.json'))+list(Path(args.model).glob('*.safetensors'))
        if args.experiment=='D':files+=list(args.cross.glob('*.parquet'))
        else:
            folder='wave_split' if args.experiment=='B' else 'full_persona'
            files+=list((args.data/'twin'/folder).rglob('*.parquet'))
            for s in (args.studies if args.experiment=='A' else ['context_effects'] if args.experiment=='C' else []):
                files+=list((args.data/'mega/data').glob(s+'-*.parquet'))
        save('config.json',{'args':vars(args),'sha256':{str(p):sha(p) for p in files}})
        try:
            with ExitStack() as handles:
                out=Output(path,handles)
                dict(A=run_a,B=run_b,C=run_c,D=run_d)[args.experiment](args,model,out)
            if not out.labels:raise ValueError('Empty final cohort')
            save('summary.json',summarize(path))
            save('status.json',{'status':'complete','adapter_counts':dict(AUDIT),'model_calls':model.calls,'model_wall_seconds':model.seconds})
        except BaseException as exc:
            save('status.json',{'status':'incomplete','error':repr(exc),'excluded':dict(AUDIT)})
            raise
        print(path)

if __name__=='__main__':main()

