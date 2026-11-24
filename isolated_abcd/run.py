"""Independent ABCD entry point. No automatic launches or shared output."""
import argparse
from collections import defaultdict, Counter
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
from scipy.special import softmax
from dataio import BASE, AUDIT, bandits, digest, mega, split, twin
from scorer import retrieve
from mechanisms import QuestionPolicy, calibrators, fit_menu, menu_features, fit_learning, replay

def dump(file,obj):
    file.write(json.dumps(obj,ensure_ascii=False,allow_nan=False)+'\n')


class Model:
    def __init__(self,args):
        self.seconds=0.
        self.calls=0
        env=dict(os.environ, PYTHONUTF8='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                 HF_HOME=str(BASE/'local'/'hf'),PYTHONDONTWRITEBYTECODE='1')
        self.worker=subprocess.Popen([args.model_python,'-B',str(BASE/'scorer.py'),args.model,args.arch],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,encoding='utf8',env=env,cwd=BASE)
    def close(self):
        self.worker.stdin.close()
        self.worker.wait()
    def score(self,q,h,mode):
        start=time.monotonic()
        ev=retrieve(h,q['text']+' '+' '.join(q['options']),mode)
        dump(self.worker.stdin,dict(q={k:q[k] for k in ('text','options')},evidence=ev,mode=mode))
        self.worker.stdin.flush()
        line=self.worker.stdout.readline()
        if not line: raise RuntimeError('Model worker stopped before returning candidate probabilities')
        self.calls+=1
        self.seconds+=time.monotonic()-start
        p=np.array(json.loads(line))
        if len(p)!=len(q['options']) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1):
            raise ValueError('Invalid candidate scores')
        return p


class Output:
    def __init__(self,path,stack):
        self.files={k:stack.enter_context((path/(k+'.jsonl')).open('x',encoding='utf8')) for k in ('predictions','labels','inputs','splits','audit')}
        self.labels={}
        self.people=set()
    def cohort(self,users,role):
        for pid in sorted(set(users)):
            dump(self.files['splits'],dict(pid=pid,split=split(pid),role=role))
    def record(self,pid,key,q,h,results,y,groups=None):
        if key in self.labels: raise ValueError('Duplicate sample '+key)
        if pid not in self.people:
            dump(self.files['splits'],dict(pid=pid,split=split(pid)))
            self.people.add(pid)
        dump(self.files['inputs'],dict(id=key,pid=pid,question=q,history=h,groups=groups or {}))
        for m,p in results.items():
            dump(self.files['predictions'],dict(id=key,pid=pid,method=m,p=np.asarray(p).tolist()))
        self.labels[key]=y
        dump(self.files['labels'],dict(id=key,actual=y))


def select_people(hist):
    return sorted(hist,key=lambda u:digest(u))

def matched(hist):
    """Nearest other held-out history: TF-IDF topics/answer frequencies + length."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    people=select_people(hist)
    if len(people)<2: raise ValueError('Need at least two users for matched control')
    docs=[' '.join(k+' '+str(v)+' '+str(v) for k,v in hist[u].items()) or '_empty_' for u in people]
    x=TfidfVectorizer(max_features=8192).fit_transform(docs)
    sim=(x@x.T).toarray()
    size=np.log1p([len(hist[u]) for u in people])
    sim-=.1*np.abs(size[:,None]-size[None,:])
    np.fill_diagonal(sim,-np.inf)
    return {u:people[j] for u,j in zip(people,sim.argmax(1))}


def run_a(args,model,out):
    cases=list(mega(args.data,args.studies))
    ids={u for u,*_ in cases}
    hist=dict(twin(args.data,users=ids))
    # Exclude overlapping prompts, even if their IDs differ across studies.
    from difflib import SequenceMatcher
    import re
    norm=lambda s:' '.join(re.findall(r'\w+',s.lower()))
    tasks={norm(q['text']) for _,_,q,_ in cases}
    from functools import lru_cache
    @lru_cache(None)
    def allowed(key):
        t=norm(key.partition('|')[2])
        return not any(t==v or (min(len(t),len(v))>=35 and (t in v or v in t))
            or (.8<len(t)/max(len(v),1)<1.25 and SequenceMatcher(None,t,v).quick_ratio()>.88 and SequenceMatcher(None,t,v).ratio()>.88) for v in tasks)
    hist={u:{k:v for k,v in h.items() if allowed(k)} for u,h in hist.items()}
    train=defaultdict(list)
    for u,s,q,y in cases:
        if u in hist and split(u)=='dev' and s!=args.holdout:
            train[u].append((s+'|'+q['qid'],q['options'][y]))
    if not train: raise ValueError('No development targets outside held-out study')
    route=QuestionPolicy(hist,train,10)
    out.cohort(train,'policy_fit')
    dump(out.files['audit'],dict(fixed_order=route.fixed,pool_size=len(route.pool)))
    routes={}
    for u in hist:
        if split(u)=='test' and len(set(hist[u])&set(route.pool))>=10:
            routes[u]={k:route.route(hist[u],k,10,random.Random(digest(u))) for k in ('random','fixed','information','adaptive')}
    three={u:dict(list(v['adaptive'].items())[:3]) for u,v in routes.items()}
    donors=matched(three)
    for u,s,q,y in cases:
        if split(u)!='test' or u not in hist:continue
        if len(set(hist[u])&set(route.pool))<10:
            AUDIT['A_incomplete_pool']+=1
            continue
        key=f'{s}/{u}/{q["qid"]}'
        preds={}
        trace={}
        for policy in ('random','fixed','information','adaptive'):
            seq=routes[u][policy]
            trace[policy]=list(seq.items())
            for n in (0,1,3,5,10):
                if n==0 and 'empty' in preds:continue
                preds[f'{policy}@{n}' if n else 'empty']=model.score(q,dict(list(seq.items())[:n]),args.reader)
        preds['matched@3']=model.score(q,three[donors[u]],args.reader)
        out.record(u,key,q,trace,preds,[y],{'study':s,'heldout_study':s==args.holdout,'matched':donors[u]})


def run_b(args,model,out):
    dev,cal=[],[]
    fitted=set()
    def features(h,p):
        return [np.log1p(len(h)),len(p),float(np.diff(np.sort(p)[-2:])[0])]
    held={}
    for u,h,q,a,b in twin(args.data,True):
        if split(u)=='test':held.setdefault(u,h)
        if split(u) in ('dev','cal'):
            fitted.add(u)
            p=model.score(q,h,args.reader)
            (dev if split(u)=='dev' else cal).append((p,model.score(q,{},'empty'),features(h,p),a,b))
    if not dev or not cal:raise ValueError('Empty development/calibration fold')
    out.cohort(fitted,'reliability_fit_or_calibration')
    predict=calibrators(dev,cal)
    donors=matched(held)
    for u,h,q,a,b in twin(args.data,True):
        if split(u)!='test':continue
        p=model.score(q,h,args.reader)
        results=predict(p,model.score(q,{},'empty'),features(h,p))
        results['matched']=model.score(q,held[donors[u]],args.reader)
        key=f'{u}/{q["qid"]}/'+digest(q['text'])[:12]
        out.record(u,key,q,retrieve(h,q['text']+' '+' '.join(q['options']),args.reader),results,[a,b],{'matched':donors[u]})
        copy=np.zeros(len(p)); copy[a]=1
        seen=dict(q,text=q['text']+'\nThis participant previously answered: '+q['options'][a])
        out.record(u,key+'/retest_seen',seen,{'earlier_answer':a},{'copy_previous':copy,'personal_seen':model.score(seen,h,args.reader)},[b])


def profile(h):
    # Stable low-dimensional projection of observed source responses, not PID.
    v=np.zeros(12)
    for k,a in h.items():
        n=int(digest(k+'='+a)[:12],16)
        v[n%12]+=1 if (n//12)%2 else -1
    return v/max(np.linalg.norm(v),1.)

def run_c(args,model,out):
    cases=list(mega(args.data,['context_effects']))
    hist=dict(twin(args.data,users={u for u,*_ in cases}))
    vs={u:profile(h) for u,h in hist.items()}
    dev=[(q,vs[u],y) for u,_,q,y in cases if split(u)=='dev' and u in vs]
    ws={k:fit_menu(dev,k) for k in ('independent','shared','personal')}
    out.cohort((u for u in hist if split(u)=='dev'),'menu_fit')
    dump(out.files['audit'],{'menu_weights':{k:w.tolist() for k,w in ws.items()}})
    us=select_people({u for u in vs if split(u)=='test'})
    swap=dict(zip(us,us[1:]+us[:1]))
    donors=matched({u:hist[u] for u in us})
    for u,_,q,y in cases:
        if u not in swap: continue
        p={k:softmax(menu_features(q,vs[u],k)@w) for k,w in ws.items()}
        p['shuffled']=softmax(menu_features(q,vs[swap[u]],'personal')@ws['personal'])
        p['whole_menu']=model.score(q,hist[u],args.reader)
        p['matched_menu']=model.score(q,hist[donors[u]],args.reader)
        out.record(u,f'{u}/{q["qid"]}',q,vs[u].tolist(),p,[y],{'menu':q['name'],'donor':swap[u],'matched':donors[u]})


def run_d(args,model,out):
    src=args.source
    choices=['safe','risky']
    dev=[(u,s,t) for u,s,t in bandits(args.cross,src,'train') if split(u)=='dev']
    if not dev:raise ValueError('No D development trajectories')
    scale=20.
    temp,stick=1.,.4
    alpha=fit_learning([t for _,_,t in dev],choices,temp,stick,scale)
    out.cohort((u for u,_,_ in dev),'learning_rate_fit')
    dump(out.files['audit'],dict(shared_alpha=alpha,temperature=temp,stickiness=stick,reward_scale=scale))
    test=[(u,s,t) for u,s,t in bandits(args.cross,src,'test') if split(u)=='test']
    if len(test)<2:raise ValueError('Need at least two held-out participants')
    estimates={}
    for u,s,t in test:
        acts=sorted({a for a,_ in s})
        estimates[u]=fit_learning([s],acts,temp,stick,100. if src!='two_armed_bandit' else scale,alpha)
    people=select_people(estimates)
    swaps=dict(zip(people,people[1:]+people[:1]))
    for u,s,t in test:
        seq={k:replay(t,choices,a,temp,stick,scale) for k,a in [('shared',alpha),('personal',estimates[u]),('shuffled',estimates[swaps[u]])]}
        for i,(act,reward) in enumerate(t):
            p={k:next(v) for k,v in seq.items()}
            copy=np.ones(2)/2 if i==0 else np.eye(2)[choices.index(t[i-1][0])]
            q={'text':'Choose the safe (0 points) or risky (variable reward) option on the next trial.','options':choices}
            h={f'{phase} trial {j+1}':f'{a}, reward {r}' for phase,tr in [('source',s),('target',t[:i])] for j,(a,r) in enumerate(tr)}
            p.update(copy=copy,trajectory=model.score(q,h,args.reader))
            out.record(u,f'{u}/{digest(str(t))[:16]}/{i}',q,h,p,[choices.index(act)],{'previous_reward':t[i-1][1] if i else None,'previous_choice':choices.index(t[i-1][0]) if i else None,'source':src})

