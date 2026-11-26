"""Learn decision-related evidence and posterior preference parameters on dev users."""
from collections import Counter, defaultdict
import random
import re
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from dataio import digest
from followups import prepare, fixed_policy, structured, entropy, question_relevance, BUDGETS
from mechanisms import menu_features, fit_menu
from profile_updates import fit, predict, decision_keys
from run import dump


def design(q):
    return menu_features(q,np.zeros(0),'shared')


def fit_preferences(dev,hist,keys):
    """Group utility plus shrunk dev-person residuals; no held-out target labels."""
    shared=fit_menu([(q,structured(hist[u]),y) for u,_,q,y in dev],'shared')
    by=defaultdict(list)
    for u,_,q,y in dev:by[u].append((design(q),y))
    ids=sorted(by); params=[]
    for u in ids:
        def objective(r):
            loss=2*(r@r); grad=4*r
            for x,y in by[u]:
                p=softmax(x@(shared+r)); loss-=np.log(max(p[y],1e-12)); grad+=p@x-x[y]
            return loss,grad
        r=minimize(objective,np.zeros(len(shared)),jac=True,method='L-BFGS-B')
        if not r.success:raise ValueError(r.message)
        params.append(r.x)
    params=np.array(params)
    # Evidence value conditional means with empirical-Bayes shrinkage.
    effects={}; priors={}
    for k in sorted(keys):
        groups=defaultdict(list)
        for i,u in enumerate(ids):
            if k in hist[u]:groups[hist[u][k]].append(i)
        groups={a:g for a,g in groups.items() if len(g)>=8}
        if len(groups)<2:continue
        effects[k]={a:params[g].sum(axis=0)/(len(g)+20) for a,g in groups.items()}
        n=np.array([len(g) for g in groups.values()],float);priors[k]=n/n.sum()
    return shared,ids,params,effects,priors


def domain_profile(h):
    domains=('price|cost|spend|afford|budget|saving','quality|durab|performance|reliab','risk|gambl|lottery|uncertain','time|convenien|effort|delay','choice|option|trade.off|prefer','environment|energy|recycl')
    # Keep positive/negative statements and numeric scales separate from counts.
    result=[]
    for domain in domains:
        answers=[a.lower().strip() for k,a in h.items() if re.search(domain,k.split('|',1)[-1],re.I)]
        pos=sum(bool(re.search(r'\b(agree|yes|often|always|important|likely)\b',a)) and not bool(re.search(r'\b(disagree|not|unlikely|never)\b',a)) for a in answers)
        neg=sum(bool(re.search(r'\b(disagree|no|never|unlikely|rarely|not)\b',a)) for a in answers)
        result.extend([(pos-neg)/(len(answers)+3),np.log1p(len(answers))/5])
    return np.asarray(result)


def residual(h,effects):
    observed=[effects[k][a] for k,a in h.items() if k in effects and a in effects[k]]
    return np.sum(observed,axis=0)/np.sqrt(len(observed)+4) if observed else np.zeros(8)


def run_decision(args,model,out):
    hist,dev,held=prepare(args,True); policy=fixed_policy(hist,dev)
    keys=decision_keys(policy); shared,ids,params,effects,priors=fit_preferences(dev,hist,keys)
    out.cohort(ids,'decision_fit'); dump(out.files['audit'],{'eligible_questions':len(effects),'dev_users':len(ids),'shared':shared.tolist(),'stage':args.experiment})
    people=sorted({u for u,*_ in held},key=digest); swap=dict(zip(people,people[1:]+people[:1]))
    if args.experiment=='C4':
        reps={u:domain_profile(h) for u,h in hist.items()}; baseline={u:structured(h) for u,h in hist.items()}
        w=fit([(q,reps[u],y) for u,_,q,y in dev]); old=fit_menu([(q,baseline[u],y) for u,_,q,y in dev],'personal')
        for u,s,q,y in held:
            p={'shared':softmax(design(q)@shared),'structured_c2':softmax(menu_features(q,baseline[u],'personal')@old),'decision_domains':predict(q,reps[u],w),'shuffled':predict(q,reps[swap[u]],w)}
            out.record(u,f'{u}/{q["qid"]}',q,reps[u].tolist(),p,[y],{'menu':q['name']})
        return
    if args.experiment=='C5':
        for u,s,q,y in held:
            x=design(q); r=residual(hist[u],effects); donor=residual(hist[swap[u]],effects)
            p={'shared':softmax(x@shared),'parameter':softmax(x@(shared+r)),'parameter_shuffled':softmax(x@(shared+donor))}
            out.record(u,f'{u}/{q["qid"]}',q,r.tolist(),p,[y],{'menu':q['name']})
        return
    for u,s,q,y in held:
        h=hist[u]; x=design(q); available=sorted(set(h)&set(effects)); assert len(available)>=10
        fixed=[k for k in policy.fixed if k in available]; fixed=(fixed+[k for k in available if k not in fixed])[:10]
        random_keys=random.Random(digest(u)).sample(available,10); relevant=question_relevance(q,available)[:10]
        seqs={'fixed':fixed,'random':random_keys,'relevant':relevant,'active':[]}
        p={'empty':softmax(x@shared),'full_history':softmax(x@(shared+residual(h,effects)))}
        for name,seq in seqs.items():
            total=np.zeros_like(shared); count=0
            for step in range(1,11):
                if name=='active':
                    def gain(k):
                        probabilities=softmax((shared+(total+np.array(list(effects[k].values())))/np.sqrt(count+5))@x.T,axis=1)
                        return -priors[k]@entropy(probabilities)
                    k=max((k for k in available if k not in seq),key=gain);seq.append(k)
                else:k=seq[step-1]
                if h[k] in effects[k]:total+=effects[k][h[k]];count+=1
                if step in BUDGETS:p[f'{name}@{step}']=softmax(x@(shared+total/np.sqrt(count+4)))
        out.record(u,f'{u}/{q["qid"]}',q,{k:[(i,h[i]) for i in seq] for k,seq in seqs.items()},p,[y],{'menu':q['name']})
