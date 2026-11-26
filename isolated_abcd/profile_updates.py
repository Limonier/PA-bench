"""C3/AC2: decision-linked user features and evidence-conditioned updates.

All representation learning, response priors and menu fitting use development
users only. The evaluation history is revealed incrementally in AC2.
"""
from collections import Counter
import random
import re
import numpy as np
from scipy.special import softmax
from scipy.optimize import minimize
from sklearn.decomposition import TruncatedSVD
from dataio import digest
from followups import prepare, fixed_policy, structured, question_relevance, entropy, BUDGETS, Acquisition
from mechanisms import menu_features, fit_menu
from run import dump


class EvidenceEncoder:
    """Learn categorical answer factors, not generic text sentiment or PID."""
    def __init__(self, hist, ids, keys):
        self.ids = ids
        self.keys = sorted(keys)
        self.blocks = {}
        columns = []
        for k in self.keys:
            values = Counter(hist[u].get(k) for u in ids)
            vals = sorted(v for v,n in values.items() if v is not None and n>=5)
            if len(vals)<2: continue
            frequency = np.array([values[v]/len(ids) for v in vals])
            self.blocks[k] = (vals,frequency,slice(len(columns),len(columns)+len(vals)))
            columns.extend((k,v) for v in vals)
        x=np.zeros((len(ids),len(columns)))
        for k,(vals,f,sl) in self.blocks.items():
            for i,u in enumerate(ids):
                a=hist[u].get(k)
                if a in vals: x[i,sl]=np.array([a==v for v in vals])-f
        if x.shape[1]<12: raise ValueError('Not enough categorical evidence')
        svd=TruncatedSVD(n_components=12,random_state=42).fit(x)
        self.contrib={}
        self.priors={}
        for k,(vals,f,sl) in self.blocks.items():
            self.contrib[k]={v:(np.eye(len(vals))[i]-f)@svd.components_[:,sl].T for i,v in enumerate(vals)}
            self.priors[k]=f/f.sum()
        self.scale=np.maximum((x@svd.components_.T).std(axis=0),.1)

    def update(self, state, key, answer):
        total,n=state
        v=self.contrib.get(key,{}).get(answer)
        return (total,n) if v is None else (total+v,n+1)

    def vector(self,state):
        total,n=state
        # Inverse square-root normalization retains evidence direction at short
        # budgets; explicit coverage lets the predictor learn shrinkage.
        return np.r_[total/self.scale/np.sqrt(max(n,1)),np.log1p(n)/5]

    def encode(self,h):
        state=(np.zeros(12),0)
        for k,a in h.items():state=self.update(state,k,a)
        return self.vector(state)


def decision_keys(policy):
    # Exclude demographics, identity, politics and religion; focus on behavior
    # and stated preferences rather than protected/sensitive user traits.
    ignore=re.compile(r'\b(race|ethnic|religion|religious|citizen|politic|political|sex|gender|sexual|age|income|employment|education|schooling|zip|marital|household)\b',re.I)
    return [k for k in policy.pool if not ignore.search(k.partition('|')[2])]


def features(q,v,conditional=False):
    base=menu_features(q,np.zeros(0),'shared')
    # All eight shared decision dimensions can interact with user evidence,
    # unlike C2, which only interacted with four relative-menu features.
    if conditional:
        category=np.array([bool(re.search(s,q.get('name',''),re.I)) for s in ('printer','tv','cell')],float)
        v=np.r_[v,*(v*c for c in category)]
    return np.column_stack([base,(base[:,:,None]*np.asarray(v)[None,None,:]).reshape(len(base),-1)])


def fit(rows,conditional=False):
    xs=[(features(q,v,conditional),y) for q,v,y in rows]
    def obj(w):
        loss=.5*(w@w); grad=w.copy()
        for x,y in xs:
            p=softmax(x@w); loss-=np.log(max(p[y],1e-12)); grad+=p@x-x[y]
        return loss,grad
    result=minimize(obj,np.zeros(xs[0][0].shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
    if not result.success:raise ValueError(result.message)
    return result.x


def predict(q,v,w,conditional=False):
    return softmax(features(q,v,conditional)@w)


def setup(args,out):
    hist,dev,held=prepare(args,True)
    policy=fixed_policy(hist,dev)
    keys=decision_keys(policy)
    encoder=EvidenceEncoder(hist,policy.ids,keys)
    out.cohort(policy.ids,'representation_and_menu_fit')
    dump(out.files['audit'],{'eligible_history_questions':len(keys),'encoded_questions':len(encoder.contrib),'latent_factors':12,'evaluation_users':len({u for u,*_ in held}),'evaluation_tracks':len(held)})
    return hist,dev,held,policy,encoder


def run_c3(args,model,out):
    hist,dev,held,policy,enc=setup(args,out)
    v={u:enc.encode(h) for u,h in hist.items()}
    st={u:structured(h) for u,h in hist.items()}
    joint={u:np.r_[v[u],st[u]] for u in v}
    shared=fit_menu([(q,st[u],y) for u,_,q,y in dev],'shared')
    c2=fit_menu([(q,st[u],y) for u,_,q,y in dev],'personal')
    ws={name:fit([(q,vs[u],y) for u,_,q,y in dev],cond) for name,vs,cond in [('categorical',v,False),('structured_full',st,False),('joint',joint,False),('conditional',joint,True)]}
    people=sorted({u for u,*_ in held},key=digest); swap=dict(zip(people,people[1:]+people[:1]))
    dump(out.files['audit'],{'shared':shared.tolist(),'structured_c2':c2.tolist(),'weights':{k:w.tolist() for k,w in ws.items()}})
    for u,s,q,y in held:
        p={'shared':softmax(menu_features(q,st[u],'shared')@shared),'structured_c2':softmax(menu_features(q,st[u],'personal')@c2)}
        p.update({name:predict(q,vs[u],ws[name],cond) for name,vs,cond in [('categorical',v,False),('structured_full',st,False),('joint',joint,False),('conditional',joint,True)]})
        p['conditional_shuffled']=predict(q,joint[swap[u]],ws['conditional'],True)
        out.record(u,f'{u}/{q["qid"]}',q,{'categorical':v[u].tolist(),'structured':st[u].tolist()},p,[y],{'menu':q['name'],'donor':swap[u]})


def partial_path(enc,h,sequence,q,w):
    state=(np.zeros(12),0); result={0:predict(q,enc.vector(state),w)}
    for i,k in enumerate(sequence,1):
        state=enc.update(state,k,h[k]);
        if i in BUDGETS:result[i]=predict(q,enc.vector(state),w)
    return result


def active_path(enc,h,keys,q,w):
    state=(np.zeros(12),0); sequence=[]; result={0:predict(q,enc.vector(state),w)}
    base=menu_features(q,np.zeros(0),'shared')
    bias=base@w[:base.shape[1]]
    coef=base@w[base.shape[1]:].reshape(base.shape[1],13)
    choices={k:np.array(list(enc.contrib[k].values())) for k in keys}
    # The predicted probabilities used to select a question are exactly the
    # probabilities used to score the resulting user model.
    for step in range(1,11):
        current=entropy(softmax(bias+coef@enc.vector(state)))
        def gain(k):
            total,n=state
            v=(total+choices[k])/enc.scale/np.sqrt(n+1)
            v=np.column_stack([v,np.full(len(v),np.log1p(n+1)/5)])
            probs=softmax(bias+v@coef.T,axis=1)
            return current-enc.priors[k]@entropy(probs)
        k=max((k for k in keys if k not in sequence),key=gain)
        sequence.append(k); state=enc.update(state,k,h[k])
        if step in BUDGETS:result[step]=predict(q,enc.vector(state),w)
    return sequence,result


def run_ac2(args,model,out):
    hist,dev,held,policy,enc=setup(args,out)
    # Train with the same partial evidence budgets used in evaluation, never
    # with evaluation answers or full-history embeddings of held-out people.
    rows=[]
    for u,s,q,y in dev:
        keys=sorted(set(hist[u])&set(enc.contrib)); rng=random.Random(digest(u+'|'+q['qid'])); rng.shuffle(keys)
        if len(keys)<10:continue
        state=(np.zeros(12),0); rows.append((q,enc.vector(state),y))
        for n,k in enumerate(keys[:10],1):
            state=enc.update(state,k,hist[u][k]);
            if n in BUDGETS:rows.append((q,enc.vector(state),y))
    w=fit(rows)
    full=fit([(q,enc.encode(hist[u]),y) for u,_,q,y in dev])
    # AC1-style posterior baseline on the same revealed sequences.
    acq=Acquisition(hist,policy); old={u:structured(hist[u]) for u in policy.ids}
    prior_w=fit_menu([(q,old[u],y) for u,_,q,y in dev],'personal')
    dump(out.files['audit'],{'partial_training_rows':len(rows),'weight':w.tolist(),'full_weight':full.tolist(),'reference_weight':prior_w.tolist()})
    for u,s,q,y in held:
        h=hist[u]; keys=sorted(set(h)&set(enc.contrib));
        if len(keys)<10:raise ValueError('Insufficient eligible encoded questions')
        fixed=[k for k in policy.fixed if k in keys]; fixed=(fixed+[k for k in keys if k not in fixed])[:10]
        rand=keys.copy(); random.Random(digest(u)).shuffle(rand); rand=rand[:10]
        rel=question_relevance(q,keys)[:10]
        active,active_p=active_path(enc,h,keys,q,w)
        seqs={'fixed':fixed,'random':rand,'relevant':rel,'active':active}
        p={'empty':active_p[0],'full_history':predict(q,enc.encode(h),full)}
        old_predictions=np.array([softmax(menu_features(q,old[v],'personal')@prior_w) for v in acq.ids])
        p['posterior_empty']=old_predictions.mean(axis=0)
        for name,seq in seqs.items():
            direct=active_p if name=='active' else partial_path(enc,h,seq,q,w)
            reference=acq.posterior_path(h,seq,old_predictions)
            for n in BUDGETS:p[f'{name}@{n}']=direct[n]; p[f'posterior_{name}@{n}']=reference[n]
        out.record(u,f'{u}/{q["qid"]}',q,{k:[(i,h[i]) for i in seq] for k,seq in seqs.items()},p,[y],{'menu':q['name']})
