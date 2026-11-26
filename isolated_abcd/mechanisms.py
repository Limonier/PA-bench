"""Small trainable mechanisms; gold is only passed to fitting/evaluation."""
from collections import Counter, defaultdict
import math
import re
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax

class QuestionPolicy:
    """Greedy dev-set risk reduction. Nodes back off to the fixed policy."""
    def __init__(self, histories, targets, budgets):
        self.h=histories
        self.y=targets
        self.ids=sorted(targets)
        common=Counter(q for u in self.ids for q in histories[u])
        self.pool=sorted(q for q,n in common.items() if n>=.8*len(self.ids))
        if len(self.pool)<budgets:
            raise ValueError('Insufficient shared historical questions')
        self.fixed=[]
        groups=[self.ids]
        for _ in range(budgets):
            key=min((q for q in self.pool if q not in self.fixed),key=lambda q:sum(self.cost(g,q) for g in groups))
            self.fixed.append(key)
            groups=[p for g in groups for p in self.parts(g,key).values()]

    def parts(self, ids, q):
        groups=defaultdict(list)
        for u in ids:
            groups[self.h[u].get(q)].append(u)
        return groups

    def cost(self, ids, q):
        cost=0
        for g in self.parts(ids,q).values():
            counts=defaultdict(Counter)
            for u in g:
                for key,answer in self.y[u]:
                    counts[key][answer]+=1
            cost+=sum(c.total()-max(c.values()) for c in counts.values())
        return cost

    def route(self, history, policy, budget, rng):
        available=set(history)&set(self.pool)
        if len(available)<budget:
            raise ValueError('Insufficient eligible answers for budget')
        fixed=self.fixed+sorted(available-set(self.fixed))
        selected={}
        ref=self.ids
        random_order=sorted(available)
        rng.shuffle(random_order)
        for _ in range(budget):
            opts=sorted(available-selected.keys())
            if policy=='random':
                q=random_order[len(selected)]
            elif policy=='fixed' or len(ref)<8:
                q=next(k for k in fixed if k in opts)
            elif policy=='adaptive':
                q=min(opts,key=lambda k:self.cost(ref,k))
            else:
                def entropy(k):
                    return -sum((len(g)/len(ref))*math.log(len(g)/len(ref)) for g in self.parts(ref,k).values())
                q=max(opts,key=entropy)
            selected[q]=history[q]
            ref=[u for u in ref if self.h[u].get(q)==history[q]]
        return selected


def mix(p,b,w):
    return w*p+(1-w)*b


def calibrators(development,cal):
    from sklearn.linear_model import LogisticRegression
    # Each tuple holds (personal, reference, features, early, late).
    x=np.asarray([v[2] for v in development])
    repeat=np.asarray([a==b for _,_,_,a,b in development],int)
    if len(set(repeat))>1:
        reg=LogisticRegression().fit(x,repeat)
        reliability=lambda z:float(reg.predict_proba([z])[0,1])
    else:
        reliability=lambda z:float(repeat.mean())
    def loss(fn):
        return np.mean([-math.log(max(fn(p,b,z)[i],1e-12)) for p,b,z,a,l in cal for i in (a,l)])
    ts=[.5,.75,1.,1.5,2.,3.]
    temp=min(ts,key=lambda t:loss(lambda p,b,z:softmax(np.log(np.maximum(p,1e-12))/t)))
    grid=np.linspace(0,1,11)
    weight=min(grid,key=lambda w:loss(lambda p,b,z:mix(p,b,w)))
    flags=[reliability(z) for _,_,z,_,_ in cal]
    cut=np.median(flags)
    weights=[]
    for flag in (False,True):
        group=[v for v,s in zip(cal,flags) if (s>cut)==flag]
        weights.append(min(grid,key=lambda w:sum(-math.log(max(mix(p,b,w)[j],1e-12)) for p,b,z,a,l in group for j in (a,l))) if group else weight)
    return lambda p,b,z:dict(personal=p,reference=b,temperature=softmax(np.log(np.maximum(p,1e-12))/temp),fixed=mix(p,b,weight),adaptive=mix(p,b,weights[int(reliability(z)>cut)]))


def menu_features(q,profile,kind):
    prices=np.array([float(n.replace(',','')) for n in re.findall(r'\bPrice:\s*\$([\d,.]+)',q['text'],re.I)])
    if len(prices)!=len(q['options']):
        raise ValueError('Price-to-option mapping failed: '+q['qid'])
    price=np.log1p(prices)
    # Absolute price has meaning across menus; menu-relative attributes are separate.
    # Category-specific native quality measures, aligned with displayed order.
    patterns=[(r'Speed:\s*([\d.]+)\s*pages',20.),(r'Screen Size:\s*([\d.]+)',50.),(r'Processor speed:\s*([\d.]+)',3.)]
    cols=[price]
    for pattern,scale in patterns:
        values=re.findall(pattern,q['text'],re.I)
        if values and len(values)!=len(prices):
            raise ValueError('Attribute-to-option mapping failed: '+q['qid'])
        cols.append(np.array(values,float)/scale if values else np.zeros(len(prices)))
    if kind!='independent':
        middle=((prices>prices.min())&(prices<prices.max())).astype(float)
        quality=sum(cols[1:])
        dominated=np.array([any(prices[j]<=px and quality[j]>=quality[i] and
                                (prices[j]<px or quality[j]>quality[i]) for j in range(len(prices))) for i,px in enumerate(prices)],float)
        features=[middle,(price-price.mean())**2,(prices==prices.min()).astype(float),dominated]
        cols.extend(features)
        if kind=='personal':
            for v in profile:
                cols.extend([f*v for f in features])
    return np.array(cols).T


def fit_menu(examples,kind):
    xs=[(menu_features(q,v,kind),y) for q,v,y in examples]
    def obj(w):
        cost=.5*w@w
        grad=w.copy()
        for x,y in xs:
            p=softmax(x@w)
            cost-=math.log(max(p[y],1e-12))
            grad+=p@x-x[y]
        return cost,grad
    fit=minimize(obj,np.zeros(xs[0][0].shape[1]),jac=True,method='L-BFGS-B')
    if not fit.success: raise ValueError(fit.message)
    return fit.x


def replay(events,actions,alpha,temp,stick,scale):
    value=np.zeros(len(actions))
    prev=None
    for a,r in events:
        logits=value/temp
        if prev is not None: logits[prev]+=stick
        yield softmax(logits)
        j=actions.index(a)
        value[j]+=alpha*(r/scale-value[j])
        prev=j


def fit_learning(traces,actions,temp,stick,scale,prior=None):
    def loss(a):
        nll=sum(-math.log(max(p[actions.index(k)],1e-12)) for t in traces for (k,_),p in zip(t,replay(t,actions,a,temp,stick,scale)))
        return nll+(2*(a-prior)**2 if prior is not None else 0)
    return min((.05,.15,.3,.55,.8,1.),key=loss)

