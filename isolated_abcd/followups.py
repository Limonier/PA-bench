"""Target-aware information acquisition and menu-personalization follow-ups."""
from collections import Counter, defaultdict
import json
import math
import random
import re
import subprocess
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from dataio import BASE, digest, mega, split, twin
from mechanisms import QuestionPolicy, fit_menu, menu_features
from run import filter_target_history, profile, dump, resume_a_predictions
from scorer import retrieve

BUDGETS = (1, 3, 5, 10)


def prepare(args, menu_only=False):
    cases = list(mega(args.data, ['context_effects'] if menu_only else args.studies))
    hist = filter_target_history(dict(twin(args.data, users={u for u, *_ in cases})), cases)
    dev = [(u,s,q,y) for u,s,q,y in cases if u in hist and split(u) == 'dev' and (menu_only or s != args.holdout)]
    held = [(u,s,q,y) for u,s,q,y in cases if u in hist and split(u) in ('cal','test')]
    return hist, dev, held


def fixed_policy(hist, dev):
    train = defaultdict(list)
    for u,s,q,y in dev:
        train[u].append((s+'|'+q['qid'], q['options'][y]))
    return QuestionPolicy(hist, train, 10)


def entropy(p):
    p = np.maximum(p, 1e-12)
    return -np.sum(p*np.log(p), axis=-1)


class Acquisition:
    """Posterior on development users; selection never observes unasked answers."""
    def __init__(self, hist, policy):
        self.ids = policy.ids
        self.pool = policy.pool
        self.parts = {}
        for key in self.pool:
            groups = defaultdict(list)
            for i,u in enumerate(self.ids):
                groups[hist[u].get(key)].append(i)
            self.parts[key] = [np.asarray(v) for v in groups.values()]
        self.answers = {k: np.asarray([hist[u].get(k) for u in self.ids], dtype=object) for k in self.pool}

    def route(self, history, predictions, candidates=None):
        keys = sorted(set(history) & set(self.pool))
        if candidates is not None:
            keys = [k for k in candidates if k in keys]
        if len(keys) < 10:
            raise ValueError('Insufficient available questions')
        posterior = np.full(len(self.ids), 1/len(self.ids))
        result = {}
        prob = {}
        prob[0] = posterior @ predictions
        for step in range(1,11):
            current = entropy(posterior @ predictions)
            def gain(k):
                # Expected posterior entropy; a soft match avoids collapsing to one person.
                total = 0.
                for group in self.parts[k]:
                    mass = posterior[group].sum()
                    if mass:
                        total += mass * entropy((posterior[group] @ predictions[group]) / mass)
                return current - total
            key = max((k for k in keys if k not in result), key=gain)
            answer = history[key]  # the selected answer is revealed only now
            result[key] = answer
            posterior *= np.where(self.answers[key] == answer, 1., .30)
            posterior /= posterior.sum()
            if step in BUDGETS:
                prob[step] = posterior @ predictions
        return result, prob

    def posterior_path(self, history, sequence, predictions):
        posterior = np.full(len(self.ids), 1/len(self.ids))
        result = {0:posterior @ predictions}
        for step,key in enumerate(sequence,1):
            posterior *= np.where(self.answers[key] == history[key], 1., .30)
            posterior /= posterior.sum()
            if step in BUDGETS:
                result[step] = posterior @ predictions
        return result


def question_relevance(q, keys):
    # Rank question text only, without peeking at unasked answers.
    docs = {k: '' for k in keys}
    return [k for k,_ in retrieve(docs,q['text']+' '+' '.join(q['options']),'bm25',k=len(docs))]


def proxy_predictions(dev, ids, q):
    # Only development-study labels; choose the most text-related source task.
    templates = {}
    for u,s,t,y in dev:
        templates.setdefault(s+'|'+t['qid']+'|'+digest(t['text']+'|'+str(t['options'])), t)
    ranks = question_relevance(q, sorted(k+'|'+v['text'] for k,v in templates.items()))
    selected = next(k for k,v in templates.items() if k+'|'+v['text']==ranks[0])
    t = templates[selected]
    observed = {u: y for u,s,x,y in dev if s+'|'+x['qid']+'|'+digest(x['text']+'|'+str(x['options']))==selected}
    n = len(t['options'])
    prior = np.bincount(list(observed.values()),minlength=n)+1.
    prior /= prior.sum()
    return np.array([np.eye(n)[observed[u]]*.9+prior*.1 if u in observed else prior for u in ids])


def run_a2(args, model, out):
    hist,dev,held = prepare(args)
    policy = fixed_policy(hist,dev)
    acq = Acquisition(hist,policy)
    out.cohort(policy.ids,'policy_fit')
    cache = {}
    saved, saved_p = resume_a_predictions(args.resume_from)
    saved_gold = {} if not args.resume_from else {r['id']:r['actual'] for r in map(json.loads,(args.resume_from/'labels.jsonl').read_text(encoding='utf8').splitlines())}
    methods = {'empty'} | {f'{m}@{n}' for m in ('fixed','random','relevant','uncertainty') for n in BUDGETS}
    reused = 0
    for u,s,q,y in held:
        h = hist[u]
        if len(set(h)&set(policy.pool))<10: continue
        track = f'{s}/{u}/{q["qid"]}'
        groups = {'study':s,'heldout_study':s==args.holdout}
        if track in saved_gold:
            row = saved[track]; p = saved_p[track]
            assert row['pid']==u and row['question']==q and row['groups']==groups and saved_gold[track]==[y]
            assert set(row['history'])=={'fixed','random','relevant','uncertainty'} and set(p)==methods
            assert all(len(seq)==10 and len(dict(seq))==10 and all(k in policy.pool and h.get(k)==v for k,v in seq) for seq in row['history'].values())
            assert all(len(v)==len(q['options']) and np.isfinite(v).all() and np.min(v)>=0 and np.isclose(sum(v),1) for v in p.values())
            out.record(u,track,q,row['history'],p,[y],groups)
            reused+=1
            continue
        keys = sorted(set(h)&set(policy.pool))
        fixed = policy.route(h,'fixed',10,random.Random(digest(u)))
        random_seq = policy.route(h,'random',10,random.Random(digest(u)))
        relevant = dict((k,h[k]) for k in question_relevance(q,keys)[:10])
        key = (s,q['qid'],q['text'])
        if key not in cache: cache[key] = proxy_predictions(dev,acq.ids,q)
        active,_ = acq.route(h,cache[key])
        seqs = dict(fixed=fixed,random=random_seq,relevant=relevant,uncertainty=active)
        requests = {'empty':(q,{},args.reader)}
        for name,seq in seqs.items():
            for n in BUDGETS: requests[f'{name}@{n}'] = (q,dict(list(seq.items())[:n]),args.reader)
        p = dict(zip(requests,model.score_many(list(requests.values()),args.batch_size)))
        out.record(u,track,q,{k:list(v.items()) for k,v in seqs.items()},p,[y],groups)
    assert reused==len(saved_gold), 'Resume contains tracks outside the current cohort'
    dump(out.files['audit'],dict(resumed_tracks=reused,batch_size=args.batch_size))


def structured(h):
    # Domain-specific stated attitude and evidence coverage, not inferential labels.
    domains = ('price|cost|money|spend|income|afford','risk|uncertain|gambl|chance|safe','quality|durab|performance|best','health|exercise|diet|medical','social|friend|family|group','environment|climate|energy|recycl')
    positive = re.compile(r'\b(agree|yes|important|often|always|likely|prefer|support)\b', re.I)
    negative = re.compile(r'\b(disagree|no|never|rarely|unlikely|oppose|not)\b', re.I)
    vec = []
    for d in domains:
        values = [a for k,a in h.items() if re.search(d,k,re.I)]
        vec.extend([sum(bool(positive.search(a))-bool(negative.search(a)) for a in values)/max(1,len(values)), math.log1p(len(values))/5])
    return np.array(vec)


def semantic_embeddings(args,hist):
    # Local pretrained sentence embeddings, not an LLM-generated profile.
    payload = {u: [' '.join(f'{k}: {v}' for k,v in rows[i:i+4]) for i in range(0,len(rows),4)]
               for u,h in hist.items() for rows in [sorted(h.items(),key=lambda kv:digest(kv[0]))[:32]]}
    model_path = BASE.parent/'data/models/paraphrase-multilingual-MiniLM-L12-v2'
    p = subprocess.run([args.model_python,'-B',str(BASE/'embed_followups.py'),str(model_path)],input=json.dumps(payload),capture_output=True,text=True,encoding='utf8',check=True)
    raw = {u:np.array(v) for u,v in json.loads(p.stdout).items()}
    pca = PCA(n_components=12,random_state=42).fit([v for u,v in raw.items() if split(u)=='dev'])
    return {u:v/max(np.linalg.norm(v),1.) for u,v in zip(raw,pca.transform(list(raw.values())))}


def run_c2(args,model,out):
    hist,dev,held = prepare(args,True)
    sem = semantic_embeddings(args,hist)
    reps = {'hash':{u:profile(h) for u,h in hist.items()},'semantic':sem,'structured':{u:structured(h) for u,h in hist.items()}}
    reps['hybrid'] = {u:np.concatenate([sem[u],reps['structured'][u]]) for u in hist}
    out.cohort({u for u,*_ in dev},'menu_fit')
    shared = fit_menu([(q,profile(hist[u]),y) for u,_,q,y in dev],'shared')
    weights = {name:fit_menu([(q,vs[u],y) for u,_,q,y in dev],'personal') for name,vs in reps.items()}
    dump(out.files['audit'],{'shared':shared.tolist(),'personal':{k:w.tolist() for k,w in weights.items()},'semantic':'MiniLM 32 deterministic history items, mean embeddings, development-only PCA; not generative LLM'})
    people = sorted({u for u,*_ in held},key=digest)
    swaps = dict(zip(people,people[1:]+people[:1]))
    for u,s,q,y in held:
        p = {'shared':softmax(menu_features(q,profile(hist[u]),'shared')@shared)}
        p.update({k:softmax(menu_features(q,v[u],'personal')@weights[k]) for k,v in reps.items()})
        p['semantic_shuffled'] = softmax(menu_features(q,sem[swaps[u]],'personal')@weights['semantic'])
        p['raw'] = model.score(q,hist[u],args.reader)
        p['hybrid_raw'] = (p['hybrid']+p['raw'])/2
        out.record(u,f'{u}/{q["qid"]}',q,{k:v[u].tolist() for k,v in reps.items()},p,[y],{'menu':q['name'],'shuffled':swaps[u]})


def run_ac1(args,model,out):
    hist,dev,held = prepare(args,True)
    policy = fixed_policy(hist,dev)
    acq = Acquisition(hist,policy)
    out.cohort(policy.ids,'policy_and_menu_fit')
    profiles = {u:structured(h) for u,h in hist.items()}
    weights = fit_menu([(q,profiles[u],y) for u,_,q,y in dev],'personal')
    for u,s,q,y in held:
        h = hist[u]
        keys = sorted(set(h)&set(policy.pool))
        if len(keys)<10:continue
        # Distribution over plausible user's preference models, fitted on dev only.
        possibility = np.array([softmax(menu_features(q,profiles[p],'personal')@weights) for p in acq.ids])
        active,active_p = acq.route(h,possibility)
        fixed = policy.route(h,'fixed',10,random.Random(digest(u)))
        rand = policy.route(h,'random',10,random.Random(digest(u)))
        rel = question_relevance(q,keys)[:10]
        seqs = dict(active=list(active),fixed=list(fixed),random=list(rand),relevant=rel)
        p = {'empty':active_p[0]}
        for name,seq in seqs.items():
            states = active_p if name=='active' else acq.posterior_path(h,seq,possibility)
            p.update({f'{name}@{n}':states[n] for n in BUDGETS})
        out.record(u,f'{u}/{q["qid"]}',q,{k:[(i,h[i]) for i in v] for k,v in seqs.items()},p,[y],{'menu':q['name']})


def followup_contrasts(path, experiment):
    """Precommitted user-paired contrasts, nominal and Bonferroni intervals."""
    from pathlib import Path
    path = Path(path)
    gold = {r['id']:r['actual'][0] for r in map(json.loads,(path/'labels.jsonl').read_text(encoding='utf8').splitlines())}
    predictions = defaultdict(dict)
    for r in map(json.loads,(path/'predictions.jsonl').read_text(encoding='utf8').splitlines()):
        y=gold[r['id']]; p=np.array(r['p'])
        predictions[r['pid'],r['id']][r['method']]=(int(p.argmax()==y),-math.log(max(p[y],1e-12)))
    if experiment in ('A2','AC1'):
        active='uncertainty' if experiment=='A2' else 'active'
        pairs=[(f'{m}@{n}',f'fixed@{n}') for n in BUDGETS for m in (active,'relevant','random')]
    elif experiment=='C3':
        pairs=[(m,'shared') for m in ('structured_c2','categorical','structured_full','joint','conditional')]+[('conditional','conditional_shuffled'),('conditional','joint'),('categorical','structured_c2')]
    elif experiment=='AC2':
        pairs=[(f'active@{n}',f'fixed@{n}') for n in BUDGETS]+[(f'active@{n}',f'posterior_active@{n}') for n in BUDGETS]+[(f'active@{n}','empty') for n in BUDGETS]+[('full_history','empty')]
    elif experiment=='C4':
        pairs=[('decision_domains','shared'),('decision_domains','structured_c2'),('decision_domains','shuffled')]
    elif experiment=='C5':
        pairs=[('parameter','shared'),('parameter','parameter_shuffled')]
    elif experiment=='AC3':
        pairs=[(f'active@{n}',f'fixed@{n}') for n in BUDGETS]+[(f'active@{n}','empty') for n in BUDGETS]+[('full_history','empty')]
    else:
        pairs=[(m,'shared') for m in ('hash','semantic','structured','hybrid','raw','hybrid_raw')]+[('semantic','semantic_shuffled')]
    report={}
    for a,b in pairs:
        by=defaultdict(list)
        for (u,_),v in predictions.items(): by[u].append(np.array(v[a])-v[b])
        d=np.array([np.mean(v,axis=0) for v in by.values()]); d[:,0]*=100
        rng=np.random.default_rng(42); boot=d[rng.integers(len(d),size=(10000,len(d)))].mean(axis=1)
        alpha=2.5/len(pairs)
        report[f'{a} vs {b}']={'users':len(d),'agreement_pp':float(d[:,0].mean()),'nll_delta':float(d[:,1].mean()),'ci95_pp':np.percentile(boot[:,0],[2.5,97.5]).tolist(),'familywise_ci_pp':np.percentile(boot[:,0],[alpha,100-alpha]).tolist(),'nll_ci95':np.percentile(boot[:,1],[2.5,97.5]).tolist(),'nll_familywise_ci':np.percentile(boot[:,1],[alpha,100-alpha]).tolist()}
    return report
