import json
from collections import defaultdict
import numpy as np

def summarize(path):
    def read(name):
        with (path/(name+'.jsonl')).open(encoding='utf8') as f:
            for line in f: yield json.loads(line)
    gold={r['id']:r['actual'] for r in read('labels')}
    groups={r['id']:r['groups'] for r in read('inputs')}
    stats=defaultdict(dict)
    transitions=defaultdict(lambda:defaultdict(list))
    for r in read('predictions'):
        p=np.array(r['p'])
        top=int(p.argmax())
        tags=groups[r['id']]
        slices=['all']
        if 'menu' in tags:slices.append(tags['menu'])
        if 'study' in tags:slices.append(tags['study'])
        if tags.get('previous_reward') is not None:slices.append('after_loss' if tags['previous_reward']<0 else 'after_nonloss')
        if tags.get('previous_reward') and tags.get('previous_choice') is not None:
            win=tags['previous_reward']>0
            prev=tags['previous_choice']
            wanted=lambda a:float((a==prev)==win)
            transitions[r['method'],'win_stay' if win else 'lose_shift'][r['pid']].append([wanted(top),wanted(gold[r['id']][0])])
        for j,y in enumerate(gold[r['id']]):
            score=(float(top==y),-float(np.log(max(p[y],1e-12))),float(np.sum((p-np.eye(len(p))[y])**2)))
            for group in slices:
                stats[(group,j,r['method'])][r['pid'],r['id']]=score
    report={}
    for (group,j,m),vals in stats.items():
        by=defaultdict(list)
        for (u,k),row in vals.items():by[u].append(row)
        mean=np.mean([np.mean(rows,axis=0) for rows in by.values()],axis=0)
        item=dict(zip(['agreement','nll','brier'],mean.tolist()),users=len(by),rows=len(vals))
        ref=next((r for r in ['fixed@3','fixed','shared','copy_previous'] if (group,j,r) in stats and set(vals)==set(stats[group,j,r])),None)
        if ref:
            refvals=stats[group,j,ref]
            delta=defaultdict(list)
            wins=losses=0
            for (u,k),v in vals.items():
                d=v[0]-refvals[u,k][0]
                delta[u].append(d)
                wins+=int(d>0);losses+=int(d<0)
            changes=np.array([np.mean(d)*100 for d in delta.values()])
            rng=np.random.default_rng(42)
            ci=np.percentile([rng.choice(changes,len(changes),replace=True).mean() for _ in range(2000)],[2.5,97.5])
            item.update(reference=ref,net_per_100=changes.mean(),ci95=ci.tolist(),corrected=wins,regressed=losses)
        report[f'{group}/label{j+1}/{m}']=item
    for (method,group),values in transitions.items():
        rates=np.mean([np.mean(v,axis=0) for v in values.values()],axis=0)
        report[f'transition/{group}/{method}']={'predicted':rates[0],'observed':rates[1],'users':len(values)}
    return report

