"""Freeze B's 72 test tracks and the wave-split source history for M01/M02."""
import json
from collections import defaultdict
from dataio import BASE, digest, twin
from run import filter_target_history


def main():
    manifest=json.loads((BASE/'b_min_validation_v1.json').read_text(encoding='utf8'))
    ids=set(manifest['samples']['test']); users={k.split('/')[0] for k in ids}
    hist={}; targets=defaultdict(list); every=[]
    for u,h,q,a,b in twin(BASE.parent/'data',True,users):
        hist[u]=h; every.append((u,'twin',q,a))
        key=f'{u}/{q["qid"]}/'+digest(q['text'])[:12]
        if key in ids:targets[u].append(dict(id=key,question=q,early=a,late=b))
    hist=filter_target_history(hist,every)
    assert {r['id'] for rs in targets.values() for r in rs}==ids and len(users)==12
    payload={'source':'wave_split; overlap filter across all retest prompts of selected users','users':{u:dict(history=hist[u],targets=targets[u]) for u in sorted(users,key=digest)}}
    path=BASE/'twin_method_test72_v1.json'
    with path.open('x',encoding='utf8') as f:json.dump(payload,f,ensure_ascii=False)
    print('Frozen',len(users),'users,',sum(len(r) for r in targets.values()),'tracks')


if __name__=='__main__':main()
