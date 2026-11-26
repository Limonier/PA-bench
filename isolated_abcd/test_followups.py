"""Small no-result preflight; run after B so it does not compete for a model."""
from types import SimpleNamespace
import numpy as np
import followups as f
from dataio import BASE, split
from run import Model


def main():
    args=SimpleNamespace(data=BASE.parent/'data',studies=['context_effects','default_eric'],holdout='default_eric',reader='bm25',batch_size=10,model_python=str(BASE.parent/'.retrieval-venv/Scripts/python.exe'),model=str(BASE.parent/'data/models/roberta-large'),arch='masked')
    hist,dev,held=f.prepare(args)
    policy=f.fixed_policy(hist,dev)
    acq=f.Acquisition(hist,policy)
    u,s,q,y=next(r for r in held if len(set(hist[r[0]])&set(policy.pool))>=10)
    sequence,p=f.Acquisition.route(acq,hist[u],f.proxy_predictions(dev,acq.ids,q))
    assert len(sequence)==10 and set(p)=={0,1,3,5,10}
    assert all(np.isfinite(v).all() and np.isclose(v.sum(),1) for v in p.values())
    subset={u:hist[u] for u in [*list(k for k in hist if split(k)=='dev')[:16],u]}
    embeddings=f.semantic_embeddings(args,subset)
    assert set(embeddings)==set(subset) and all(v.shape==(12,) and np.isfinite(v).all() for v in embeddings.values())
    assert f.structured(hist[u]).shape==(12,)
    model=Model(args)
    try:
        result=model.score_many([(q,dict(list(sequence.items())[:n]),args.reader) for n in (1,3,5,10)],10)
        assert len(result)==4 and all(len(v)==len(q['options']) and np.isclose(v.sum(),1) for v in result)
    finally:
        model.close()
    print('FOLLOWUPS_PREFLIGHT_OK',flush=True)


if __name__=='__main__':main()
