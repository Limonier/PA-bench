"""Validate C3/AC2 against actual development data without saving predictions."""
from types import SimpleNamespace
from io import StringIO
import numpy as np
from dataio import BASE
import profile_updates as p


def main():
    args=SimpleNamespace(data=BASE.parent/'data')
    hist,dev,held=p.prepare(args,True)
    assert len(held)==579 and len({u for u,*_ in held})==193
    assert not {u for u,*_ in dev}&{u for u,*_ in held}
    policy=p.fixed_policy(hist,dev)
    enc=p.EvidenceEncoder(hist,policy.ids,p.decision_keys(policy))
    assert all(len(set(hist[u])&set(enc.contrib))>=10 for u,*_ in held)
    class Out:
        def __init__(self):self.files={'audit':StringIO()};self.n=0
        def cohort(self,*a):pass
        def record(self,u,key,q,h,pred,y,g):
            assert all(len(v)==len(q['options']) and np.isfinite(v).all() and np.isclose(v.sum(),1) for v in pred.values())
            self.n+=1
    p.prepare=lambda *a:(hist,dev,held[:2])
    for run in (p.run_c3,p.run_ac2):
        out=Out();run(args,None,out);assert out.n==2
    print('PROFILE_UPDATES_PREFLIGHT_OK')


if __name__=='__main__':main()
