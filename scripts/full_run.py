"""Bounded resumable none-baseline execution; each job has a separate log."""
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor,as_completed
import time
from run_context import ROOT,RESULTS,EFFORT

def run(job):
    name,cmd,output,expected,kind=job
    row={'name':name,'expected':expected,'status':'running','started':time.time()}
    for attempt in range(3):
        with (RESULTS/(name+'.log')).open('a',encoding='utf-8') as log:
            code=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
        try:
            count=len(json.loads(output.read_text(encoding='utf-8'))['golds']) if kind=='lamp' else len(output.read_text(encoding='utf-8').splitlines())
        except (FileNotFoundError,ValueError):count=0
        row.update(returncode=code,completed=count,attempts=attempt+1)
        if code==0 and count==expected:
            row['status']='predictions_complete'
            score_cmd=None
            if kind=='lamp':score_cmd=[sys.executable,'scripts/score_lamp.py','--task',name.split('-')[1]]
            if name=='fingertip':score_cmd=[sys.executable,'scripts/score_fingertip.py']
            if score_cmd:
                with (RESULTS/(name+'-score.log')).open('a',encoding='utf-8') as log:
                    row['score_returncode']=subprocess.run(score_cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                row['status']='score_script_finished' if row['score_returncode']==0 else 'scoring_failed'
            break
        row['status']='incomplete'
    row['ended']=time.time()
    return row

def main():
    if EFFORT!='none':raise RuntimeError('This launcher is reserved for the approved none experiment')
    jobs=[('fingertip',[sys.executable,'scripts/fingertip.py'],RESULTS/'fingertip-predictions.jsonl',1000,'jsonl')]
    for t,n in [(1,1500),(2,1557),(3,2500),(4,1800),(5,1500),(7,1500)]:
        jobs.append((f'lamp-{t}',[sys.executable,'scripts/lamp.py','--task',str(t)],RESULTS/f'lamp-{t}-test.json',n,'lamp'))
    import pandas as pd
    for f in sorted((ROOT/'data/mega/data').glob('*.parquet')):
        study=f.stem.split('-00000')[0]
        jobs.append(('mega-'+study,[sys.executable,'scripts/mega.py','--study',study],RESULTS/f'mega-{study}.jsonl',len(pd.read_parquet(f)),'jsonl'))
    rows=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(run,j) for j in jobs]
        for future in as_completed(futures):
            rows.append(future.result())
            (RESULTS/'jobs.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps(rows[-1]),flush=True)
    print('Prediction jobs finished. Scores are a separate acceptance requirement.',flush=True)

if __name__=='__main__':main()
