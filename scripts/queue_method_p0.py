"""Run the three P0 stages under a single model protocol."""
import argparse
import getpass
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',action='store_true');p.add_argument('--foreground',action='store_true');a=p.parse_args()
    out=a.output.resolve()
    if not a.run:
        if out.exists() and not a.foreground:raise FileExistsError(out)
        key=getpass.getpass('SII model credential: ')
        if not key:raise ValueError('Missing credential')
        env=dict(os.environ,PA_BENCH_ALT_MODEL=a.model,PA_BENCH_ALT_KEY=key)
        out.mkdir(parents=True,exist_ok=a.foreground)
        with (out/'queue.log').open('a' if a.foreground else 'w',encoding='utf8') as log,(out/'queue.err.log').open('a' if a.foreground else 'w',encoding='utf8') as err:
            if a.foreground:
                print('Queue started in attached session',flush=True)
                subprocess.run([sys.executable,'-B',__file__,'--run','--model',a.model,'--output',str(out)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=err,check=True)
                return
            child=subprocess.Popen([sys.executable,'-B',__file__,'--run','--model',a.model,'--output',str(out)],cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=err,creationflags=subprocess.CREATE_NEW_PROCESS_GROUP|subprocess.DETACHED_PROCESS)
        print('Queue started',child.pid,flush=True);return
    for phase in ('M01','M02','M03'):
        print('BEGIN',phase,datetime.now().isoformat(),flush=True)
        subprocess.run([sys.executable,'-B',str(ROOT/'scripts/run_method_p0.py'),phase,'--output',str(out/('fingertip' if phase=='M03' else 'twin'))],cwd=ROOT,check=True)
        print('END',phase,datetime.now().isoformat(),flush=True)
    print('METHOD_P0_FINISHED',flush=True)


if __name__=='__main__':main()
