"""Explicit, one-user CLI; the benchmark owns prediction and scoring."""
import argparse
import json
from pathlib import Path
from .methods import METHODS, GUIDES, History, AlignGenerator, create


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--history',type=Path,required=True,help='JSON array of visible History records')
    p.add_argument('--query',default='')
    p.add_argument('--task',choices=sorted(GUIDES),default='vita')
    p.add_argument('--storage',type=Path,help='Unique persistent directory for O-Mem / Mem0')
    p.add_argument('--align-checkpoint',type=Path)
    p.add_argument('--quantize',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    opts={}
    if args.method=='gpg':opts['task']=args.task
    if args.method=='alignxplore_plus':
        if not args.align_checkpoint:p.error('AlignXplore+ requires its own checkpoint')
        opts['generator']=AlignGenerator(args.align_checkpoint,args.quantize)
    if args.method in ('omem','mem0'):
        if not args.storage:p.error('Use a unique --storage directory for this user/trial')
        opts['storage']=args.storage
    rows=json.loads(args.history.read_text(encoding='utf-8'))
    memory=create(args.method,**opts)
    try:
        memory.update([History(**r) for r in rows])
        context=memory.retrieve(args.query)
        args.output.write_text(json.dumps({'method':args.method,'context':context},ensure_ascii=False,indent=2),encoding='utf-8')
    finally:
        if hasattr(memory,'close'):memory.close()


if __name__=='__main__':main()
