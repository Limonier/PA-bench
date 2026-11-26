"""Probe SII with the existing cached token; never persist credentials."""
import argparse
import json
from pathlib import Path
from sii import request, chat

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model')
    a=p.parse_args()
    try:
        result=chat(a.model,'Reply exactly OK.',max_tokens=64) if a.model else request('/models',timeout=30)
    except Exception as exc:
        print(json.dumps({'status':type(exc).__name__,'http_status':getattr(exc,'code',None)}))
        raise SystemExit(1)
    dst=Path(__file__).resolve().parents[1]/'results'/('sii-completion.json' if a.model else 'sii-models.json')
    dst.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=True))

if __name__=='__main__':main()
