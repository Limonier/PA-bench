"""Download official pinned data; write a SHA-256 manifest. No model calls."""
import argparse
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
REPOS = {
    'vita': ('meituan-longcat/VitaBench-2.0', 'a4553e13ec081be65cc37b48845da5819692d438'),
    'mega': ('LLM-Digital-Twin/Twin-2K-500-Mega-Study', '0401b715a341ac4b5f98b4424b4aecf9d29570d0'),
    'twin': ('LLM-Digital-Twin/Twin-2K-500', 'f883165a3026fde855dfd448e0cd16443ab257b6'),
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument('bench', choices=['vita', 'mega', 'twin', 'lamp'])
    p.add_argument('--proxy', help='Optional download proxy, e.g. http://127.0.0.1:7890')
    p.add_argument('--splits',nargs='+',default=['train','dev','test'],choices=['train','dev','test'])
    a = p.parse_args()
    s = requests.Session()
    s.trust_env = False
    if a.proxy:
        s.proxies = {'http': a.proxy, 'https': a.proxy}
    if a.bench == 'lamp':
        files = [(f'LaMP_{t}/{split}/{split}_{kind}.json',
                  f'https://ciir.cs.umass.edu/downloads/LaMP/time/LaMP_{t}/' + ('new/' if t=='2' else '') + f'{split}/{split}_{kind}.json')
                 for t in ['1','2','3','4','5','7'] for split in a.splits for kind in ['questions','outputs']]
    else:
        repo, rev = REPOS[a.bench]
        r = s.get(f'https://huggingface.co/api/datasets/{repo}/tree/{rev}?recursive=true', timeout=60)
        r.raise_for_status()
        entries = r.json()
        if r.links.get('next'):
            raise RuntimeError('Paginated tree: implement pagination before downloading this dataset')
        files = [(x['path'], f'https://huggingface.co/datasets/{repo}/resolve/{rev}/{x["path"]}')
                 for x in entries if x['type']=='file' and (x['path'].endswith('.parquet') or x['path']=='tasks.json')]
    manifest = []
    for name, url in files:
        dst = ROOT/'data'/a.bench/name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            print('Downloading', name, flush=True)
            tmp = dst.with_suffix(dst.suffix+'.part')
            for attempt in range(5):
                offset = tmp.stat().st_size if tmp.exists() else 0
                try:
                    with s.get(url, headers={'Range': f'bytes={offset}-'} if offset else {}, stream=True, timeout=120) as r:
                        r.raise_for_status()
                        with tmp.open('ab' if offset and r.status_code==206 else 'wb') as f:
                            for chunk in r.iter_content(1024*1024): f.write(chunk)
                    tmp.replace(dst)
                    break
                except requests.RequestException:
                    if attempt==4: raise
        manifest.append({'path': str(dst.relative_to(ROOT)), 'url':url,
                         'bytes':dst.stat().st_size, 'sha256':hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()})
    (ROOT/'results'/f'{a.bench}-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

if __name__=='__main__': main()
