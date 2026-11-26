"""Fetch original selected episodes from official Kaggle version 2, per file."""
import argparse
import json
import struct
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

from fingertip_data import ROOT, tasks

BASE = 'https://www.kaggle.com/api/v1/datasets/download/qinglongyang/fingertip-20k/'
VERSION = 2


def image_index(prefixes):
    cache = ROOT / '.cache/fingertip-kaggle-v2-directory.bin'
    if cache.exists():
        data = cache.read_bytes()
    else:
        archive = BASE.rstrip('/') + f'?datasetVersionNumber={VERSION}'
        def byte_range(value):
            request = Request(archive, headers={'User-Agent': 'kaggle-api-client', 'Range': 'bytes=' + value})
            with urlopen(request, timeout=90) as response:
                if response.status != 206:
                    raise RuntimeError('Official archive does not support selective range reads')
                return response.read()
        tail = byte_range('-65536')
        index = tail.rfind(b'PK\x06\x06')
        if index >= 0:
            fields = struct.unpack_from('<4sQ2H2L4Q', tail, index)
            size, start = fields[-2:]
        else:
            fields = struct.unpack_from('<4s4H2LH', tail, tail.rfind(b'PK\x05\x06'))
            size, start = fields[5:7]
        data = byte_range(f'{start}-{start + size - 1}')
        if len(data) != size:
            raise ValueError('Incomplete ZIP directory')
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    result = {prefix: [] for prefix in prefixes}
    offset = 0
    while offset < len(data):
        if data[offset:offset + 4] != b'PK\x01\x02':
            raise ValueError('Invalid cached ZIP directory')
        length, extra, comment = struct.unpack_from('<3H', data, offset + 28)
        name = data[offset + 46:offset + 46 + length].decode('utf-8')
        for prefix in result:
            if name.startswith(prefix) and name.lower().endswith('.jpg'):
                result[prefix].append(name)
        offset += 46 + length + extra + comment
    return {prefix: sorted(names) for prefix, names in result.items()}


def fetch(remote, local):
    if local.exists():
        return local.read_bytes()
    url = BASE + quote(remote, safe='') + f'?datasetVersionNumber={VERSION}'
    with urlopen(Request(url, headers={'User-Agent': 'kaggle-api-client'}), timeout=60) as response:
        data = response.read()
    local.parent.mkdir(parents=True, exist_ok=True)
    temp = local.with_suffix(local.suffix + '.part')
    temp.write_bytes(data)
    temp.replace(local)
    return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=5)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/fingertip-screenshots')
    args = parser.parse_args()
    if args.limit < 1:
        parser.error('--limit must be positive')
    selected = tasks(sampled=True)[:args.limit]
    index = image_index([f'{t["user_id"]}/{t["user_id"]}/{t["time"]}/' for t in selected])
    for task in selected:
        user, stamp = task['user_id'], task['time']
        # Version 2 has an outer per-user folder in addition to episode user ID.
        remote = f'{user}/{user}/{stamp}'
        episode = args.output / user / stamp
        actions = fetch(remote + '/action.jsonl', episode / 'action.jsonl')
        survey = json.loads(fetch(remote + '/survey_result.json', episode / 'survey_result.json'))
        if str(survey.get('user_id')) != user or str(survey.get('time')) != stamp:
            raise ValueError('Downloaded episode does not match selected official CSV row')
        images = index[remote + '/']
        if len(images) < 3:
            raise ValueError(f'{task["id"]}: fewer than 3 original screenshots')
        for name in images[:3]:
            relative = Path(name[len(remote) + 1:])
            if '..' in relative.parts or len(relative.parts) != 2:
                raise ValueError('Unexpected screenshot path')
            fetch(name, episode / relative)
        print(task['id'], task['legacy_id'], 'downloaded first 3 official screenshots', flush=True)


if __name__ == '__main__':
    main()
