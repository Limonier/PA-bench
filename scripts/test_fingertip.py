"""Offline regression checks for row identity and evaluation validity."""
import collections
import json
import tempfile
import base64
from pathlib import Path
from unittest.mock import patch

from fingertip_data import tasks, align_predictions
from fingertip_images import screenshots, request_content
import timed_nex
import score_fingertip as scoring


def image_checks():
    from PIL import Image
    selected = tasks(sampled=True)
    assert len(selected) == 100 and len({r['id'] for r in selected}) == 100
    full = {t['id']: t for t in tasks()}
    assert all(full[t['id']] == t for t in selected)
    with tempfile.TemporaryDirectory(prefix='fingertip-images-') as tmp:
        root = Path(tmp)
        task = {'id': 'fixture', 'user_id': '7', 'time': '20250101_000000'}
        directory = root / '7/20250101_000000/Screenshots'
        directory.mkdir(parents=True)
        for name, color in [('003.png', 'blue'), ('001.jpg', 'red'), ('002.png', 'green')]:
            Image.new('RGB', (2, 2), color).save(directory / name)
        assert screenshots(root, task, 0) == []
        assert request_content('EXACT OFFICIAL PROMPT', []) == 'EXACT OFFICIAL PROMPT'
        for count in (1, 2, 3):
            paths = screenshots(root, task, count)
            assert [p.name for p in paths] == ['001.jpg', '002.png', '003.png'][:count]
            content = request_content('EXACT OFFICIAL PROMPT', paths)
            assert len(content) == count + 1 and content[-1] == {'type': 'text', 'text': 'EXACT OFFICIAL PROMPT'}
            for part, path in zip(content, paths):
                assert base64.b64decode(part['image_url']['url'].split(',', 1)[1]) == path.read_bytes()
        content = request_content('EXACT OFFICIAL PROMPT', screenshots(root, task, 3))
        bodies = []
        class FakeConn:
            def __init__(self, *a, **k): self.sock = None
            def connect(self): pass
            def request(self, method, url, body, headers): bodies.append(json.loads(body))
            def getresponse(self):
                self.status = 503 if len(bodies) == 1 else 200
                self.lines = iter([b'data: {"choices":[{"delta":{"content":"fixture answer"},"finish_reason":"stop"}]}\n', b'data: [DONE]\n'])
                return self
            def readline(self): return next(self.lines, b'')
            def close(self): pass
        with patch.object(timed_nex.http.client, 'HTTPSConnection', FakeConn), patch.object(timed_nex, 'load_key', return_value='test'):
            result = timed_nex.legacy_answer(content, temperature=0, max_tokens=16384)
        assert result['status'] == 'complete' and result['fallback']
        assert [body['reasoning_effort'] for body in bodies] == ['medium', 'none']
        assert all(body['model'] == 'nex-n2.5-pro' and body['max_tokens'] == 16384 for body in bodies)
        assert bodies[0]['messages'] == bodies[1]['messages'] == [{'role': 'user', 'content': content}]
        (directory / '003.png').unlink()
        try:
            screenshots(root, task, 3)
            raise AssertionError('Missing screenshot silently accepted')
        except ValueError:
            pass
    print('Native image order/bytes, official subset IDs, missing-image rejection and clean multimodal fallback passed')


def main():
    task_rows = tasks()
    assert len(task_rows) == len({t['id'] for t in task_rows}) == 1000
    source = scoring.DEFAULT_PREDICTIONS
    original = source.read_bytes()
    old = [json.loads(s) for s in original.decode('utf-8').splitlines()]
    mapped = align_predictions(old, task_rows)
    assert len(mapped) == len({r['id'] for r in mapped}) == 1000
    assert collections.Counter(r['prediction'] for r in mapped) == collections.Counter(r['prediction'] for r in old)
    assert align_predictions(mapped, task_rows) == mapped
    padded = [dict(row, prediction='\n\n' + row['prediction'] + ' ') for row in mapped]
    assert align_predictions(padded, task_rows) == mapped
    assert source.read_bytes() == original
    for invalid in [old[:-1], old + old[:1], [dict(r, prediction='') for r in old]]:
        try:
            align_predictions(invalid, task_rows)
            raise AssertionError('Invalid sample coverage accepted')
        except ValueError:
            pass
    for text, expected in [('0', 0), ('1', 1), ('maybe', None), ('', None)]:
        with patch.object(scoring, 'answer', return_value={'status': 'complete', 'answer': text}):
            result = scoring.classify(task_rows[0], mapped[0])
        assert result['label'] == expected
        assert (result['status'] == 'complete') == (expected is not None)
    print('1000 row identities preserved; legacy source unchanged; invalid coverage and judge labels rejected')
    image_checks()


if __name__ == '__main__':
    main()
