"""Read original screenshots in official chronological filename order, unchanged."""
import base64
import hashlib
from pathlib import Path


def screenshots(root, task, count):
    if count not in (0, 1, 2, 3):
        raise ValueError('Screenshot level must be 0, 1, 2, or 3')
    if count == 0:
        return []
    episode = (Path(root) / task['user_id'] / task['time']).resolve()
    if not episode.is_relative_to(Path(root).resolve()):
        raise ValueError('Episode path escapes screenshot root')
    if not episode.is_dir():
        raise FileNotFoundError(f'Missing episode directory: {episode}')
    folders = [p for p in episode.iterdir() if p.is_dir() and p.name.lower().startswith('screenshots')]
    if len(folders) != 1:
        raise ValueError(f'Expected exactly one screenshots folder: {episode}')
    files = sorted(p for p in folders[0].iterdir() if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if len(files) < count:
        raise ValueError(f'{task["id"]}: requires {count} screenshots; found {len(files)}')
    return files[:count]


def image_bytes(path):
    from PIL import Image
    data = path.read_bytes()
    with Image.open(path) as image:
        if image.format not in ('JPEG', 'PNG'):
            raise ValueError(f'Unsupported image format: {path}')
        mime = 'image/jpeg' if image.format == 'JPEG' else 'image/png'
        image.verify()
    return data, mime


def image_metadata(paths):
    return [{'path': str(p.resolve()), 'sha256': hashlib.sha256(image_bytes(p)[0]).hexdigest()} for p in paths]


def request_content(prompt, paths):
    if not paths:
        return prompt  # Preserve existing text-only request exactly.
    content = []
    for path in paths:
        data, mime = image_bytes(path)
        content.append({'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,' + base64.b64encode(data).decode('ascii')}})
    content.append({'type': 'text', 'text': prompt})
    return content
