"""Loopback-only OpenAI embedding service for the labelled local-RAG variant."""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    from sentence_transformers import SentenceTransformer
    name = 'paraphrase-multilingual-MiniLM-L12-v2'
    model = SentenceTransformer(str(Path(__file__).resolve().parents[1] / 'data/models' / name), device='cpu', local_files_only=True)
    lock = threading.Lock()
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            if self.path != '/v1/embeddings':
                self.send_error(404); return
            try:
                data = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if data.get('model') != name:
                    raise ValueError('Incorrect local embedding model name')
                texts = data['input']
                if isinstance(texts, str): texts = [texts]
                if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
                    raise ValueError('Only text embedding input is supported')
                with lock:
                    vectors = model.encode(texts, normalize_embeddings=True).tolist()
                result = {'object': 'list', 'model': name,
                          'data': [{'object': 'embedding', 'index': i, 'embedding': v} for i, v in enumerate(vectors)],
                          'usage': {'prompt_tokens': 0, 'total_tokens': 0}}
                encoded = json.dumps(result).encode()
                self.send_response(200); self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(encoded))); self.end_headers(); self.wfile.write(encoded)
            except Exception as exc:
                self.send_error(400, type(exc).__name__)
    print('Local embedding replacement:', name, 'port', args.port, flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__': main()
