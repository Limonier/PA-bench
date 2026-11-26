"""Offline mean-pooled MiniLM user-history embeddings."""
import json
import sys
import torch
from transformers import AutoModel, AutoTokenizer

path = sys.argv[1]
data = json.load(sys.stdin)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, local_files_only=True).eval().to(device)
items = [(pid, text) for pid, texts in data.items() for text in (texts or [''])]
by = {}
with torch.inference_mode():
    for i in range(0,len(items),16):
        batch = items[i:i+16]
        x = tok([t for _,t in batch], padding=True, truncation=True, max_length=128, return_tensors='pt').to(device)
        hidden = model(**x).last_hidden_state
        mask = x.attention_mask.unsqueeze(-1)
        vec = (hidden*mask).sum(1)/mask.sum(1).clamp(min=1)
        for (pid,_),v in zip(batch,vec.cpu()): by.setdefault(pid,[]).append(v)
print(json.dumps({pid:torch.stack(v).mean(0).tolist() for pid,v in by.items()}))
