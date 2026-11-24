"""Isolated line-based scorer worker. The parent never sees torch caches."""
import json
import math
import re
import sys
from collections import Counter

def retrieve(history, query, method, k=8):
    if method=='empty':
        return []
    def tokens(s):
        return re.findall(r'\w+',s.lower())
    docs=[(key,value,Counter(tokens(key+' '+value))) for key,value in history.items()]
    if not docs:
        return []
    length=sum(d.total() for _,_,d in docs)/len(docs) or 1
    terms=set(tokens(query))
    freq={t:sum(t in d for _,_,d in docs) for t in terms}
    def score(row):
        d=row[2]
        return sum(math.log1p((len(docs)-freq[t]+.5)/(freq[t]+.5))*d[t]*2.2/(d[t]+1.2*(.25+.75*d.total()/length)) for t in terms)
    return [(k,v) for k,v,_ in sorted(docs,key=score,reverse=True)[:k]]


def serve(path, causal):
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM,AutoModelForMaskedLM
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    model=(AutoModelForCausalLM if causal else AutoModelForMaskedLM).from_pretrained(path,local_files_only=True).eval()
    device='cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    limit=min(2048,getattr(model.config,'max_position_embeddings',512)-2,tok.model_max_length)
    def encode(text):
        return tok(text,return_tensors='pt').to(device)
    for line in sys.stdin:
        req=json.loads(line)
        q=req['q']
        hist='\n'.join(f"{k}: {v}" for k,v in req['evidence'])
        if req['mode']=='gpg':
            if not causal:
                raise ValueError("GPG requires a generative causal checkpoint")
            prompt='Identify decision habits and exceptions grounded in these prior choices. Write a concise preference profile.\n'+hist+'\nProfile:'
            x=encode(prompt)
            ids=x.input_ids[:,-(limit-96):]
            with torch.inference_mode():
                out=model.generate(ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,pad_token_id=tok.eos_token_id)
            hist=tok.decode(out[0,ids.shape[1]:],skip_special_tokens=True)+'\n'+hist
        letters=[chr(65+i) for i in range(len(q['options']))]
        codes=[tok.encode(' '+c,add_special_tokens=False) for c in letters]
        if any(len(c)!=1 for c in codes) or len({c[0] for c in codes})!=len(codes):
            raise ValueError('Ambiguous choice code')
        question=q['text']+'\n'+'\n'.join(f'{a}. {b}' for a,b in zip(letters,q['options']))+'\nPredict this person\'s choice. Answer:'+('' if causal else ' '+tok.mask_token)
        # Protect the entire question and every option when packing evidence.
        if len(tok.encode(question))>limit:
            raise ValueError('Question exceeds model context')
        prefix=tok.encode(hist,add_special_tokens=False,truncation=True,max_length=limit)[:max(0,limit-len(tok.encode(question))-3)]
        x=encode(tok.decode(prefix)+'\n'+question)
        if x.input_ids.size(1)>limit:
            x={k:v[:,-limit:] for k,v in x.items()}
        pos=-1 if causal else int((x['input_ids'][0]==tok.mask_token_id).nonzero()[-1].item())
        with torch.inference_mode():
            logits=model(**x).logits[0,pos,[c[0] for c in codes]].float()
        print(json.dumps(logits.softmax(0).tolist()),flush=True)

if __name__=='__main__':
    serve(sys.argv[1],sys.argv[2]=='causal')

