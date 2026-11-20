import ast,json
from threading import local
_state=local()
from pathlib import Path
from rank_bm25 import BM25Okapi
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'upstream/LaMP/LaMP/prompts'
scope={}
exec((base/'utils.py').read_text(),scope)
template_names={1:'create_classification_citation_prompt',2:'create_classification_movies_prompt',3:'create_classification_review_prompt',4:'create_generation_news_prompt',5:'create_generation_paper_prompt',7:'create_parphrase_tweet_prompt'}
names={1:'classification_citation_query_corpus_maker',2:'classification_movies_query_corpus_maker',3:'classification_review_query_corpus_maker',4:'generation_news_query_corpus_maker',5:'generation_paper_query_corpus_maker',7:'parphrase_tweet_query_corpus_maker'}
tree=ast.parse((base/'prompts.py').read_text())
for n in tree.body:
 if isinstance(n,ast.FunctionDef) and n.name in set(names.values()) | set(template_names.values()):exec(compile(ast.Module(body=[n],type_ignores=[]),'<official LaMP>','exec'),scope)
def augment(task,sample,k=4):
 if k<1: raise ValueError('k must be positive')
 if not sample['profile']: return sample['input'],[]
 corpus,query=scope[names[task]](sample['input'],sample['profile'])
 selected=BM25Okapi([x.split() for x in corpus]).get_top_n(query.split(),list(range(len(corpus))),n=min(k,len(corpus)))
 history=[sample['profile'][i] for i in selected]
 from transformers import AutoTokenizer
 if not hasattr(_state,'tokenizer'):
  _state.tokenizer=AutoTokenizer.from_pretrained(str(ROOT/'data/models/flan-t5-base-tokenizer'),local_files_only=True)
 # Invoke the exact upstream formatter. Allocate enough input space to avoid truncating any selected field.
 tokenizer=_state.tokenizer
 longest=max(len(tokenizer(str(v),add_special_tokens=False)['input_ids']) for h in history for v in h.values())
 input_budget=(longest+1024)*len(history)
 prompt=scope[template_names[task]](sample['input'],history,input_budget,tokenizer)
 return prompt,selected
