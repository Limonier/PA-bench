from huggingface_hub import snapshot_download,list_repo_files
for repo,target in [('facebook/contriever','data/models/contriever'),('FacebookAI/roberta-large','data/models/roberta-large')]:
 files=list_repo_files(repo)
 weights='model.safetensors' if 'model.safetensors' in files else 'pytorch_model.bin'
 snapshot_download(repo,local_dir=target,allow_patterns=[weights,'config.json','tokenizer*.json','special_tokens_map.json','vocab.txt','vocab.json','merges.txt'])
 print('Downloaded '+repo,flush=True)
