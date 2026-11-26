import json,sys,urllib.request
sys.path.insert(0,'scripts');from sii import BASE,load_key
op=urllib.request.build_opener(urllib.request.ProxyHandler({}));headers={'Authorization':'Bearer '+load_key(),'Content-Type':'application/json'}
with op.open(urllib.request.Request(BASE.rsplit('/v1',1)[0]+'/openapi.json',headers=headers),timeout=15) as r:j=json.load(r)
for path in ['/v1/tokenize','/v1/detokenize']:
 schema=j['paths'][path]['post']['requestBody']['content']['application/json']['schema'];print(path,schema)
 if '$ref' in schema:
  ref=schema['$ref'].split('/')[-1];print(json.dumps(j['components']['schemas'][ref]))
