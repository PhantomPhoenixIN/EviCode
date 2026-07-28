"""Resumable seven-language CodeNetTrans-QS static feature extraction."""
from __future__ import annotations
import argparse, hashlib, json, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'datasets'/'Predictions_by_LLMs'
ALIASES={'C':'c','C++':'cpp','C#':'csharp','Python':'python','Ruby':'ruby','Kotlin':'kotlin','Swift':'swift'}

def teacher(path):
 n=path.name.lower(); return 'DeepSeek-Coder' if 'deepseek' in n else 'QwenCoder' if 'qwen' in n else 'StarCoder'
def candidate(r,t):
 p={'DeepSeek-Coder':'deepseekcoder','QwenCoder':'qwencoder','StarCoder':'starcoder'}[t]
 return r.get(f'{p}_translation_clean') or r.get('translated_java_code') or r.get(f'{p}_translation_raw') or ''
def tasks(languages,scores):
 for path in sorted(DATA.glob('codenet_single_solution_*_scored*.jsonl')):
  t=teacher(path)
  with path.open(encoding='utf-8') as f:
   for line_no,line in enumerate(f,1):
    r=json.loads(line); score=int(r.get('score',-1)); lang=r.get('input_language')
    if lang in languages and r.get('output_language')=='Java' and score in scores:
     yield {'example_id':f"{t.lower().replace('-','')}_{line_no}",'problem_id':str(r.get('problem_id')),'language':lang,'generator':t,'quality_score':score,'source':r.get('source_code') or '', 'candidate':candidate(r,t)}
def extract(task):
 import sys; sys.path.insert(0,str(ROOT/'src'))
 from evicode.features import static_features
 try:
  values=static_features(task['source'],task['candidate'],ALIASES[task['language']],'java')
  return {k:v for k,v in task.items() if k not in {'source','candidate'}}|{'status':'ok','source_sha256':hashlib.sha256(task['source'].encode()).hexdigest(),'candidate_sha256':hashlib.sha256(task['candidate'].encode()).hexdigest(),'features':values}
 except Exception as e:return {k:v for k,v in task.items() if k not in {'source','candidate'}}|{'status':'error','error':f'{type(e).__name__}: {e}','features':{}}
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='authentic_only_study/config.yaml');a=p.parse_args()
 cfg=yaml.safe_load((ROOT/a.config).read_text());out=ROOT/cfg['output_dir'];(out/'datasets').mkdir(parents=True,exist_ok=True);cache=out/'datasets'/'features.jsonl'
 done=set()
 if cache.exists():
  with cache.open(encoding='utf-8') as f:
   for line in f:
    try:done.add(json.loads(line)['example_id'])
    except:pass
 pending=(x for x in tasks(set(cfg['languages']),set(cfg['scores'])) if x['example_id'] not in done)
 with cache.open('a',encoding='utf-8') as target, ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
  for i,row in enumerate(pool.map(extract,pending,chunksize=8),1):
   target.write(json.dumps(row,separators=(',',':'))+'\n');target.flush()
   if i%250==0:print('extracted',i,flush=True)
 sys.path.insert(0,str(ROOT/'src'))
 from evicode.taxonomy import feature_to_category
 features=[name for name,category in feature_to_category().items() if category!='Dynamic']
 manifest={'config':cfg,'feature_names':features,'cache_sha256':hashlib.sha256(cache.read_bytes()).hexdigest()}
 (out/'extraction_manifest.json').write_text(json.dumps(manifest,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
