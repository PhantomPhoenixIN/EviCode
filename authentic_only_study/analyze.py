"""Analyze authentic-only CodeNetTrans-QS verification across seven languages."""
from __future__ import annotations
import argparse, hashlib, json, platform, subprocess, sys
from pathlib import Path
import joblib, matplotlib.pyplot as plt, numpy as np, pandas as pd, seaborn as sns, yaml
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,confusion_matrix,f1_score,precision_score,recall_score,roc_auc_score,roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from evicode.taxonomy import OBSERVATION_FAMILIES, feature_to_category  # noqa: E402
FAMILIES = OBSERVATION_FAMILIES
FORBIDDEN={'quality_score','label','generator','language','problem_id','execution','compiler','reference','tests'}
def args():
 p=argparse.ArgumentParser();p.add_argument('--config',default='authentic_only_study/config.yaml');return p.parse_args()
def load(path,features):
 rows=[]
 with path.open(encoding='utf-8') as f:
  for line in f:
   r=json.loads(line)
   if r.get('status')!='ok' or int(r['quality_score']) not in {1,2,3}:continue
   rows.append({'example_id':r['example_id'],'problem_id':r['problem_id'],'language':r['language'],'generator':r['generator'],'quality_score':int(r['quality_score']),'label':int(r['quality_score']==3),**{x:float(r['features'][x]) for x in features}})
 return pd.DataFrame(rows).drop_duplicates('example_id')
def split(df,seed,size):
 g=GroupShuffleSplit(n_splits=1,test_size=size,random_state=seed);a,b=next(g.split(df,df.label,df.problem_id));tr,te=df.iloc[a].copy(),df.iloc[b].copy();assert set(tr.problem_id).isdisjoint(te.problem_id);return tr,te
def fit(df,features,seed):
 m=Pipeline([('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,random_state=seed,class_weight=None))]);m.fit(df[features],df.label);return m
def ece(y,p,n=10):
 edges=np.linspace(0,1,n+1);return sum(((p>=a)&(p<(b if b<1 else b+1e-9))).mean()*abs(p[(p>=a)&(p<(b if b<1 else b+1e-9))].mean()-y[(p>=a)&(p<(b if b<1 else b+1e-9))].mean()) for a,b in zip(edges[:-1],edges[1:]) if ((p>=a)&(p<(b if b<1 else b+1e-9))).any())
def metric(y,p):
 q=(p>=.5).astype(int);return {'roc_auc':roc_auc_score(y,p),'pr_auc':average_precision_score(y,p),'precision':precision_score(y,q,zero_division=0),'recall':recall_score(y,q,zero_division=0),'f1':f1_score(y,q,zero_division=0),'brier':brier_score_loss(y,p),'ece':ece(np.asarray(y),np.asarray(p))}
def evaluate(model,df,features,scope,value='All'):
 p=model.predict_proba(df[features])[:,1];return {'scope':scope,'value':value,'n':len(df),'positives':int(df.label.sum()),**metric(df.label.to_numpy(),p)},p
def bootstrap(df,p,iters,seed):
 rng=np.random.default_rng(seed);groups=df.problem_id.unique();idx={g:np.flatnonzero(df.problem_id.to_numpy()==g) for g in groups};vals=[]
 for _ in range(iters):
  ids=np.concatenate([idx[g] for g in rng.choice(groups,len(groups),replace=True)]);x=df.iloc[ids]
  if x.label.nunique()>1:vals.append(metric(x.label.to_numpy(),p[ids]))
 return {f'{k}_ci_low':np.quantile([x[k] for x in vals],.025) for k in vals[0]}|{f'{k}_ci_high':np.quantile([x[k] for x in vals],.975) for k in vals[0]}
def main():
 cfg=yaml.safe_load((ROOT/args().config).read_text());out=ROOT/cfg['output_dir'];
 for d in ['metrics','models','statistics','feature_importance','calibration','failures','figures','tables','paper_materials']: (out/d).mkdir(parents=True,exist_ok=True)
 features=[name for name,category in feature_to_category().items() if category!='Dynamic'];assert len(features)==45 and set(features).isdisjoint(FORBIDDEN)
 assigned=[name for names in FAMILIES.values() for name in names]
 assert len(assigned)==len(set(assigned))==len(features) and set(assigned)==set(features)
 df=load(out/'datasets'/'features.jsonl',features);df.to_parquet(out/'datasets'/'analysis_dataset.parquet',index=False)
 train,test=split(df,cfg['seed'],cfg['test_size']);model=fit(train,features,cfg['seed']);joblib.dump({'pipeline':model,'features':features},out/'models'/'authentic_all_languages.joblib')
 rows=[];r,p=evaluate(model,test,features,'Overall');r.update(bootstrap(test,p,cfg['bootstrap_iterations'],cfg['seed']));rows.append(r)
 hard=test[test.quality_score.isin([2,3])].copy();rh,ph=evaluate(model,hard,features,'Hard Score 2 vs 3');rh.update(bootstrap(hard,ph,cfg['bootstrap_iterations'],cfg['seed']));rows.append(rh)
 for col,scope in [('language','Language'),('generator','Generator'),('quality_score','Quality')]:
  for value,g in test.groupby(col):
   if g.label.nunique()==2:r,_=evaluate(model,g,features,scope,str(value));rows.append(r)
 # leave-one-domain-out uses authentic rows only; metadata defines partitions but never enters X.
 for col,scope in [('language','Leave-one-language-out'),('generator','Leave-one-generator-out')]:
  for value in sorted(df[col].unique()):
   tr,te=df[df[col]!=value],df[df[col]==value]
   m=fit(tr,features,cfg['seed']);r,_=evaluate(m,te,features,scope,str(value));rows.append(r)
 metrics=pd.DataFrame(rows);metrics.to_csv(out/'metrics'/'all_metrics.csv',index=False)
 # family ablation on the fixed grouped split
 ab=[]
 base=metric(test.label.to_numpy(),p)
 for family,names in FAMILIES.items():
  keep=[x for x in features if x not in names];m=fit(train,keep,cfg['seed']);q=m.predict_proba(test[keep])[:,1];z=metric(test.label.to_numpy(),q);ab.append({'removed':family,**z,'delta_auc':z['roc_auc']-base['roc_auc'],'delta_f1':z['f1']-base['f1']})
 pd.DataFrame(ab).to_csv(out/'statistics'/'family_ablation.csv',index=False)
 # importance and progression
 scaled=model.named_steps['scale'].transform(train[features]);coef=model.named_steps['model'].coef_[0];mi=mutual_info_classif(train[features],train.label,random_state=cfg['seed'])
 imp=pd.DataFrame({'feature':features,'coefficient':coef,'mean_abs_linear_shap':np.mean(abs(scaled*coef),axis=0),'mutual_information':mi});imp['family']=imp.feature.map(lambda x:next((f for f,n in FAMILIES.items() if x in n),'Other'));imp.to_csv(out/'feature_importance'/'importance.csv',index=False)
 prog=[]
 for (lang,gen,score),g in df.groupby(['language','generator','quality_score']):
  for family,names in FAMILIES.items():prog.append({'language':lang,'generator':gen,'quality_score':score,'family':family,'mean':g[[x for x in names if x in features]].mean(axis=1).mean(),'n':len(g)})
 pd.DataFrame(prog).to_csv(out/'feature_importance'/'quality_progression.csv',index=False)
 # predictions, calibration and failures
 pred=test[['example_id','problem_id','language','generator','quality_score','label']].copy();pred['confidence']=p;pred['predicted']=(p>=.5).astype(int);pred.to_csv(out/'metrics'/'predictions.csv',index=False)
 bins=[]
 for scope,col in [('Overall',None),('Language','language'),('Generator','generator')]:
  groups=[('All',test)] if col is None else test.groupby(col)
  for value,g in groups:
   q=model.predict_proba(g[features])[:,1];bins.append({'scope':scope,'value':value,'n':len(g),'brier':brier_score_loss(g.label,q),'ece':ece(g.label.to_numpy(),q),'mean_confidence':q.mean(),'prevalence':g.label.mean()})
 pd.DataFrame(bins).to_csv(out/'calibration'/'calibration_summary.csv',index=False)
 fp=pred[(pred.label==0)&(pred.confidence>=.8)].sort_values('confidence',ascending=False).head(25);fn=pred[(pred.label==1)&(pred.confidence<=.2)].sort_values('confidence').head(25);pd.concat([fp.assign(case='high-confidence incorrect'),fn.assign(case='low-confidence correct')]).to_csv(out/'failures'/'representative_failures.csv',index=False)
 # publication figures
 sns.set_theme(style='whitegrid');figdir=out/'figures'
 for scope,name in [('Language','language_wise_auc'),('Generator','generator_wise_auc'),('Leave-one-language-out','logo_language_auc'),('Leave-one-generator-out','logo_generator_auc')]:
  g=metrics[metrics.scope==scope].sort_values('roc_auc');ax=sns.barplot(g,x='roc_auc',y='value',color='#2878B5');ax.axvline(.5,ls='--',c='gray');ax.set(xlim=(0,1),xlabel='ROC-AUC',ylabel='');plt.tight_layout();plt.savefig(figdir/f'{name}.pdf');plt.close()
 obs,mean=calibration_curve(test.label,p,n_bins=10,strategy='quantile');plt.plot(mean,obs,'o-',label='Verifier');plt.plot([0,1],[0,1],'--',c='gray');plt.xlabel('Mean confidence');plt.ylabel('Observed correctness');plt.legend();plt.tight_layout();plt.savefig(figdir/'calibration.pdf');plt.close()
 sns.histplot(data=pred,x='confidence',hue='label',bins=20,stat='density',common_norm=False);plt.tight_layout();plt.savefig(figdir/'confidence_histogram.pdf');plt.close()
 top=imp.nlargest(15,'mean_abs_linear_shap');sns.barplot(top,y='feature',x='mean_abs_linear_shap',hue='family',dodge=False);plt.tight_layout();plt.savefig(figdir/'evidence_importance.pdf');plt.close()
 pivot=pd.DataFrame(prog).groupby(['family','quality_score'])['mean'].mean().unstack();sns.heatmap(pivot,annot=True,fmt='.2f',cmap='viridis');plt.tight_layout();plt.savefig(figdir/'quality_progression_heatmap.pdf');plt.close()
 cm=confusion_matrix(test.label,pred.predicted);sns.heatmap(cm,annot=True,fmt='d',cmap='Blues');plt.xlabel('Predicted');plt.ylabel('Actual');plt.tight_layout();plt.savefig(figdir/'confusion_matrix.pdf');plt.close()
 # exact dataset distribution and reproducibility
 dist=df.groupby(['language','generator','quality_score']).size().rename('count').reset_index();dist.to_csv(out/'tables'/'dataset_distribution.csv',index=False)
 monotonic=pd.DataFrame(prog).groupby(['language','generator','family','quality_score'])['mean'].mean().unstack();monotonic['monotonic_1_3']=(monotonic[1]<=monotonic[2])&(monotonic[2]<=monotonic[3]);monotonic.reset_index().to_csv(out/'statistics'/'progression_monotonicity.csv',index=False)
 manifest={'config':cfg,'rows':len(df),'problems':df.problem_id.nunique(),'features':features,'forbidden_overlap':sorted(set(features)&FORBIDDEN),'git_revision':subprocess.run(['git','rev-parse','HEAD'],capture_output=True,text=True).stdout.strip(),'python':sys.version,'platform':platform.platform(),'packages':{'numpy':np.__version__,'pandas':pd.__version__},'dataset_sha256':hashlib.sha256((out/'datasets'/'features.jsonl').read_bytes()).hexdigest()};(out/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2))
 print(metrics[['scope','value','n','roc_auc','f1','brier','ece']].to_string(index=False));return 0
if __name__=='__main__':raise SystemExit(main())
