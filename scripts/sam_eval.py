import json, csv, sys, math
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from run_benchR import load_benchR
rows=list(csv.DictReader(open('build/sam/mask_features.csv',encoding='utf-8')))
for r in rows:
    for k in ('conf','verifier','mask_area','mask_area_ratio','mask_fill','compactness','gt_iou'):
        r[k]=float(r[k])
    r['y']=int(r['y'])
def feats(r,cols): return [r[c] for c in cols]
base=['conf','verifier']; mask=['mask_fill','mask_area_ratio','compactness']; comb=base+mask
def auc(sub,cols):
    y=np.array([r['y'] for r in sub]); s=np.array([sum(feats(r,cols)) if len(cols)>1 else feats(r,cols)[0] for r in sub])
    return round(float(roc_auc_score(y,s)),4) if len(set(y))>1 else None
tr=[r for r in rows if r['set']=='val_cal']; te={k:[r for r in rows if r['set']==k] for k in ('val_report','dev')}
model=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced'))
model.fit([feats(r,comb) for r in tr],[r['y'] for r in tr])
model_mask=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced'))
model_mask.fit([feats(r,mask) for r in tr],[r['y'] for r in tr])
for r in rows:
    r['p_mask']=float(model.predict_proba([feats(r,comb)])[0][1])
    r['p_mask_only']=float(model_mask.predict_proba([feats(r,mask)])[0][1])
def choose(sub,col,target):
    y=[r[col] for r in sub]; pos=[r for r in sub if r['y']==1]; neg=[r for r in sub if r['y']==0]
    best=None
    for thr in sorted(set(y),reverse=True):
        tp=sum(1 for r in pos if r[col]>=thr); fp=sum(1 for r in neg if r[col]>=thr)
        fpr=fp/max(len(neg),1); rec=tp/max(len(pos),1)
        if fpr<=target and (best is None or rec>best[0]): best=(rec,fpr,float(thr),tp,fp)
    return best
def eval_metrics(sub,col,thr):
    pos=[r for r in sub if r['y']==1]; neg=[r for r in sub if r['y']==0]
    tp=sum(1 for r in pos if r[col]>=thr); fp=sum(1 for r in neg if r[col]>=thr)
    return {'recall':round(tp/max(len(pos),1),4),'fpr':round(fp/max(len(neg),1),4),'tp':tp,'fp':fp,'n_pos':len(pos),'n_neg':len(neg)}
def mcn(a,b,ids,col_a,col_b,ta,tb):
    b01=sum(1 for r in ids if r[col_a]>=ta and not r[col_b]>=tb); b10=sum(1 for r in ids if not r[col_a]>=ta and r[col_b]>=tb)
    p=binomtest(min(b01,b10),b01+b10,0.5).pvalue if b01+b10 else 1.0
    return {'a_only':b01,'b_only':b10,'p':float(p)}
res={'detection_rows':len(rows),'train_val_cal':len(tr),'auc':{},'detection_matched_fpr':{},'controlled_vs_uncontrolled':{}}
for name,sub in [('val_cal',tr),*te.items()]:
    res['auc'][name]={'conf':auc(sub,['conf']),'verifier':auc(sub,['verifier']),
                      'mask_fill':auc(sub,['mask_fill']),'mask_area_ratio':auc(sub,['mask_area_ratio']),
                      'compactness':auc(sub,['compactness']),'mask_only':round(float(roc_auc_score([r['y'] for r in sub],[r['p_mask_only'] for r in sub])),4) if len(set(r['y'] for r in sub))>1 else None,'verifier_plus_mask':round(float(roc_auc_score([r['y'] for r in sub],[r['p_mask'] for r in sub])),4) if len(set(r['y'] for r in sub))>1 else None}
for target in (0.02,0.05,0.10):
    res['detection_matched_fpr'][str(target)]={}
    for name,sub in te.items():
        b=choose(sub,'verifier',target); c=choose(sub,'p_mask',target)
        if not b or not c: continue
        res['detection_matched_fpr'][str(target)][name]={'verifier':{**eval_metrics(sub,'verifier',b[2]),'thr':b[2]},
            'verifier_plus_mask':{**eval_metrics(sub,'p_mask',c[2]),'thr':c[2]},
            'mcnemar_pos':mcn(sub,sub,[r for r in sub if r['y']==1],'verifier','p_mask',b[2],c[2]),
            'mcnemar_neg':mcn(sub,sub,[r for r in sub if r['y']==0],'verifier','p_mask',b[2],c[2])}
# controlled vs uncontrolled on DEV positives
recs,member=load_benchR(); ft={r.id:(r.meta.get('flame_types') or ['unknown']) for r in recs if member[r.id]=='dev'}
devpos=[r for r in rows if r['set']=='dev' and r['y']==1]
ctrl=[r for r in devpos if any(t == 'controlled' for t in ft.get(r['id'],[])) and not any(t == 'uncontrolled' for t in ft.get(r['id'],[]))]
unc=[r for r in devpos if any(t == 'uncontrolled' for t in ft.get(r['id'],[])) and not any(t == 'controlled' for t in ft.get(r['id'],[]))]
res['controlled_vs_uncontrolled']={'n_controlled':len(ctrl),'n_uncontrolled':len(unc),
 'fill_median_controlled':round(float(np.median([r['mask_fill'] for r in ctrl])),4) if ctrl else None,
 'fill_median_uncontrolled':round(float(np.median([r['mask_fill'] for r in unc])),4) if unc else None,
 'area_median_controlled':round(float(np.median([r['mask_area_ratio'] for r in ctrl])),4) if ctrl else None,
 'area_median_uncontrolled':round(float(np.median([r['mask_area_ratio'] for r in unc])),4) if unc else None,
 'auc_fill_controlled_vs_uncontrolled':round(float(roc_auc_score([1]*len(ctrl)+[0]*len(unc),[r['mask_fill'] for r in ctrl+unc])),4) if ctrl and unc else None,
 'auc_area_controlled_vs_uncontrolled':round(float(roc_auc_score([1]*len(ctrl)+[0]*len(unc),[r['mask_area_ratio'] for r in ctrl+unc])),4) if ctrl and unc else None,
 'note':'only detections with P50 conf>=0.10; controlled/uncontrolled labels from bench_R DEV flame_type'}
Path('docs/benchR/sam_experiment.json').write_text(json.dumps(res,indent=2),encoding='utf-8')
print(json.dumps(res,indent=2))
