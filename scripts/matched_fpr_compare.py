import json, sys, math
from pathlib import Path
import numpy as np
from scipy.stats import binomtest
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from firesight.config import PROJECT_ROOT
from run_benchR import load_benchR

def load_preds(path):
    out={}
    for l in open(path,encoding='utf-8'):
        if not l.strip(): continue
        r=json.loads(l); out[r['id']]=max((float(d.get('confidence',d.get('conf',0))) for d in r.get('detections',[])),default=0.0)
    return out
def labels_from_sample(path):
    return {r['id']:int(bool(r.get('labels'))) for r in json.loads(Path(path).read_text(encoding='utf-8'))}
def metrics(scores,labels,thr):
    pos=[i for i,y in labels.items() if y]; neg=[i for i,y in labels.items() if not y]
    tp=sum(1 for i in pos if scores.get(i,0)>=thr); fp=sum(1 for i in neg if scores.get(i,0)>=thr)
    return tp,fp,len(pos),len(neg)
def choose(scores,labels,target):
    pos=[i for i,y in labels.items() if y]; neg=[i for i,y in labels.items() if not y]
    best=None
    for thr in sorted(set(scores.values()),reverse=True):
        tp=sum(1 for i in pos if scores.get(i,0)>=thr); fp=sum(1 for i in neg if scores.get(i,0)>=thr)
        fpr=fp/max(len(neg),1); rec=tp/max(len(pos),1) if pos else 0.0
        if pos:
            if fpr<=target and (best is None or rec>best[0] or (rec==best[0] and fpr<best[1])):
                best=(rec,fpr,float(thr),tp,fp)
        else:
            if fpr<=target and (best is None or fpr>best[1]):
                best=(0.0,fpr,float(thr),0,fp)
    return best
def mcn(a,b,ids,ta,tb):
    b01=sum(1 for i in ids if a.get(i,0)>=ta and not b.get(i,0)>=tb); b10=sum(1 for i in ids if not a.get(i,0)>=ta and b.get(i,0)>=tb)
    p=binomtest(min(b01,b10),b01+b10,0.5).pvalue if b01+b10 else 1.0
    return {'a_only':b01,'b_only':b10,'p':float(p)}

recs,member=load_benchR(); dev=[r for r in recs if member[r.id]=='dev']; dev_labels={r.id:int(bool(r.meta.get('has_fire') or r.meta.get('has_smoke'))) for r in dev}
report_labels=labels_from_sample('build/verifier/sample_val_report.json')
indep_ids=set(load_preds('build/stage_b/preds_E1_indep.jsonl')); indep_labels={i:0 for i in indep_ids}
# P50
p50_dev=load_preds('bench_results/predictions/P50_benchR.jsonl'); p50_dev={i:p50_dev[i] for i in dev_labels if i in p50_dev}
p50_rep=load_preds('build/pilot_eval/preds_P50.jsonl'); p50_rep={i:p50_rep[i] for i in report_labels if i in p50_rep}
p50_ind=load_preds('build/verifier/det_eval_neg.jsonl'); p50_ind={i:p50_ind[i] for i in indep_ids if i in p50_ind}
# E1
e1_dev_raw=load_preds('build/stage_b/preds_E1_benchR_dev.jsonl'); _m={Path(r.path).stem:r.id for r in dev}; e1_dev={_m[k]:v for k,v in e1_dev_raw.items() if k in _m}; e1_rep=load_preds('build/stage_b/preds_E1_report.jsonl'); e1_ind=load_preds('build/stage_b/preds_E1_indep.jsonl')
# Salah
sa_dev=load_preds('bench_results/predictions/SalahYOLOv26_benchR.jsonl'); sa_dev={i:sa_dev[i] for i in dev_labels if i in sa_dev}
sa_rep=load_preds('build/stage_b/preds_Salah_report.jsonl'); sa_ind=load_preds('build/stage_b/preds_Salah_indep.jsonl')
sets={'bench_R_DEV':({'P50':p50_dev,'E1':e1_dev,'SalahYOLOv26':sa_dev},dev_labels),
      'val_report':({'P50':p50_rep,'E1':e1_rep,'SalahYOLOv26':sa_rep},report_labels),
      'independent_neg':({'P50':p50_ind,'E1':e1_ind,'SalahYOLOv26':sa_ind},indep_labels)}
out={}
for sname,(models,labels) in sets.items():
    out[sname]={}
    for target in (0.009,0.015,0.02):
        row={}
        for m,sc in models.items():
            b=choose(sc,labels,target); row[m]=None if b is None else {'recall':round(b[0],4),'fpr':round(b[1],4),'threshold':b[2],'tp':b[3],'fp':b[4]}
        # mcnemar P50 vs Salah, E1 vs Salah on pos/neg
        pos=[i for i,y in labels.items() if y]; neg=[i for i,y in labels.items() if not y]
        if row.get('P50') and row.get('SalahYOLOv26'):
            row['mcn_P50_vs_Salah_pos']=mcn(models['P50'],models['SalahYOLOv26'],pos,row['P50']['threshold'],row['SalahYOLOv26']['threshold'])
            row['mcn_P50_vs_Salah_neg']=mcn(models['P50'],models['SalahYOLOv26'],neg,row['P50']['threshold'],row['SalahYOLOv26']['threshold'])
        if row.get('E1') and row.get('SalahYOLOv26'):
            row['mcn_E1_vs_Salah_pos']=mcn(models['E1'],models['SalahYOLOv26'],pos,row['E1']['threshold'],row['SalahYOLOv26']['threshold'])
            row['mcn_E1_vs_Salah_neg']=mcn(models['E1'],models['SalahYOLOv26'],neg,row['E1']['threshold'],row['SalahYOLOv26']['threshold'])
        out[sname][str(target)]=row
Path('docs/benchR/matched_fpr_external.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
