from __future__ import annotations
import json, sys, random, time, math
from pathlib import Path
from collections import defaultdict
import numpy as np, cv2, joblib
from PIL import Image
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT
from ultralytics import SAM

OUT=PROJECT_ROOT/'build/sam'; OUT.mkdir(parents=True,exist_ok=True)
def load_dets(p): return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]
def load_sample(p): return {r['id']:r for r in json.loads(Path(p).read_text(encoding='utf-8'))}
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
def vmap(feat_path):
    d=np.load(feat_path,allow_pickle=True); m=[json.loads(x) for x in d['meta']]
    model=joblib.load('build/verifier/model_v4.joblib')['model']; sc=model.predict_proba(d['X'])[:,1]
    out={}
    for mm,s in zip(m,sc):
        key=(mm['id'],round(float(mm['conf']),6)); out[key]=max(out.get(key,0.0),float(s))
    return out
sets={
 'val_cal':('build/verifier/det_val_cal.jsonl','build/verifier/sample_val_cal.json','build/verifier/features_val_cal.npz',500),
 'val_report':('build/verifier/det_val_report.jsonl','build/verifier/sample_val_report.json','build/verifier/features_val_report.npz',500),
 'dev':('build/verifier/det_dev.jsonl','build/verifier/sample_dev.json','build/verifier/features_dev.npz',99999),
}
sam=SAM('mobile_sam.pt')
allrows=[]; timing=[]
for name,(dpath,spath,fpath,nmax) in sets.items():
    rows=[r for r in load_dets(dpath) if r.get('detections')]
    sample=load_sample(spath); vm=vmap(fpath)
    flat=[]
    for r in rows:
        for d in r['detections']:
            if d['conf']<0.10: continue
            flat.append((r['id'],d))
    random.Random(20261102).shuffle(flat); flat=flat[:nmax]
    by=defaultdict(list)
    for rid,d in flat: by[rid].append(d)
    print(name,'images',len(by),'dets',len(flat),flush=True)
    for rid,dets in by.items():
        s=sample.get(rid); 
        if not s: continue
        p=PROJECT_ROOT/s['path']
        try: im=Image.open(p).convert('RGB'); W,H=im.size
        except Exception: continue
        t=time.perf_counter()
        try: res=sam(str(p),bboxes=[d['box'] for d in dets],verbose=False)[0]
        except Exception: continue
        timing.append((time.perf_counter()-t)/max(len(dets),1))
        gts=[g['box'] for g in s.get('labels',[])]
        for j,(d,mask) in enumerate(zip(dets,res.masks.xy if res.masks is not None else [])):
            area=float(cv2.contourArea(np.array(mask,dtype=np.float32))) if len(mask)>=3 else 0.0
            b=d['box']; ba=max((b[2]-b[0])*(b[3]-b[1]),1e-6)
            per=float(cv2.arcLength(np.array(mask,dtype=np.float32),True)) if len(mask)>=3 else 0.0
            fill=max(0.0,min(2.0,area/ba)); aratio=area/max(W*H,1); compact=(4*math.pi*area/(per*per)) if per>0 else 0.0
            best=max((iou(b,g) for g in gts),default=0.0)
            y=1 if best>=0.3 else 0
            allrows.append({'set':name,'id':rid,'conf':d['conf'],'label':d['label'],'box':b,'w':W,'h':H,
                            'gt_iou':best,'y':y,'verifier':vm.get((rid,round(float(d['conf']),6)),0.0),
                            'mask_area':area,'mask_area_ratio':aratio,'mask_fill':fill,'compactness':compact})
import csv
with (OUT/'mask_features.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(allrows[0])); w.writeheader(); w.writerows(allrows)
print('rows',len(allrows),'sam ms/det',round(float(np.mean(timing))*1000,2),'median',round(float(np.median(timing))*1000,2))
print('by set', {k:sum(1 for r in allrows if r['set']==k) for k in sets})
