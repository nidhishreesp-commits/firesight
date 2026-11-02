import sys, json, time, math
from pathlib import Path
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from run_benchR import load_benchR
from ultralytics import YOLO
WIL=lambda k,n,z=1.96: (None if n==0 else (lambda p,d,c,h:[round(max(0,c-h),4),round(min(1,c+h),4)])(k/n,1+z*z/n,(k/n+z*z/(2*n))/(1+z*z/n),z*math.sqrt((k/n*(1-k/n)+z*z/(4*n))/n)/(1+z*z/n)))
THR={'fire':0.13,'smoke':0.39}
recs,member=load_benchR(); dev=[r for r in recs if member[r.id]=='dev']
keys=['aerial','drone','wildfire','wild fire','air tanker','airtanker','helicopter','from above','overlook','hillside','mountain','forest fire','wildland','landscape','high angle']
sub=[r for r in dev if any(k in (r.id+' '+Path(r.path).stem).lower() for k in keys)]
pos=[r for r in sub if r.meta.get('has_fire') or r.meta.get('has_smoke')]; neg=[r for r in sub if not (r.meta.get('has_fire') or r.meta.get('has_smoke'))]
m=YOLO('build/kaggle_weights/runs/P50/weights/best.onnx',task='detect')
res={}
for img in (640,960,1280):
    times=[]; scores={}
    for r in sub:
        t=time.perf_counter(); out=m.predict([str(r.path)],imgsz=img,conf=0.01,batch=1,device='cpu',verbose=False)[0]; times.append((time.perf_counter()-t)*1000)
        scores[r.id]=max((float(b.conf.item()) for b in out.boxes),default=0.0)
    tp=sum(1 for r in pos if scores.get(r.id,0)>=min(THR.values()))
    # class-specific exact: need labels; use max conf only as proxy? compute with detections labels
    # redo class-specific using out boxes labels
    tp=fp=0; tn=fn=0
    for r in sub:
        out=m.predict([str(r.path)],imgsz=img,conf=0.01,batch=1,device='cpu',verbose=False)[0]
        alarm=any((('fire' if int(b.cls.item())==0 else 'smoke') in THR) and float(b.conf.item())>=THR['fire' if int(b.cls.item())==0 else 'smoke'] for b in out.boxes)
        y=bool(r.meta.get('has_fire') or r.meta.get('has_smoke'))
        if y and alarm: tp+=1
        elif y: fn+=1
        elif alarm: fp+=1
        else: tn+=1
    res[str(img)]={'n':len(sub),'pos':len(pos),'neg':len(neg),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                   'recall':tp/len(pos) if pos else None,'recall_ci':WIL(tp,len(pos)) if pos else None,
                   'fpr':fp/len(neg) if neg else None,'fpr_ci':WIL(fp,len(neg)) if neg else None,
                   'mean_ms':round(sum(times)/len(times),2),'median_ms':round(sorted(times)[len(times)//2],2)}
    print(img,res[str(img)])
Path('build/resolution_test_dev.json').write_text(json.dumps({'subset':'high_mounted_aerial_keyword_proxy','n':len(sub),'pos':len(pos),'neg':len(neg),'results':res},indent=2),encoding='utf-8')
