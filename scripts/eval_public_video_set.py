from __future__ import annotations
import json, math, sys, csv
from pathlib import Path
from collections import Counter, defaultdict
import cv2, numpy as np
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT, load_config
from firesight.temporal import TemporalConfirmer
from firesight.types import Box, Detection
from firesight.video_filter import FilterConfig, VideoFalseAlarmFilter
WIL=lambda k,n,z=1.96:(None if n==0 else (lambda p,d,c,h:[round(max(0,c-h),4),round(min(1,c+h),4)])(k/n,1+z*z/n,(k/n+z*z/(2*n))/(1+z*z/n),z*math.sqrt((k/n*(1-k/n)+z*z/(4*n))/n)/(1+z*z/n)))
THR={'fire':0.29,'smoke':0.33}; FPS=2.0
def font(sz=10):
    try:
        from PIL import ImageFont
        return ImageFont.truetype('arial.ttf',sz)
    except Exception:
        from PIL import ImageFont
        try: return ImageFont.load_default(size=sz)
        except Exception: return ImageFont.load_default()
def main():
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument('--manifest',default='data/video_eval/public/curated_manifest.jsonl'); ap.add_argument('--out',default='build/video_eval/results'); ap.add_argument('--model',default='build/kaggle_weights/runs/P50/weights/best.onnx'); ap.add_argument('--name',default='P50'); ap.add_argument('--fire-thr',type=float,default=0.29); ap.add_argument('--smoke-thr',type=float,default=0.33); ap.add_argument('--smoke-static-min-conf',type=float,default=None); ap.add_argument('--imgsz',type=int,default=640); ap.add_argument('--assume-smoke',action='store_true'); args=ap.parse_args()
    rows=[json.loads(l) for l in open(args.manifest,encoding='utf-8') if l.strip()]
    out=PROJECT_ROOT/args.out; (out/'annotated').mkdir(parents=True,exist_ok=True)
    from ultralytics import YOLO
    model=YOLO(str(PROJECT_ROOT/args.model),task='detect')
    names={int(k):str(v).lower() for k,v in (model.names or {}).items()}
    if args.assume_smoke:
        names={k:'smoke' for k in names}
    THR={'fire':args.fire_thr,'smoke':args.smoke_thr}
    cfg=load_config(); cfg.temporal.tracker='iou'
    if args.smoke_static_min_conf is not None:
        cfg.temporal.smoke_static_min_conf=args.smoke_static_min_conf
    fcfg=FilterConfig()
    if args.smoke_static_min_conf is not None:
        fcfg.smoke_static_min_conf=args.smoke_static_min_conf
    results=[]; best_tiles=[]
    for c in rows:
        p=PROJECT_ROOT/c['path']; cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS) or 25
        step=max(1,round(fps/FPS)); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        limit=c.get('max_seconds') or (n/max(fps,1)); dest=out/'annotated'/f"{c['id'].replace(':','_')}_annotated.mp4"
        wr=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),FPS,(w,h))
        tc=TemporalConfirmer(cfg.temporal); vf=VideoFalseAlarmFilter(FilterConfig(),out/f"{c['id'].replace(':','_')}_filter.jsonl")
        idx=sampled=0; raw=temp=layer=False; maxf=maxs=0.0; best=None
        while True:
            ok,fr=cap.read()
            if not ok: break
            t=idx/fps
            if t>limit: break
            if idx%step: idx+=1; continue
            r=model.predict(fr,imgsz=args.imgsz,conf=0.05,device='cpu',verbose=False)[0]
            dets=[]
            for b in r.boxes:
                nm=names.get(int(b.cls.item()),''); lab='fire' if 'fire' in nm else 'smoke' if 'smoke' in nm else None
                if lab is None: continue
                conf=float(b.conf.item())
                if lab=='fire': maxf=max(maxf,conf)
                else: maxs=max(maxs,conf)
                dets.append(Detection(lab,conf,Box(*[float(v) for v in b.xyxy[0].tolist()])))
            if any(d.confidence>=THR[d.label] for d in dets): raw=True
            tconf=tc.update(sampled,t,dets,size=(h,w))
            if tconf.confirmed:
                temp=True
                kept,review,supp=vf.update(c['id'],t,fr,tconf.confirmed)
                if kept: layer=True
            ann=fr.copy()
            for d in dets:
                if d.confidence<0.10: continue
                b=d.box.to_list(); x1,y1,x2,y2=map(int,b)
                col=(0,0,255) if d.label=='fire' else (0,140,255)
                cv2.rectangle(ann,(x1,y1),(x2,y2),col,2); cv2.putText(ann,f'{d.label} {d.confidence:.2f}',(x1,max(12,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,0.5,col,2)
            cv2.putText(ann,f"{c['label']} t={t:.1f}s maxF={maxf:.2f} maxS={maxs:.2f}",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            wr.write(ann)
            if best is None or max(maxf,maxs)>best[0]: best=(max(maxf,maxs),ann.copy(),t,maxf,maxs)
            sampled+=1; idx+=1
        cap.release(); wr.release()
        rec={**c,'eval_seconds':round(min(limit,n/max(fps,1)),1),'sampled_frames':sampled,'max_fire':round(maxf,4),'max_smoke':round(maxs,4),
             'raw_alert':raw,'temporal_alert':temp,'layer_alert':layer,'annotated':str(dest.relative_to(PROJECT_ROOT))}
        results.append(rec)
        if best: best_tiles.append((rec,best))
        print(c['label'],c['category'],c['id'][:20],'raw',raw,'temp',temp,'layer',layer,'max',round(maxf,2),round(maxs,2),flush=True)
    # metrics
    pos=[r for r in results if r['label'] in ('fire','smoke')]; neg=[r for r in results if r['label']=='nonfire']
    metrics={}
    for mode in ('raw_alert','temporal_alert','layer_alert'):
        tp=sum(1 for r in pos if r[mode]); fn=len(pos)-tp; fp=sum(1 for r in neg if r[mode]); tn=len(neg)-fp
        prec=tp/(tp+fp) if tp+fp else None; rec=tp/len(pos) if pos else None; spec=tn/len(neg) if neg else None
        metrics[mode]={'tp':tp,'fp':fp,'tn':tn,'fn':fn,'precision':None if prec is None else round(prec,4),
                       'recall':None if rec is None else round(rec,4),'recall_ci':WIL(tp,len(pos)) if pos else [None,None],'specificity':None if spec is None else round(spec,4),'fpr':round(fp/len(neg),4) if neg else None,'fpr_ci':WIL(fp,len(neg)) if neg else [None,None],
                       'accuracy':round((tp+tn)/(len(pos)+len(neg)),4)}
    percat={}
    for cat in sorted({r['category'] for r in results}):
        sub=[r for r in results if r['category']==cat]; p=sum(1 for r in sub if r['label'] in ('fire','smoke'))
        percat[cat]={'n':len(sub),'positives':p,'layer_alerts':sum(1 for r in sub if r['layer_alert']),'raw_alerts':sum(1 for r in sub if r['raw_alert'])}
    nonfire_hours=sum(r['eval_seconds'] for r in neg)/3600
    summary={'n':len(results),'positives':len(pos),'negatives':len(neg),'metrics':metrics,'per_category':percat,
             'nonfire_hours_evaluated':round(nonfire_hours,3),'layer_fa_per_hour':round(sum(1 for r in neg if r['layer_alert'])/nonfire_hours,3) if nonfire_hours else None,
             'note':'2 fps; nonfire clips capped at 120 s; fire/smoke full clip','model':args.name,'thresholds':THR,'smoke_static_min_conf':cfg.temporal.smoke_static_min_conf}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    with (out/'per_clip.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['id','label','category','title','eval_seconds','sampled_frames','max_fire','max_smoke','raw_alert','temporal_alert','layer_alert','annotated']); w.writeheader()
        for r in results: w.writerow({k:r.get(k) for k in w.fieldnames})
    # contact sheet of best frames
    from PIL import Image, ImageDraw
    cols=6; tw,th=300,220; rowsn=(len(best_tiles)+cols-1)//cols
    S=Image.new('RGB',(cols*tw,rowsn*th),(15,15,18)); d=ImageDraw.Draw(S)
    for i,(rec,b) in enumerate(best_tiles):
        im=Image.fromarray(cv2.cvtColor(b[1],cv2.COLOR_BGR2RGB)); im.thumbnail((tw-8,th-44))
        x=(i%cols)*tw+4; y=(i//cols)*th+20; S.paste(im,(x,y))
        d.text((x,y-17),f"{rec['label']} {rec['category']} {rec['id'][8:30]} L={rec['layer_alert']}",fill=(255,255,255),font=font(8))
    S.save(out/'contact_sheet.jpg',quality=84)
    print(json.dumps(summary,indent=2))
if __name__=='__main__': raise SystemExit(main())
