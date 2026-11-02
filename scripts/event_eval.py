"""Event-level evaluation harness.

Manifest JSONL per clip:
  {"id","path","label":"fire"/"nonfire","onset_s":<float or null>,"camera_id","source","licence"}

Runs P50 / E1 / P50@0.10+verifier v4 through temporal N-of-M and the video layer, then reports:
  - fire recall within 5/10/30 s of onset
  - time-to-detection (TTD) mean/median/p90
  - non-fire alarm episodes per camera-hour
  - Wilson CIs for proportions and cluster-bootstrap CIs for rates/TTD
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
from collections import defaultdict
import cv2, numpy as np
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT, load_config
from firesight.temporal import TemporalConfirmer
from firesight.types import Box, Detection
from firesight.video_filter import FilterConfig, VideoFalseAlarmFilter

def wilson(k,n,z=1.96):
    if n==0: return [None,None]
    p=k/n; d=1+z*z/n; c=(p+z*z/(2*n))/d; h=z*math.sqrt((p*(1-p)+z*z/(4*n))/n)/d
    return [round(max(0,c-h),4),round(min(1,c+h),4)]

def episodes(times,gap=1.5):
    eps=[]
    for t in times:
        if not eps or t-eps[-1][-1]>gap: eps.append([t])
        else: eps[-1].append(t)
    return eps

def boot_ci(values, n=1000, seed=0):
    if not values: return [None,None]
    rng=np.random.default_rng(seed); vals=np.array(values,dtype=float)
    bs=[float(np.mean(rng.choice(vals,len(vals),replace=True))) for _ in range(n)]
    return [round(float(np.percentile(bs,2.5)),3),round(float(np.percentile(bs,97.5)),3)]

class Verifier:
    def __init__(self,path):
        import joblib, torch
        from transformers import CLIPModel, CLIPProcessor
        self.b=joblib.load(path); self.m=self.b['model']
        self.clip=CLIPModel.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K').eval()
        self.proc=CLIPProcessor.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
        self.torch=torch
    def _crop(self,im,b,s=1.0):
        W,H=im.size; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*s,(b[3]-b[1])*s
        x1,y1=max(0,int(cx-w/2)),max(0,int(cy-h/2)); x2,y2=min(W,int(cx+w/2)),min(H,int(cy+h/2))
        return im.crop((x1,y1,max(x1+1,x2),max(y1+1,y2)))
    def score(self,frame,dets):
        if not dets: return []
        from PIL import Image
        im=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); W,H=im.size
        full=im.copy(); full.thumbnail((448,448)); crops=[]
        for d in dets:
            b=d.box.to_list(); crops += [self._crop(im,b,1.15),self._crop(im,b,2.5)]
        embs=[]
        for s in range(0,len(crops),32):
            inp=self.proc(images=crops[s:s+32],return_tensors='pt')
            with self.torch.no_grad():
                vo=self.clip.vision_model(pixel_values=inp['pixel_values']); f=self.clip.visual_projection(vo.pooler_output); embs.append(f/f.norm(dim=-1,keepdim=True))
        E=self.torch.cat(embs)
        inp=self.proc(images=[full],return_tensors='pt')
        with self.torch.no_grad():
            vo=self.clip.vision_model(pixel_values=inp['pixel_values']); ff=self.clip.visual_projection(vo.pooler_output); ff=ff/ff.norm(dim=-1,keepdim=True)
        rows=[]
        for j,d in enumerate(dets):
            b=d.box.to_list(); bw,bh=(b[2]-b[0])/max(W,1),(b[3]-b[1])/max(H,1)
            nums=np.array([d.confidence,1.0 if d.label=='fire' else 0.0,bw,bh,bw*bh,bw/max(bh,1e-6),W/640.0,H/640.0],dtype=np.float32)
            rows.append(np.concatenate([E[2*j].numpy(),E[2*j+1].numpy(),ff[0].numpy(),nums]))
        return self.m.predict_proba(np.stack(rows))[:,1].tolist()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--name',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--fire-thr',type=float,default=0.29)
    ap.add_argument('--smoke-thr',type=float,default=0.33)
    ap.add_argument('--verifier',default=None)
    ap.add_argument('--verifier-v',type=float,default=0.011936064809560776)
    ap.add_argument('--sample-fps',type=float,default=5.0)
    ap.add_argument('--max-seconds',type=float,default=None)
    ap.add_argument('--no-video-layer',action='store_true')
    args=ap.parse_args()
    from ultralytics import YOLO
    model=YOLO(args.model,task='detect')
    ver=Verifier(args.verifier) if args.verifier else None
    cfg=load_config(); cfg.temporal.tracker='iou'
    clips=[json.loads(l) for l in open(args.manifest,encoding='utf-8') if l.strip()]
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for c in clips:
        cap=cv2.VideoCapture(str(PROJECT_ROOT/c['path'])); fps=cap.get(cv2.CAP_PROP_FPS) or 25
        step=max(1,round(fps/args.sample_fps)); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur=n/max(fps,1)
        tc=TemporalConfirmer(cfg.temporal); vf=VideoFalseAlarmFilter(FilterConfig(),out/f"{c['id']}_filter.jsonl")
        idx=sampled=0; alarms=[]; raw=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if idx%step: idx+=1; continue
            t=idx/fps
            r=model.predict(frame,imgsz=640,conf=0.05,device='cpu',verbose=False)[0]
            dets=[Detection('fire' if int(b.cls.item())==0 else 'smoke',float(b.conf.item()),Box(*[float(v) for v in b.xyxy[0].tolist()])) for b in r.boxes]
            if dets: raw+=1
            if ver is not None:
                cand=[d for d in dets if d.confidence>=0.10]; scores=ver.score(frame,cand) if cand else []
                dets=[d for d,s in zip(cand,scores) if s>=args.verifier_v]
            dets=[d for d in dets if d.confidence>=(args.fire_thr if d.label=='fire' else args.smoke_thr)]
            temp=tc.update(sampled,t,dets,size=(frame.shape[0],frame.shape[1]))
            if temp.confirmed:
                if args.no_video_layer: alarms.append(round(t,2))
                else:
                    kept,review,supp=vf.update(c['id'],t,frame,temp.confirmed)
                    if kept: alarms.append(round(t,2))
            sampled+=1; idx+=1
        cap.release()
        eps=episodes(alarms); row={**c,'duration_s':round(dur,2),'sampled_frames':sampled,'raw_frames_with_det':raw,
                                  'alarm_times':alarms,'alarm_episodes':len(eps),'first_alarm_s':alarms[0] if alarms else None}
        if c.get('onset_s') is not None and alarms:
            row['ttd_s']=round(alarms[0]-float(c['onset_s']),2)
            row['alarm_after_onset']=alarms[0]>=float(c['onset_s'])
        rows.append(row)
        print(c['id'],c.get('label'),row['first_alarm_s'],row.get('ttd_s'),flush=True)
    fire=[r for r in rows if r.get('label')=='fire' and r.get('onset_s') is not None]
    nonfire=[r for r in rows if r.get('label')=='nonfire']
    summary={'name':args.name,'n_clips':len(rows),'fire_clips':len(fire),'nonfire_clips':len(nonfire),'nonfire_hours':round(sum(r['duration_s'] for r in nonfire)/3600,4)}
    for N in (5,10,30):
        k=sum(1 for r in fire if r.get('first_alarm_s') is not None and r['first_alarm_s']>=r['onset_s'] and r['first_alarm_s']<=r['onset_s']+N)
        summary[f'recall_within_{N}s']=round(k/max(len(fire),1),4); summary[f'recall_within_{N}s_ci']=wilson(k,max(len(fire),1))
    ttds=[r['ttd_s'] for r in fire if r.get('ttd_s') is not None and r['ttd_s']>=0]
    summary['ttd_mean_s']=round(float(np.mean(ttds)),2) if ttds else None
    summary['ttd_median_s']=round(float(np.median(ttds)),2) if ttds else None
    summary['ttd_p90_s']=round(float(np.percentile(ttds,90)),2) if ttds else None
    summary['ttd_median_ci']=boot_ci(ttds)
    eps=sum(r['alarm_episodes'] for r in nonfire); hours=summary['nonfire_hours'] or 1e-9
    summary['nonfire_alarm_episodes']=eps; summary['fa_per_camera_hour']=round(eps/hours,4)
    summary['fa_per_camera_hour_ci']=boot_ci([r['alarm_episodes']/max(r['duration_s']/3600,1e-9) for r in nonfire]) if nonfire else [None,None]
    (out/'per_clip.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': raise SystemExit(main())
