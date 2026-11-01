from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--input',default=r'C:\Users\Nidhi\My projects\Sample_videos')
    ap.add_argument('--model',default='build/kaggle_weights/runs/P50/weights/best.onnx')
    ap.add_argument('--name',default='P50')
    ap.add_argument('--out',default='build/annotated')
    ap.add_argument('--conf',type=float,default=0.05)
    ap.add_argument('--draw-conf',type=float,default=0.10)
    ap.add_argument('--sample-fps',type=float,default=5.0)
    ap.add_argument('--only',default=None,help='comma-separated video names')
    args=ap.parse_args()
    src=Path(args.input); out=Path(args.out)/args.name; out.mkdir(parents=True,exist_ok=True)
    model=YOLO(args.model,task='detect')
    names={int(k):str(v).lower() for k,v in (model.names or {}).items()}
    fire_thr=0.29; smoke_thr=0.33
    files=[p for p in sorted(src.iterdir()) if p.suffix.lower() in ('.mp4','.mov','.avi','.mkv')]
    if args.only:
        want={x.strip() for x in args.only.split(',') if x.strip()}
        files=[p for p in files if p.name in want]
    summary={}
    sp=out/'summary.json'
    if sp.exists():
        try: summary=json.loads(sp.read_text(encoding='utf-8'))
        except Exception: summary={}
    for p in files:
        dest=out/f'{p.stem}_annotated.mp4'
        if dest.exists() and dest.stat().st_size>100000 and p.name in summary:
            print('skip',p.name,flush=True); continue
        cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS) or 25
        w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur=n/max(fps,1)
        stride=max(1,round(fps/args.sample_fps)); out_fps=min(args.sample_fps,fps)
        writer=cv2.VideoWriter(str(dest),cv2.VideoWriter_fourcc(*'mp4v'),out_fps,(w,h))
        maxc={'fire':0.0,'smoke':0.0}; frames={'fire':0,'smoke':0}; first={'fire':None,'smoke':None}; fd=0
        idx=sampled=0
        while True:
            ok,frame=cap.read()
            if not ok: break
            if idx%stride:
                idx+=1; continue
            t=idx/fps
            r=model.predict(frame,imgsz=640,conf=args.conf,device='cpu',verbose=False)[0]
            dets=[]
            for b in r.boxes:
                lab=names.get(int(b.cls.item()),str(int(b.cls.item())))
                if 'fire' in lab: lab='fire'
                elif 'smoke' in lab: lab='smoke'
                else: continue
                c=float(b.conf.item()); box=[int(v) for v in b.xyxy[0].tolist()]
                maxc[lab]=max(maxc[lab],c)
                if c>=args.draw_conf:
                    frames[lab]+=1; fd+=1
                    if first[lab] is None: first[lab]=round(t,2)
                dets.append((lab,c,box))
            for lab,c,box in dets:
                alert=c>=(fire_thr if lab=='fire' else smoke_thr)
                color=(0,0,255) if lab=='fire' else (0,140,255)
                if not alert: color=(90,90,90)
                cv2.rectangle(frame,(box[0],box[1]),(box[2],box[3]),color,3 if alert else 1)
                cv2.putText(frame,f'{lab} {c:.2f}',(box[0],max(12,box[1]-4)),cv2.FONT_HERSHEY_SIMPLEX,0.55,color,2)
            cv2.putText(frame,f'{p.name}  t={t:.1f}s  fire={maxc["fire"]:.2f} smoke={maxc["smoke"]:.2f}',(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.55,(255,255,255),2)
            writer.write(frame); sampled+=1; idx+=1
        cap.release(); writer.release()
        alert=bool(maxc['fire']>=fire_thr or maxc['smoke']>=smoke_thr)
        summary[p.name]={'duration_s':round(dur,1),'sampled_frames':sampled,'annotated_video':str(dest),
                         'max_fire':round(maxc['fire'],4),'max_smoke':round(maxc['smoke'],4),
                         'frames_fire_ge_draw':frames['fire'],'frames_smoke_ge_draw':frames['smoke'],
                         'first_fire_s':first['fire'],'first_smoke_s':first['smoke'],
                         'p50_alert_fire_or_smoke':alert}
        print(p.name,json.dumps(summary[p.name]),flush=True)
        sp.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('\nwrote',out/'summary.json')

if __name__=='__main__': raise SystemExit(main())
