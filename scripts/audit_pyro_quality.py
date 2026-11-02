import json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT
from ultralytics import YOLO
from transformers import CLIPModel, CLIPProcessor

OUT=PROJECT_ROOT/'build/pyro_audit'
rows=[json.loads(l) for l in open(OUT/'manifest.jsonl',encoding='utf-8') if l.strip()]
# P50
model=YOLO(str(PROJECT_ROOT/'build/kaggle_weights/runs/P50/weights/best.onnx'),task='detect')
def iou(a,b):
    ix1,iy1=max(a[0],b[0]),max(a[1],b[1]); ix2,iy2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
    return inter/ua if ua>0 else 0
for r in rows:
    im=Image.open(PROJECT_ROOT/r['path']).convert('RGB')
    out=model.predict([str(PROJECT_ROOT/r['path'])],imgsz=640,conf=0.05,batch=1,device='cpu',verbose=False)[0]
    dets=[{'label':'fire' if int(b.cls.item())==0 else 'smoke','conf':float(b.conf.item()),'box':[float(v) for v in b.xyxy[0].tolist()]} for b in out.boxes]
    r['p50_dets']=dets
    r['max_fire']=max((d['conf'] for d in dets if d['label']=='fire'),default=0.0)
    r['max_smoke']=max((d['conf'] for d in dets if d['label']=='smoke'),default=0.0)
    gts=[a['box'] for a in r['annotations']]
    # missing box: annotation empty + strong det, or det strong but no overlap with any GT
    r['missing_box_suspect']=bool((not gts and max(r['max_fire'],r['max_smoke'])>=0.50) or (gts and any(d['conf']>=0.50 and max((iou(d['box'],g) for g in gts),default=0)<0.1 for d in dets)))
# CLIP
cm=CLIPModel.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K').eval()
cp=CLIPProcessor.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
prompts=['a photo of smoke','a photo of fog or mist','a photo of clouds','a photo of fire','a photo of a clear sky']
with torch.no_grad():
    t=cp(text=prompts,return_tensors='pt',padding=True)
    to=cm.text_model(input_ids=t['input_ids'],attention_mask=t.get('attention_mask'))
    tf=cm.text_projection(to.pooler_output); tf=tf/tf.norm(dim=-1,keepdim=True)
for s in range(0,len(rows),32):
    batch=rows[s:s+32]; imgs=[Image.open(PROJECT_ROOT/r['path']).convert('RGB') for r in batch]
    inp=cp(images=imgs,return_tensors='pt')
    with torch.no_grad():
        vo=cm.vision_model(pixel_values=inp['pixel_values']); f=cm.visual_projection(vo.pooler_output); f=f/f.norm(dim=-1,keepdim=True)
    sims=f@tf.T
    for r,sim in zip(batch,sims):
        d={p:float(v) for p,v in zip(prompts,sim.tolist())}
        r['clip']={k:round(v,4) for k,v in d.items()}
        smoke=d['a photo of smoke']; fog=max(d['a photo of fog or mist'],d['a photo of clouds'])
        r['fog_as_smoke_suspect']=bool(r['n_boxes']>0 and fog>smoke+0.01)
# sheets
def font(sz=10):
    try: return ImageFont.truetype('arial.ttf',sz)
    except Exception:
        try: return ImageFont.load_default(size=sz)
        except Exception: return ImageFont.load_default()
cols=10; tw,th=220,190; nrows=(len(rows)+cols-1)//cols
S=Image.new('RGB',(cols*tw,nrows*th),(10,10,12)); d0=ImageDraw.Draw(S)
for i,r in enumerate(rows):
    im=Image.open(PROJECT_ROOT/r['path']).convert('RGB'); im.thumbnail((tw-6,th-48))
    ox=(i%cols)*tw+(tw-im.width)//2; oy=(i//cols)*th+20; S.paste(im,(ox,oy)); d=ImageDraw.Draw(S)
    W,H=Image.open(PROJECT_ROOT/r['path']).size; sx,sy=im.width/W,im.height/H
    for a in r['annotations']:
        b=a['box']; d.rectangle([ox+b[0]*sx,oy+b[1]*sy,ox+b[2]*sx,oy+b[3]*sy],outline=(0,255,0),width=2)
    for det in r['p50_dets']:
        b=det['box']; d.rectangle([ox+b[0]*sx,oy+b[1]*sy,ox+b[2]*sx,oy+b[3]*sy],outline=(255,40,40),width=1)
    flags=[]
    if r.get('fog_as_smoke_suspect'): flags.append('FOG?')
    if r.get('missing_box_suspect'): flags.append('MISS?')
    if not r['n_boxes']: flags.append('empty')
    d.text(((i%cols)*tw+3,(i//cols)*th+2),f"{i:03d} {r['camera'][:12]} {' '.join(flags)}",fill=(255,255,0),font=font(8))
S.save(OUT/'audit_100.jpg',quality=85)
summary={'n':len(rows),'with_boxes':sum(1 for r in rows if r['n_boxes']),'empty':sum(1 for r in rows if not r['n_boxes']),
         'fog_as_smoke_suspects':sum(1 for r in rows if r.get('fog_as_smoke_suspect')),
         'missing_box_suspects':sum(1 for r in rows if r.get('missing_box_suspect')),
         'fog_examples':[{'i':r['idx'],'image':r['image_name'],'clip':r['clip']} for r in rows if r.get('fog_as_smoke_suspect')][:10],
         'missing_examples':[{'i':r['idx'],'image':r['image_name'],'max_fire':r['max_fire'],'max_smoke':r['max_smoke'],'n_boxes':r['n_boxes']} for r in rows if r.get('missing_box_suspect')][:10]}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (OUT/'audit.csv').open('w',newline='',encoding='utf-8') as f:
    import csv
    w=csv.writer(f); w.writerow(['idx','image_name','camera','n_boxes','max_fire','max_smoke','fog_as_smoke_suspect','missing_box_suspect','clip_smoke','clip_fog','clip_cloud','clip_fire'])
    for r in rows:
        w.writerow([r['idx'],r['image_name'],r['camera'],r['n_boxes'],round(r['max_fire'],3),round(r['max_smoke'],3),r['fog_as_smoke_suspect'],r['missing_box_suspect'],r['clip']['a photo of smoke'],r['clip']['a photo of fog or mist'],r['clip']['a photo of clouds'],r['clip']['a photo of fire']])
print(json.dumps(summary,indent=2))
