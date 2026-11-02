import json, sys
from pathlib import Path
from PIL import Image
import numpy as np, torch
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT
from transformers import CLIPModel, CLIPProcessor
OUT=PROJECT_ROOT/'build/pyro_audit'; rows=[json.loads(l) for l in open(OUT/'manifest.jsonl',encoding='utf-8') if l.strip()]
import csv as _csv
meta={int(r['idx']):r for r in _csv.DictReader(open(OUT/'audit.csv',encoding='utf-8'))}
for r in rows:
    m=meta.get(int(r['idx']),{})
    r['max_fire']=float(m.get('max_fire') or 0); r['max_smoke']=float(m.get('max_smoke') or 0); r['missing_box_suspect']=str(m.get('missing_box_suspect')).lower()=='true'
cm=CLIPModel.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K').eval(); cp=CLIPProcessor.from_pretrained('laion/CLIP-ViT-B-32-laion2B-s34B-b79K')
prompts=['a photo of smoke','a photo of fog or mist','a photo of clouds','a photo of fire','a photo of a clear sky']
with torch.no_grad():
    t=cp(text=prompts,return_tensors='pt',padding=True); to=cm.text_model(input_ids=t['input_ids'],attention_mask=t.get('attention_mask')); tf=cm.text_projection(to.pooler_output); tf=tf/tf.norm(dim=-1,keepdim=True)
def crop(im,b,s=2.5):
    W,H=im.size; cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*s,(b[3]-b[1])*s
    x1,y1=max(0,int(cx-w/2)),max(0,int(cy-h/2)); x2,y2=min(W,int(cx+w/2)),min(H,int(cy+h/2))
    return im.crop((x1,y1,max(x1+1,x2),max(y1+1,y2)))
for r in rows:
    im=Image.open(PROJECT_ROOT/r['path']).convert('RGB'); W,H=im.size; crops=[]; areas=[]
    for a in r['annotations']:
        b=a['box']; crops.append(crop(im,b)); areas.append(((b[2]-b[0])*(b[3]-b[1]))/(W*H))
    r['box_area_ratios']=areas; r['tiny_box']=bool(areas and max(areas)<0.005)
    if not crops: r['crop_clip']={}; r['fog_as_smoke_crop']=False; continue
    inp=cp(images=crops,return_tensors='pt')
    with torch.no_grad():
        vo=cm.vision_model(pixel_values=inp['pixel_values']); f=cm.visual_projection(vo.pooler_output); f=f/f.norm(dim=-1,keepdim=True)
    sims=f@tf.T
    # median over boxes
    med=sims.median(dim=0).values.tolist(); d={p:float(v) for p,v in zip(prompts,med)}
    r['crop_clip']={k:round(v,4) for k,v in d.items()}
    r['fog_as_smoke_crop']=bool(d['a photo of fog or mist']>d['a photo of smoke']+0.01 or d['a photo of clouds']>d['a photo of smoke']+0.01)
summary={'n':len(rows),'with_boxes':sum(1 for r in rows if r['n_boxes']),'empty':sum(1 for r in rows if not r['n_boxes']),
         'fog_as_smoke_crop_suspects':sum(1 for r in rows if r.get('fog_as_smoke_crop')),
         'tiny_box_images':sum(1 for r in rows if r.get('tiny_box')),
         'missing_box_suspects':sum(1 for r in rows if r.get('missing_box_suspect'))}
(OUT/'summary_crop.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
with (OUT/'audit_crop.csv').open('w',newline='',encoding='utf-8') as f:
    import csv
    w=csv.writer(f); w.writerow(['idx','image_name','camera','n_boxes','max_fire','max_smoke','tiny_box','fog_as_smoke_crop','missing_box_suspect','crop_smoke','crop_fog','crop_cloud'])
    for r in rows:
        c=r.get('crop_clip',{})
        w.writerow([r['idx'],r['image_name'],r['camera'],r['n_boxes'],round(r['max_fire'],3),round(r['max_smoke'],3),r.get('tiny_box'),r.get('fog_as_smoke_crop'),r.get('missing_box_suspect'),c.get('a photo of smoke'),c.get('a photo of fog or mist'),c.get('a photo of clouds')])
print(json.dumps(summary,indent=2))
print('fog examples',[(r['idx'],r['image_name'],r.get('crop_clip')) for r in rows if r.get('fog_as_smoke_crop')][:8])
