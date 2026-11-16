import csv, json, random, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT
OUT=PROJECT_ROOT/'docs/pyro_cleanup'; OUT.mkdir(parents=True,exist_ok=True)
meta={r['idx']:r for r in (json.loads(l) for l in open('build/pyro_audit/manifest.jsonl',encoding='utf-8') if l.strip())}
audit={r['idx']:r for r in csv.DictReader(open('build/pyro_audit/audit_crop.csv',encoding='utf-8'))}
sus=[]; non=[]
for idx,r in audit.items():
    if int(float(r['n_boxes']))==0: continue
    (sus if r['fog_as_smoke_crop'].lower()=='true' else non).append(idx)
rng=random.Random(20261102); rng.shuffle(non); non=non[:40]
print('suspects',len(sus),'non-suspects',len(non))
def font(sz=10):
    try: return ImageFont.truetype('arial.ttf',sz)
    except Exception:
        try: return ImageFont.load_default(size=sz)
        except Exception: return ImageFont.load_default()
def crop_for(idx):
    m=meta[int(idx)]
    im=Image.open(PROJECT_ROOT/m['path']).convert('RGB'); W,H=im.size
    b=max((a['box'] for a in m['annotations']), key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
    cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; w,h=(b[2]-b[0])*3.0,(b[3]-b[1])*3.0
    x1,y1=max(0,int(cx-w/2)),max(0,int(cy-h/2)); x2,y2=min(W,int(cx+w/2)),min(H,int(cy+h/2))
    return im.crop((x1,y1,max(x1+8,x2),max(y1+8,y2))), b
items=[('suspect',i) for i in sus]+[('non',i) for i in non]
cols=9; tw,th=230,220; rows=(len(items)+cols-1)//cols
sheet=Image.new('RGB',(cols*tw,rows*th),(12,12,14)); d0=ImageDraw.Draw(sheet)
index=[]
for n,(grp,idx) in enumerate(items):
    m=meta[int(idx)]; a=audit[str(idx)] if str(idx) in audit else audit.get(int(idx),{})
    try: im,b=crop_for(idx)
    except Exception: continue
    scale=min((tw-8)/max(im.width,1),(th-52)/max(im.height,1))
    im=im.resize((max(1,int(im.width*scale)),max(1,int(im.height*scale))),Image.LANCZOS)
    ox=(n%cols)*tw+(tw-im.width)//2; oy=(n//cols)*th+24; sheet.paste(im,(ox,oy)); d=ImageDraw.Draw(sheet)
    label=f'{n+1:02d} {grp} {m["camera"][:14]}'
    d.text(((n%cols)*tw+3,(n//cols)*th+3),label,fill=(255,230,0) if grp=='suspect' else (180,220,255),font=font(9))
    d.text(((n%cols)*tw+3,(n//cols)*th+th-20),f'{m["image_name"][:32]}',fill=(170,170,170),font=font(7))
    index.append({'tile':n+1,'group':grp,'idx':int(idx),'image_name':m['image_name'],'camera':m['camera'],'n_boxes':m['n_boxes'],
                  'box_area_ratio':a.get('crop_smoke') and '' or '', 'crop_smoke':a.get('crop_smoke'),'crop_fog':a.get('crop_fog'),'crop_cloud':a.get('crop_cloud'),
                  'real_smoke':''})
sheet.save(OUT/'contact_sheet.jpg',quality=88)
with (OUT/'index.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(index[0])); w.writeheader(); w.writerows(index)
with (OUT/'marks_template.csv').open('w',newline='',encoding='utf-8') as f:
    w=csv.writer(f); w.writerow(['tile','group','image_name','camera','real_smoke_yes_no','notes'])
    for r in index: w.writerow([r['tile'],r['group'],r['image_name'],r['camera'],'',''])
print('wrote',OUT/'contact_sheet.jpg',len(index))
