from __future__ import annotations
import json, random, io, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pyarrow.parquet as pq
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT

OUT=PROJECT_ROOT/'build/pyro_audit'; (OUT/'images').mkdir(parents=True,exist_ok=True)
files=sorted((PROJECT_ROOT/'data/raw/pyro-sdis/data').glob('*.parquet'))
rng=random.Random(20261102)
rows=[]
for fi,p in enumerate(files):
    tab=pq.read_table(p)
    n=tab.num_rows
    take=rng.sample(range(n), min(25,n))
    for idx in take:
        r={k:tab.column(k)[idx].as_py() for k in tab.column_names}
        rows.append((p.name,idx,r))
print('sampled',len(rows),flush=True)
# decode & save
records=[]
for i,(fname,idx,r) in enumerate(rows):
    img_obj=r['image']; raw=img_obj.get('bytes') if isinstance(img_obj,dict) else None
    if raw is None:
        print('no bytes',i,fname,idx); continue
    im=Image.open(io.BytesIO(raw)).convert('RGB')
    dest=OUT/'images'/f'audit_{i:03d}.jpg'; im.save(dest,'JPEG',quality=90)
    anns=[]
    for line in (r.get('annotations') or '').splitlines():
        parts=line.split()
        if len(parts)==5:
            cls=int(float(parts[0])); cx,cy,bw,bh=map(float,parts[1:])
            W,H=im.size; x1=(cx-bw/2)*W; y1=(cy-bh/2)*H; x2=(cx+bw/2)*W; y2=(cy+bh/2)*H
            anns.append({'cls':cls,'box':[x1,y1,x2,y2]})
    records.append({'idx':i,'file':fname,'row':idx,'image_name':r.get('image_name'),'partner':r.get('partner'),
                    'camera':r.get('camera'),'date':r.get('date'),'path':str(dest.relative_to(PROJECT_ROOT)),
                    'annotations':anns,'n_boxes':len(anns)})
(OUT/'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in records)+'\n',encoding='utf-8')
print('decoded',len(records),'with boxes',sum(1 for r in records if r['n_boxes']),'empty',sum(1 for r in records if not r['n_boxes']))
