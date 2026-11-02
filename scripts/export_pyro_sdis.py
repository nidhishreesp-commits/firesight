from __future__ import annotations
import io, json, sys
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT
OUT=PROJECT_ROOT/'build/pyro_sdis_v2'; (OUT/'images').mkdir(parents=True,exist_ok=True); (OUT/'labels').mkdir(parents=True,exist_ok=True)
rows=[]; total=0
for p in sorted((PROJECT_ROOT/'data/raw/pyro-sdis/data').glob('*.parquet')):
    split='val' if p.name.startswith('val') else 'train'
    (OUT/'images'/split).mkdir(parents=True,exist_ok=True); (OUT/'labels'/split).mkdir(parents=True,exist_ok=True)
    tab=pq.read_table(p)
    for r in tab.to_pylist():
        total+=1
        img_obj=r['image']; raw=img_obj.get('bytes') if isinstance(img_obj,dict) else None
        if raw is None: continue
        name=r['image_name']; stem=Path(name).stem
        try:
            im=Image.open(io.BytesIO(raw)).convert('RGB')
        except Exception: continue
        dest=OUT/'images'/split/name; im.save(dest,'JPEG',quality=90)
        lines=[]
        for line in (r.get('annotations') or '').splitlines():
            parts=line.split()
            if len(parts)==5:
                cls=int(float(parts[0])); cx,cy,bw,bh=map(float,parts[1:])
                lines.append(f'0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}')
        (OUT/'labels'/split/(stem+'.txt')).write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
        rows.append({'id':f'pyro:{name}','path':str(dest.relative_to(PROJECT_ROOT)),'split':split,'image_name':name,
                     'partner':r.get('partner'),'camera':r.get('camera'),'date':r.get('date'),'n_boxes':len(lines)})
        if total%2000==0: print(total,flush=True)
(OUT/'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n',encoding='utf-8')
print(json.dumps({'total_rows':total,'exported':len(rows),'train':sum(1 for r in rows if r['split']=='train'),'val':sum(1 for r in rows if r['split']=='val'),'with_boxes':sum(1 for r in rows if r['n_boxes'])},indent=2))
