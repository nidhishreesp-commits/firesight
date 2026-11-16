import json
from pathlib import Path
rows=[json.loads(l) for l in open('data/video_eval/public/manifest.jsonl',encoding='utf-8')]
# keep -> (label, category)
keep={0:('fire','fire'),3:('fire','fire'),4:('fire','fire'),6:('fire','fire'),7:('fire','fire'),8:('fire','fire'),
      9:('fire','fire'),10:('fire','fire'),11:('fire','fire'),12:('fire','fire'),13:('fire','fire'),14:('fire','fire'),
      17:('fire','fire'),18:('fire','fire'),19:('fire','fire'),
      24:('smoke','smoke'),26:('smoke','smoke'),30:('smoke','smoke'),31:('smoke','smoke'),32:('smoke','smoke'),
      33:('nonfire','cloud'),34:('nonfire','cloud'),35:('nonfire','cloud'),36:('nonfire','cloud'),37:('nonfire','cloud'),
      38:('nonfire','cloud'),40:('nonfire','cloud'),41:('nonfire','cloud'),43:('nonfire','cloud'),44:('nonfire','cloud'),
      48:('nonfire','nonfire'),49:('nonfire','nonfire'),50:('nonfire','nonfire'),51:('nonfire','nonfire'),52:('nonfire','nonfire'),
      53:('nonfire','nonfire'),54:('nonfire','nonfire'),55:('nonfire','nonfire'),56:('nonfire','nonfire'),57:('nonfire','nonfire'),
      58:('nonfire','nonfire'),59:('nonfire','nonfire')}
cur=[]
for i,pair in keep.items():
    label,cat=pair; rec=dict(rows[i]); rec['label']=label; rec['category']=cat; rec['curated']=True
    rec['max_seconds']=120 if label=='nonfire' and float(rec.get('duration_s') or 0)>120 else None
    cur.append(rec)
Path('data/video_eval/public/curated_manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in cur)+'\n',encoding='utf-8')
from collections import Counter
print('curated',len(cur),Counter(r['label'] for r in cur),Counter(r['category'] for r in cur))
