import json, cv2
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.insert(0,'src'); from firesight.config import PROJECT_ROOT
rows=[json.loads(l) for l in open('data/video_eval/public/manifest.jsonl',encoding='utf-8') if l.strip()]
out=PROJECT_ROOT/'build/video_eval/contact_sheets'; out.mkdir(parents=True,exist_ok=True)
def font(sz=9):
    try: return ImageFont.truetype('arial.ttf',sz)
    except Exception:
        try: return ImageFont.load_default(size=sz)
        except Exception: return ImageFont.load_default()
for cat in ('fire','smoke','cloud','nonfire'):
    sub=[r for r in rows if r['category']==cat]
    cols=5; tw,th=300,220; tiles=[]
    for r in sub:
        cap=cv2.VideoCapture(str(PROJECT_ROOT/r['path'])); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for frac in (0.15,0.45,0.75):
            cap.set(cv2.CAP_PROP_POS_FRAMES,int(max(0,n*frac-1))); ok,fr=cap.read()
            if not ok: continue
            im=Image.fromarray(cv2.cvtColor(fr,cv2.COLOR_BGR2RGB)); im.thumbnail((tw-8,th-40)); tiles.append((r,im.copy()))
        cap.release()
    if not tiles: continue
    rowsn=(len(tiles)+cols-1)//cols; S=Image.new('RGB',(cols*tw,rowsn*th),(15,15,18)); d=ImageDraw.Draw(S)
    for i,(r,im) in enumerate(tiles):
        x=(i%cols)*tw+4; y=(i//cols)*th+18; S.paste(im,(x,y)); d.text((x,y-15),f"{r['id'][8:32]} {r['title'][:34]}",fill=(230,230,230),font=font(8))
    S.save(out/f'{cat}.jpg',quality=82); print(cat,len(sub),'clips',len(tiles),'tiles')
