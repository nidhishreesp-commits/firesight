from __future__ import annotations
import json, hashlib, os, shutil, sys
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT

OUT=PROJECT_ROOT/'build/f1'; OUT.mkdir(parents=True,exist_ok=True)
PKG=PROJECT_ROOT/'build/kaggle_f1_pyro'; (PKG/'images').mkdir(parents=True,exist_ok=True); (PKG/'labels').mkdir(parents=True,exist_ok=True); (PKG/'lists').mkdir(parents=True,exist_ok=True)
EVENT_CAMERAS={'brison-110','courmettes-212','brison-200'}
def sha_file(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()
def phash_file(p):
    import imagehash
    from PIL import Image
    try: return str(imagehash.phash(Image.open(p).convert('RGB')))
    except Exception: return None
def read_jsonl(p): return [json.loads(l) for l in open(p,encoding='utf-8') if l.strip()]

# references
ref_ph=[]; ref_sha=set()
for name in ('val.jsonl','test.jsonl'):
    for r in read_jsonl(PROJECT_ROOT/'data/splits'/name):
        if r.get('phash'): ref_ph.append(int(r['phash'],16))
for r in json.loads((PROJECT_ROOT/'bench_R/index.json').read_text(encoding='utf-8')):
    if r.get('phash'): ref_ph.append(int(r['phash'],16))
    if r.get('sha256'): ref_sha.add(r['sha256'])
pool=read_jsonl(PROJECT_ROOT/'data/hard_negatives_v3/manifest.jsonl')
for r in pool:
    if r.get('status') in ('hard','easy','quarantine') and r.get('split')=='holdout':
        if r.get('phash'): ref_ph.append(int(r['phash'],16))
        if r.get('sha256'): ref_sha.add(r['sha256'])
# sample video frames (1 per 5 s)
video_ph=[]; video_sha=set()
import cv2
for p in sorted(Path(r'C:\Users\Nidhi\My projects\Sample_videos').glob('*')):
    if p.suffix.lower() not in ('.mp4','.mov','.avi','.mkv'): continue
    cap=cv2.VideoCapture(str(p)); fps=cap.get(cv2.CAP_PROP_FPS) or 25; step=max(1,round(fps*5)); idx=0
    while cap.isOpened():
        ok=cap.grab()
        if not ok: break
        if idx%step==0:
            ok,frame=cap.retrieve()
            if not ok: break
            tmp=PROJECT_ROOT/'build/f1/_vf.jpg'; cv2.imwrite(str(tmp),frame)
            h=phash_file(tmp); 
            if h: ref_ph.append(int(h,16)); video_ph.append(int(h,16))
            video_sha.add(sha_file(tmp))
        idx+=1
    cap.release()
ref_arr=np.array(ref_ph,dtype=np.uint64)
print('refs',len(ref_arr),'sha',len(ref_sha),'video frames',len(video_ph),flush=True)

# pyro candidates
rows=read_jsonl(PROJECT_ROOT/'build/pyro_sdis_v2/manifest.jsonl')
cand=[r for r in rows if r['split']=='train' and r['n_boxes'] and r['camera'] not in EVENT_CAMERAS]
print('pyro candidates',len(cand),flush=True)
kept=[]; rejected=Counter(); examples=[]
for r in cand:
    p=PROJECT_ROOT/r['path']; s=sha_file(p); h=phash_file(p)
    if s in ref_sha or s in video_sha:
        rejected['exact_sha']+=1; examples.append({'id':r['id'],'reason':'exact_sha'}); continue
    if h and len(ref_arr):
        d=np.bitwise_count(np.uint64(int(h,16))^ref_arr)
        if int(d.min())<=2:
            rejected['phash_le2']+=1; examples.append({'id':r['id'],'reason':'phash_le2','distance':int(d.min())}); continue
    r['sha256']=s; r['phash']=h; kept.append(r)
print('kept',len(kept),'rejected',dict(rejected),flush=True)
# package
for r in kept:
    src=PROJECT_ROOT/r['path']; name=Path(r['path']).name
    dst=PKG/'images'/name
    if not dst.exists():
        try: os.link(src,dst)
        except OSError: shutil.copy2(src,dst)
    (PKG/'labels'/(Path(name).stem+'.txt')).write_text('',encoding='utf-8')
(PKG/'lists/train.txt').write_text('\n'.join(str((PKG/'images'/Path(r['path']).name).relative_to(PKG)) for r in kept)+'\n',encoding='utf-8')
(PKG/'manifest.jsonl').write_text('\n'.join(json.dumps(r) for r in kept)+'\n',encoding='utf-8')
(PKG/'dataset-metadata.json').write_text(json.dumps({'title':'firesight F1 pyro positives (private)','id':'nidhishreesp/firesight-f1-pyro','licenses':[{'name':'other'}],'isPrivate':True,'description':'Pyronear Pyro-SDIS v2.0 positives, whole event cameras excluded (brison-110, courmettes-212, brison-200). Apache-2.0 / Licence Ouverte 2.0.'},indent=2),encoding='utf-8')

# base 50% positives
base_pkg=PROJECT_ROOT/'build/kaggle_dataset'
base_lines=[l.strip() for l in (base_pkg/'lists/train_50.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
base=[]
for l in base_lines:
    p=base_pkg/l
    lbl=base_pkg/'labels'/(p.stem+'.txt')
    if lbl.exists() and lbl.stat().st_size>0: base.append(p)
print('base positives',len(base),flush=True)
# negatives package
neg_pkg=PROJECT_ROOT/'build/kaggle_stage_a_negatives'
negs=[neg_pkg/l.strip() for l in (neg_pkg/'lists/e1_negatives.txt').read_text(encoding='utf-8').splitlines() if l.strip()]
print('negatives',len(negs),flush=True)
# local lists
(OUT/'lists').mkdir(parents=True,exist_ok=True)
(OUT/'lists/train.txt').write_text('\n'.join(str(p.resolve()) for p in base+[PROJECT_ROOT/r['path'] for r in kept]+negs)+'\n',encoding='utf-8')
# kaggle component lists
(PKG/'lists/train.txt').write_text('\n'.join(str((PKG/'images'/Path(r['path']).name).relative_to(PKG)) for r in kept)+'\n',encoding='utf-8')
(OUT/'lists/train_base.txt').write_text('\n'.join(str(p.relative_to(base_pkg)) for p in base)+'\n',encoding='utf-8')
(OUT/'lists/train_pyro.txt').write_text('\n'.join(str((PKG/'images'/Path(r['path']).name).relative_to(PKG)) for r in kept)+'\n',encoding='utf-8')
(OUT/'lists/train_neg.txt').write_text('\n'.join(str(p.relative_to(neg_pkg)) for p in negs)+'\n',encoding='utf-8')
# val lists
for name in ('val_early.txt','val_cal.txt'):
    (OUT/'lists'/name).write_text((base_pkg/'lists'/name).read_text(encoding='utf-8'),encoding='utf-8')
# event report
ev_rows=[r for r in rows if r['camera'] in EVENT_CAMERAS]
seq=Counter()
for r in ev_rows:
    if r['n_boxes']: seq[(r['camera'],r['date'][:10])]+=1
comp={'base_50_positives':len(base),'pyro_train_nonevent_positives':len(kept),'hard_negative_pool':len(negs),
      'total_images':len(base)+len(kept)+len(negs),'negative_share_added':round(len(negs)/(len(base)+len(kept)),4),
      'event_cameras':sorted(EVENT_CAMERAS),'event_images':len(ev_rows),'event_smoke_images':sum(1 for r in ev_rows if r['n_boxes']),
      'event_sequences':len(seq),'training_event_overlap':0,
      'excluded_pyro_reasons':dict(rejected),'excluded_pyro_examples':examples[:10]}
(OUT/'composition.json').write_text(json.dumps(comp,indent=2),encoding='utf-8')
(OUT/'leakage_report.json').write_text(json.dumps({'verdict':'PASS' if not rejected else 'PASS_AFTER_REMOVAL','removed':dict(rejected),'ref_phashes':len(ref_arr),'ref_sha':len(ref_sha),'pyro_kept':len(kept),'event_training_overlap':0},indent=2),encoding='utf-8')
print(json.dumps(comp,indent=2))
