from __future__ import annotations
import json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0,'src'); sys.path.insert(0,'scripts')
from firesight.config import PROJECT_ROOT
from run_benchR import load_benchR

WILSON=lambda k,n,z=1.96: (None if n==0 else (lambda p,d,c,h:[round(max(0,c-h),4),round(min(1,c+h),4)])(k/n,1+z*z/n,(k/n+z*z/(2*n))/(1+z*z/n),z*math.sqrt((k/n*(1-k/n)+z*z/(4*n))/n)/(1+z*z/n)))
THR={'fire':0.13,'smoke':0.39}
VER_V={'1.5%':0.011936064809560776,'2.0%':0.004953139461576939}

def font(sz=11):
    try: return ImageFont.truetype('arial.ttf',sz)
    except Exception:
        try: return ImageFont.load_default(size=sz)
        except Exception: return ImageFont.load_default()

def draw(rec, preds, gt, out, title, tile=(360,270)):
    try: im=Image.open(rec.path).convert('RGB')
    except Exception: return
    im.thumbnail((tile[0]-8,tile[1]-54))
    t=Image.new('RGB',tile,(18,18,20)); d=ImageDraw.Draw(t)
    ox,oy=(tile[0]-im.width)//2,26; t.paste(im,(ox,oy))
    sx,sy=im.width/max(rec.width,1),im.height/max(rec.height,1)
    for g in gt:
        b=g.box.to_list(); d.rectangle([ox+b[0]*sx,oy+b[1]*sy,ox+b[2]*sx,oy+b[3]*sy],outline=(0,255,0),width=2)
    for p in preds:
        b=p['box']; c=p['confidence']; lab=p['label']
        d.rectangle([ox+b[0]*sx,oy+b[1]*sy,ox+b[2]*sx,oy+b[3]*sy],outline=(255,40,40),width=2)
        d.text((ox+b[0]*sx+1,oy+b[1]*sy+1),f'{lab} {c:.2f}',fill=(255,230,0),font=font(9))
    d.text((4,4),title[:70],fill=(235,235,235),font=font(10))
    d.text((4,tile[1]-22),f'{rec.id.split(":")[-1][:44]} | {rec.meta.get("family")} | {rec.meta.get("flame_types")}',fill=(175,175,175),font=font(8))
    out.parent.mkdir(parents=True,exist_ok=True); t.save(out,quality=88)

def main():
    recs,member=load_benchR(); dev=[r for r in recs if member[r.id]=='dev']
    preds={r['id']:r.get('detections',[]) for r in (json.loads(l) for l in open('bench_results/predictions/P50_benchR.jsonl',encoding='utf-8') if l.strip())}
    out=PROJECT_ROOT/'build/miss_analysis/DEV'; (out/'contact_sheets').mkdir(parents=True,exist_ok=True)
    rows=[]; missed=[]
    for r in dev:
        pos=bool(r.meta.get('has_fire') or r.meta.get('has_smoke'))
        ds=preds.get(r.id,[]); mf=max((d['confidence'] for d in ds if d['label']=='fire'),default=0.0); ms=max((d['confidence'] for d in ds if d['label']=='smoke'),default=0.0)
        alarm=mf>=THR['fire'] or ms>=THR['smoke']
        if not pos: continue
        try:
            im=Image.open(r.path).convert('RGB'); arr=np.asarray(im.resize((64,64)),dtype=np.float32); bright=float(arr.mean())
        except Exception: bright=None
        areas=[]
        for g in r.labels:
            b=g.box.to_list(); areas.append(((b[2]-b[0])*(b[3]-b[1]))/(max(r.width,1)*max(r.height,1)))
        max_area=max(areas,default=0.0)
        txt=(r.id+' '+Path(r.path).stem).lower()
        aerial=any(k in txt for k in ['aerial','drone','wildfire','wild fire','air tanker','airtanker','helicopter','from above','overlook','hillside','mountain','forest fire','wildland','landscape','high angle'])
        row={'id':r.id,'path':r.path,'family':r.meta.get('family'),'category':r.category,'flame_types':r.meta.get('flame_types'),
             'has_fire':r.meta.get('has_fire'),'has_smoke':r.meta.get('has_smoke'),'width':r.width,'height':r.height,
             'max_fire':round(mf,4),'max_smoke':round(ms,4),'max_conf':round(max(mf,ms),4),'alarm_baseline':alarm,
             'max_gt_area_ratio':round(max_area,5),'n_gt':len(r.labels),'brightness':None if bright is None else round(bright,1),
             'aerial_high_mounted':aerial}
        if not alarm:
            tags=[]
            if row['has_smoke'] and ms<THR['smoke']: tags.append('thin_smoke')
            if max_area<0.02: tags.append('small_distant')
            if bright is not None and bright<60: tags.append('low_light')
            if 'indoor' in (r.meta.get('family') or '') or 'indoor' in r.category: tags.append('indoor')
            if len(r.labels)>=3 and max_area<0.05: tags.append('occlusion')
            if r.meta.get('family') in ('controlled_fire','spark_screen'): tags.append('unusual_scene')
            if aerial: tags.append('aerial_high_mounted')
            if not tags: tags=['other']
            row['cause_tags']=tags; row['primary_cause']=tags[0]
            missed.append((r,row,ds))
        rows.append(row)
    # threshold recovery
    pos=[r for r in rows]; neg=[r for r in dev if not (r.meta.get('has_fire') or r.meta.get('has_smoke'))]
    def rec_fpr(thr):
        tp=sum(1 for r in rows if r['max_conf']>=thr); fp=sum(1 for r in dev if not (r.meta.get('has_fire') or r.meta.get('has_smoke')) and r.id in preds and max((d['confidence'] for d in preds[r.id]),default=0)>=thr)
        return {'recall':tp/len(rows),'recall_ci':WILSON(tp,len(rows)),'fpr':fp/len(neg),'fpr_ci':WILSON(fp,len(neg)),'tp':tp,'fp':fp}
    recov={str(t):rec_fpr(t) for t in (0.20,0.15,0.10)}
    # verifier v4 cascade on DEV
    import joblib
    b=joblib.load('build/verifier/model_v4.joblib'); model=b['model']
    d=np.load('build/verifier/features_dev.npz',allow_pickle=True); X=d['X']; meta=[json.loads(m) for m in d['meta']]; sc=model.predict_proba(X)[:,1]
    by=defaultdict(list)
    for m,s in zip(meta,sc):
        if float(m['conf'])>=0.10: by[m['id']].append(float(s))
    casc={}
    for name,v in VER_V.items():
        tp=sum(1 for r in rows if max(by.get(r['id'],[-1]))>=v)
        fp=sum(1 for r in neg if max(by.get(r.id,[-1]))>=v)
        recovered=sum(1 for r,row,ds in missed if max(by.get(r.id,[-1]))>=v)
        casc[name]={'recall':tp/len(rows),'recall_ci':WILSON(tp,len(rows)),'fpr':fp/len(neg),'fpr_ci':WILSON(fp,len(neg)),'tp':tp,'fp':fp,'baseline_missed_recovered':recovered,'baseline_missed':len(missed)}
    # contact sheets
    for sheet,tag in [('missed_uncontrolled_fire',lambda r,row:r.meta.get('family')=='uncontrolled_fire' or 'uncontrolled' in (r.meta.get('flame_types') or [])),
                      ('missed_smoke',lambda r,row:bool(r.meta.get('has_smoke'))),
                      ('missed_all',lambda r,row:True)]:
        items=[(r,row,ds) for r,row,ds in missed if tag(r,row)]
        if not items: continue
        cols=6; tw,th=360,270; nrows=(len(items)+cols-1)//cols
        S=Image.new('RGB',(cols*tw,nrows*th),(10,10,12)); dd=ImageDraw.Draw(S)
        for i,(r,row,ds) in enumerate(items):
            tile=out/'contact_sheets'/f'_tile_{i}.jpg'; draw(r,ds,r.labels,tile,'',tile=(tw,th))
            try: S.paste(Image.open(tile),( (i%cols)*tw,(i//cols)*th ))
            except Exception: pass
            tile.unlink(missing_ok=True)
        S.save(out/'contact_sheets'/f'{sheet}.jpg',quality=86)
    # per primary cause
    bycause=defaultdict(list)
    for r,row,ds in missed: bycause[row['primary_cause']].append((r,row,ds))
    for cause,items in bycause.items():
        cols=5; tw,th=360,270; nrows=(len(items)+cols-1)//cols
        S=Image.new('RGB',(cols*tw,nrows*th),(10,10,12))
        for i,(r,row,ds) in enumerate(items):
            tile=out/'contact_sheets'/f'_tile_{cause}_{i}.jpg'; draw(r,ds,r.labels,tile,'',tile=(tw,th))
            try: S.paste(Image.open(tile),((i%cols)*tw,(i//cols)*th))
            except Exception: pass
            tile.unlink(missing_ok=True)
        S.save(out/'contact_sheets'/f'cause_{cause}.jpg',quality=86)
    with (out/'missed_positives.csv').open('w',newline='',encoding='utf-8') as f:
        import csv; fields=[]
        for rr in rows:
            for kk in rr:
                if kk not in fields: fields.append(kk)
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary={'dev_images':len(dev),'dev_positives':len(rows),'baseline_thresholds':THR,'baseline_missed':len(missed),
             'baseline_missed_fire_only':sum(1 for r,row,ds in missed if row['has_fire'] and not row['has_smoke']),
             'baseline_missed_smoke_only':sum(1 for r,row,ds in missed if row['has_smoke'] and not row['has_fire']),
             'baseline_missed_both':sum(1 for r,row,ds in missed if row['has_fire'] and row['has_smoke']),
             'missed_by_flame_type':dict(Counter(t for r,row,ds in missed for t in (r.meta.get('flame_types') or ['unknown']))),
             'missed_by_family':dict(Counter(row['family'] for r,row,ds in missed)),
             'missed_by_primary_cause':dict(Counter(row['primary_cause'] for r,row,ds in missed)),
             'recovery_global_threshold':recov,'verifier_v4_cascade':casc,'aerial_high_mounted_dev':sum(1 for r in rows if r['aerial_high_mounted'])}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__': raise SystemExit(main())
