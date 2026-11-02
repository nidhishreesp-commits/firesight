from __future__ import annotations
import argparse, json, re, time, sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import requests
sys.path.insert(0,'src')
from firesight.config import PROJECT_ROOT
API='https://commons.wikimedia.org/w/api.php'
QUERIES={
 'fire':['fire','wildfire','campfire','bonfire','house fire','vehicle fire','fireplace'],
 'smoke':['smoke','chimney smoke','welding smoke','factory smoke','smoke plume'],
 'cloud':['clouds timelapse','fog','haze','mist'],
 'nonfire':['traffic','street scene','construction site','crowd','factory interior','kitchen cooking'],
}
ALLOWED=re.compile(r'(cc0|public domain|cc by|cc-by|pd-)',re.I)
BLOCK=re.compile(r'(nc|nd|non-commercial|noderiv)',re.I)
def clean(s): return re.sub(r'<[^>]+>','',s or '').strip()
def lic_ok(s):
    s=clean(s).lower(); return bool(ALLOWED.search(s)) and not bool(BLOCK.search(s))
def pick_deriv(vi):
    ds=vi.get('derivatives') or []
    for key in ('360p.mpeg4.mov','480p.mpeg4.mov','240p.mpeg4.mov','360p.vp9.webm','480p.vp9.webm'):
        for d in ds:
            if d.get('transcodekey')==key: return d.get('src'),key
    return vi.get('url'),'original'
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--out',default='data/video_eval/public'); ap.add_argument('--per-query',type=int,default=3); ap.add_argument('--cap',type=int,default=60)
    args=ap.parse_args(); out=PROJECT_ROOT/args.out; out.mkdir(parents=True,exist_ok=True)
    s=requests.Session(); s.headers['User-Agent']='FireSightVideoEval/0.1 (research; local)'
    rows=[]; seen=set(); total_bytes=0
    for cat,queries in QUERIES.items():
        for q in queries:
            if len(rows)>=args.cap: break
            params={'action':'query','format':'json','generator':'search','gsrsearch':f'{q} filetype:video','gsrnamespace':'6','gsrlimit':'20','prop':'videoinfo','viprop':'url|extmetadata|size|derivatives'}
            try:
                r=s.get(API,params=params,timeout=40); r.raise_for_status(); j=r.json()
            except Exception as e: print('API fail',q,e,flush=True); continue
            got=0
            for p in (j.get('query') or {}).get('pages',{}).values():
                if got>=args.per_query or len(rows)>=args.cap: break
                title=p.get('title','')
                if title in seen: continue
                vi=(p.get('videoinfo') or [{}])[0]; md=vi.get('extmetadata',{})
                lic=clean(md.get('LicenseShortName',{}).get('value',''))
                if not lic_ok(lic): continue
                url,key=pick_deriv(vi)
                if not url: continue
                ext='.mov' if url.lower().split('?')[0].endswith('.mov') else ('.webm' if '.webm' in url.lower() else '.mp4')
                digest=re.sub(r'[^A-Za-z0-9]+','_',title)[:60].strip('_')
                dest=out/(digest+ext)
                if not dest.exists():
                    ok=False
                    for attempt in range(6):
                        try:
                            rr=s.get(url,timeout=120)
                            if rr.status_code==429:
                                time.sleep(20+attempt*10); continue
                            rr.raise_for_status()
                            if len(rr.content)<10000: break
                            dest.write_bytes(rr.content); total_bytes+=len(rr.content); ok=True; break
                        except Exception as e:
                            print('DL retry',repr(title)[:80],e,flush=True); time.sleep(5+attempt*5)
                    if not ok: continue
                seen.add(title); got+=1
                rows.append({'id':f'commons:{digest[:40]}','path':str(dest.relative_to(PROJECT_ROOT)),'title':title,'category':cat,'query':q,
                             'label':'fire' if cat=='fire' else 'smoke' if cat=='smoke' else 'nonfire',
                             'lookalike':cat=='cloud','licence':lic,'page_url':vi.get('descriptionurl',''),'source_url':url,
                             'duration_s':vi.get('duration'),'size_bytes':dest.stat().st_size,'derivative':key})
                time.sleep(2.5)
            print(cat,q,'->',len(rows),flush=True)
    (out/'manifest.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n',encoding='utf-8')
    print(json.dumps({'clips':len(rows),'MB':round(total_bytes/1e6,1),'by_category':{c:sum(1 for r in rows if r['category']==c) for c in QUERIES}},indent=2))
if __name__=='__main__': raise SystemExit(main())
