import requests,re,json,sys,time
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
sys.path.insert(0,'scripts')
from collect_stage_a_images import allowed_license,clean
s=requests.Session(); s.headers['User-Agent']='FireSightSourceAudit/0.1 (research; local)'
terms=['factory fire','industrial fire','warehouse fire','refinery fire','power plant fire','chemical plant fire',
'house fire interior','kitchen fire','electrical fire','building fire interior','candle flame','campfire','bonfire','fireplace fire','controlled burn','burn barrel','chimney smoke','factory smoke','steam','welding sparks','street lamp','neon sign','safety vest','hot metal','furnace']
out={}
for q in terms:
    try:
        r=s.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','generator':'search','gsrsearch':f'{q} filetype:bitmap','gsrnamespace':'6','gsrlimit':'50','prop':'imageinfo','iiprop':'extmetadata'},timeout=40)
        j=r.json(); pages=(j.get('query') or {}).get('pages',{}); total=len(pages); allowed=0; lic={}
        for p in pages.values():
            vi=(p.get('imageinfo') or [{}])[0]; md=vi.get('extmetadata',{}); L=clean(md.get('LicenseShortName',{}).get('value',''))
            lic[L]=lic.get(L,0)+1
            if allowed_license(L): allowed+=1
        out[q]={'sampled':total,'allowed':allowed,'acceptance':round(allowed/max(total,1),3),'licences':dict(sorted(lic.items(),key=lambda kv:-kv[1])[:5])}
    except Exception as e: out[q]={'error':str(e)}
    time.sleep(0.3)
print(json.dumps(out,indent=2))
open('build/video_eval/commons_licence_sample.json','w',encoding='utf-8').write(json.dumps(out,indent=2))
