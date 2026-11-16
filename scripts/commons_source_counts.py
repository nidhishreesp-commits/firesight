import requests,re,json,time,sys
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
s=requests.Session(); s.headers['User-Agent']='FireSightSourceAudit/0.1 (research; local)'
terms={
 'industrial_fire':['factory fire','industrial fire','warehouse fire','refinery fire','power plant fire','chemical plant fire'],
 'indoor_fire':['house fire interior','kitchen fire','electrical fire','building fire interior','room fire'],
 'vehicle_ship':['vehicle fire','car fire','ship fire','truck fire'],
 'controlled':['candle flame','campfire','bonfire','fireplace fire','controlled burn','burn barrel'],
 'positive_smoke':['chimney smoke','factory smoke','smoke plume','wildfire smoke'],
 'hard_neg':['steam','welding sparks','street lamp','neon sign','computer monitor fire','vehicle lights','safety vest','hot metal','furnace','lamp'],
}
out={}
for group,qs in terms.items():
    out[group]={}
    for q in qs:
        try:
            r=s.get('https://commons.wikimedia.org/w/api.php',params={'action':'query','format':'json','list':'search','srsearch':f'{q} filetype:bitmap','srnamespace':'6','srlimit':'1'},timeout=30)
            j=r.json(); hits=j.get('query',{}).get('searchinfo',{}).get('totalhits','?')
            out[group][q]=int(hits) if str(hits).isdigit() else hits
        except Exception as e: out[group][q]=None
        time.sleep(0.25)
print(json.dumps(out,indent=2))
Path=__import__('pathlib').Path
Path('build/video_eval/commons_counts.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
