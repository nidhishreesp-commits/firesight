"""Local review tool for the fire image bench.

  python review_fire_bench.py        # then open http://127.0.0.1:5055

Serve images from C:\\Users\\Nidhi\\My projects\\fire_bench_raw\\{fire,nonfire},
and let the reviewer Approve (Y) / Reject (N) each one. Decisions persist in
review_decisions.json. "Export" copies approved images into
fire_bench_clean\\{fire,nonfire}.
"""
import json
import os
import shutil

from flask import Flask, jsonify, request, send_from_directory

RAW = r"C:\Users\Nidhi\My projects\fire_bench_raw"
CLEAN = r"C:\Users\Nidhi\My projects\fire_bench_clean"
DECISIONS = r"C:\Users\Nidhi\My projects\fire_bench_review.json"
PORT = 5055

app = Flask(__name__)

def list_items():
    items = []
    for cls in ("fire", "nonfire"):
        d = os.path.join(RAW, cls)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if fn.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                items.append({"cls": cls, "fn": fn})
    return items

def load_dec():
    if os.path.exists(DECISIONS):
        try:
            return json.load(open(DECISIONS))
        except Exception:
            return {}
    return {}

def save_dec(d):
    json.dump(d, open(DECISIONS, "w"), indent=1)

@app.route("/")
def index():
    return HTML

@app.get("/items")
def items():
    dec = load_dec()
    out = []
    for it in list_items():
        key = f"{it['cls']}/{it['fn']}"
        out.append({**it, "decision": dec.get(key)})
    return jsonify(out)

@app.get("/img/<cls>/<path:fn>")
def img(cls, fn):
    if cls not in ("fire", "nonfire"):
        return ("bad class", 400)
    return send_from_directory(os.path.join(RAW, cls), fn)

@app.post("/decide")
def decide():
    body = request.get_json(force=True)
    key = f"{body['cls']}/{body['fn']}"
    dec = load_dec()
    if body["decision"] is None:
        dec.pop(key, None)
    else:
        dec[key] = body["decision"]
    save_dec(dec)
    return jsonify({"ok": True})

@app.post("/export")
def export():
    dec = load_dec()
    for cls in ("fire", "nonfire"):
        out = os.path.join(CLEAN, cls)
        os.makedirs(out, exist_ok=True)
    n = {"fire": 0, "nonfire": 0}
    for it in list_items():
        key = f"{it['cls']}/{it['fn']}"
        if dec.get(key) == "y":
            src = os.path.join(RAW, it["cls"], it["fn"])
            dst = os.path.join(CLEAN, it["cls"], it["fn"])
            shutil.copy2(src, dst)
            n[it["cls"]] += 1
    return jsonify({"approved": n})


HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Fire bench review</title>
<style>
 body{font-family:system-ui,Segoe UI,Arial;background:#111;color:#eee;margin:0}
 #bar{padding:10px 14px;background:#1b1b1b;display:flex;gap:18px;align-items:center;font-size:14px}
 #bar b{color:#ffd400}
 #wrap{display:flex;justify-content:center;padding:14px}
 img{max-height:78vh;max-width:92vw;border:3px solid #333;border-radius:6px;background:#000}
 .btns{display:flex;gap:12px;justify-content:center;padding:8px}
 button{font-size:18px;padding:10px 26px;border-radius:8px;border:0;cursor:pointer;font-weight:600}
 #yes{background:#15803d;color:#fff}#no{background:#b91c1c;color:#fff}#undo{background:#374151;color:#fff}
 #info{text-align:center;padding:6px;color:#aaa;font-size:13px}
 .pill{padding:2px 8px;border-radius:999px;background:#333}
</style></head><body>
<div id="bar"><span>Approve / Reject — keys <b>Y</b>=yes <b>N</b>=no <b>U</b>=undo</span>
<span class="pill" id="prog"></span><span class="pill" id="cls"></span>
<button id="exp" style="margin-left:auto;background:#2563eb;color:#fff;font-size:14px;padding:6px 14px">Export approved</button></div>
<div id="wrap"><img id="im" src=""></div>
<div class="btns"><button id="yes">Approve (Y)</button><button id="no">Reject (N)</button><button id="undo">Undo (U)</button></div>
<div id="info"></div>
<script>
let items=[],i=0,hist=[];
async function load(){items=await (await fetch('/items')).json();
  i=items.findIndex(x=>!x.decision); if(i<0)i=0; render();}
function render(){
  const it=items[i]; if(!it){document.getElementById('info').textContent='DONE';return;}
  document.getElementById('im').src=`/img/${it.cls}/${encodeURIComponent(it.fn)}`;
  document.getElementById('cls').textContent=it.cls.toUpperCase();
  const y=items.filter(x=>x.decision==='y').length, n=items.filter(x=>x.decision==='n').length;
  const yf=items.filter(x=>x.decision==='y'&&x.cls==='fire').length, yn=items.filter(x=>x.decision==='y'&&x.cls==='nonfire').length;
  document.getElementById('prog').textContent=`${i+1}/${items.length}  approved fire=${yf} nonfire=${yn}`;
  document.getElementById('info').textContent='decided: '+y+' yes / '+n+' no';
}
async function decide(d){const it=items[i]; if(!it)return;
  it.decision=d; hist.push(i);
  await fetch('/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cls:it.cls,fn:it.fn,decision:d})});
  let j=items.findIndex((x,k)=>k>i&&!x.decision); if(j<0)j=items.findIndex(x=>!x.decision); i=(j<0?i+1:j); render();}
function undo(){const k=hist.pop(); if(k===undefined)return; const it=items[k]; it.decision=null;
  fetch('/decide',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cls:it.cls,fn:it.fn,decision:null})});
  i=k; render();}
document.getElementById('yes').onclick=()=>decide('y');
document.getElementById('no').onclick=()=>decide('n');
document.getElementById('undo').onclick=undo;
document.getElementById('exp').onclick=async()=>{const r=await (await fetch('/export',{method:'POST'})).json(); alert('Approved: fire='+r.approved.fire+' nonfire='+r.approved.nonfire);};
addEventListener('keydown',e=>{const k=e.key.toLowerCase(); if(k==='y')decide('y'); else if(k==='n')decide('n'); else if(k==='u')undo();});
load();
</script></body></html>"""

if __name__ == "__main__":
    print(f"Review tool: http://127.0.0.1:{PORT}  (raw={RAW})")
    app.run(host="127.0.0.1", port=PORT, debug=False)
