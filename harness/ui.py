#!/usr/bin/env python3
"""Crowdcast local web UI: ask a group a multiple-choice question and see how it would split.
Standard library only. Serves http://127.0.0.1:8050

  ../.venv/bin/python harness/ui.py            # then open http://127.0.0.1:8050
  ../.venv/bin/python harness/ui.py --port 9000

Runs live through harness/cli.py predict() (same code as the CLI and the scored system). Until the fine-tune is
deployed (eval/models.json -> fw-ft), it runs in preview mode (DeepSeek + survey memory) and says so on screen.
"""
import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "harness"))
from tasks import simbench as T  # noqa: E402
import crowdcast as C  # noqa: E402
import cli  # noqa: E402

STATE = {"system": None, "system_error": None}
LOCK = threading.Lock()


def get_system():
    with LOCK:
        if STATE["system"] is None and STATE["system_error"] is None:
            try:
                STATE["system"] = cli.load_system(allow_partial=True)
            except Exception as e:
                STATE["system_error"] = str(e)
        return STATE["system"], STATE["system_error"]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else (json.dumps(body) if ctype == "application/json" else body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path)
        try:
            if u.path == "/":
                return self.send(200, PAGE, "text/html")
            if u.path == "/api/info":
                sysm, err = get_system()
                return self.send(200, {"live_ready": sysm is not None, "live_error": err,
                                       "partial": bool(sysm and sysm["partial"]),
                                       "weights": sysm["weights"] if sysm else None,
                                       "o_model": sysm["o_cfg"]["model"] if sysm else None})
            return self.send(404, {"error": "not found"})
        except Exception as e:
            return self.send(500, {"error": str(e)})

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            groups = [g.strip() for g in body.get("groups", []) if g.strip()][:6]
            options = [o.strip() for o in body.get("options", []) if o.strip()]
            question = (body.get("question") or "").strip()
            if not groups or not question:
                return self.send(400, {"error": "need at least one group and a question"})
            sysm, err = get_system()
            if sysm is None:
                return self.send(400, {"error": err})
            return self.send(200, cli.predict(sysm, groups, question, options))
        except cli.SetupError as e:
            return self.send(400, {"error": str(e)})
        except Exception as e:
            return self.send(500, {"error": str(e)})


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>Crowdcast</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --paper:#FAF9F5;--card:#FFFFFF;--sand:#F0EEE6;--line:#E8E6DC;--ink:#141413;--mute:#73726C;--clay:#D97757;--clay-d:#C15F3C;
  --serif:"Copernicus","Tiempos Headline","Tiempos Text","Iowan Old Style","Charter","Georgia",serif;
  --sans:"Styrene B","Söhne",ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
button,input,textarea,select{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--clay);outline-offset:2px}
.top{display:flex;align-items:center;gap:10px;padding:18px 28px}
.mark{display:grid;grid-template-columns:repeat(3,5px);gap:2px}.mark i{width:5px;height:5px;border-radius:50%;background:var(--clay)}
.mark i:nth-child(2n){opacity:.55}
.brand{font:500 17px var(--serif);letter-spacing:-.01em}
.wrap{max-width:960px;margin:0 auto;padding:8px 24px 80px}
.hero{text-align:center;margin:28px 0 22px}
.hero h1{font:400 40px/1.15 var(--serif);letter-spacing:-.02em;margin:0 0 8px}
.hero p{color:var(--mute);margin:0}
.seg{display:flex;justify-content:center;margin:0 0 22px}
.seg div{display:inline-flex;background:var(--sand);border-radius:12px;padding:4px;gap:2px}
.seg button{border:0;background:none;padding:7px 14px;border-radius:9px;cursor:pointer;color:var(--mute)}
.seg button.on{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(20,20,19,.08)}
.composer{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:14px 16px;box-shadow:0 2px 10px rgba(20,20,19,.04)}
.composer input,.composer textarea{width:100%;border:0;outline:0;background:transparent;resize:vertical;padding:4px 2px}
.composer textarea{min-height:56px}
.composer .lab{font-size:13px;color:var(--mute);margin:10px 0 2px}
.composer .bar{display:flex;align-items:center;gap:10px;justify-content:space-between;margin-top:10px}
.cbar{display:flex;align-items:center;gap:8px}
select{border:1px solid var(--line);background:var(--paper);border-radius:9px;padding:6px 8px}
.send{border:0;background:var(--clay);color:#fff;border-radius:10px;padding:8px 14px;cursor:pointer;font-weight:500}
.send:hover{background:var(--clay-d)}.send:disabled{opacity:.5;cursor:default}
.chips{display:flex;flex-wrap:wrap;gap:8px;justify-content:center;margin:14px 0 18px}
.chip{border:1px solid var(--line);background:var(--card);border-radius:999px;padding:7px 13px;cursor:pointer;color:var(--ink)}
.chip:hover{background:var(--sand)}
.list{background:var(--card);border:1px solid var(--line);border-radius:14px;max-height:260px;overflow:auto;margin:-6px 0 18px}
.list button{display:block;width:100%;text-align:left;border:0;border-bottom:1px solid var(--line);background:none;padding:10px 14px;cursor:pointer}
.list button:hover{background:var(--paper)}.list button:last-child{border-bottom:0}
.list small{display:block;color:var(--mute);font-size:12.5px;margin-top:2px}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:16px;margin-top:18px}
.big .stack{height:34px;border-radius:9px}.big .stack div{font-size:13px}
details{margin-top:16px;border-top:1px solid var(--line);padding-top:12px}summary{cursor:pointer;color:var(--mute);font-size:13.5px}
details .rows{margin-top:12px}.addrow{display:flex;gap:8px;flex-wrap:wrap}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px}
.src{font-size:12.5px;color:var(--mute)}
.who{font:500 19px/1.3 var(--serif);letter-spacing:-.01em;margin:4px 0 6px}
.ask{margin:0 0 14px;color:#3D3D3A}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:0 0 14px;font-size:13px;color:#3D3D3A}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:10px;height:10px;border-radius:3px;display:inline-block}
.rows{display:grid;grid-template-columns:128px 1fr 52px;gap:9px 12px;align-items:center}
.rows .h{font-size:12px;color:var(--mute)}.rows .h.r{text-align:right}
.nm{font-size:13.5px;color:#3D3D3A}.nm.human{font-weight:600;color:var(--ink)}
.nm.ours{font-weight:600;color:var(--clay-d);display:flex;align-items:center;gap:6px}.nm.ours:before{content:"";width:7px;height:7px;border-radius:50%;background:var(--clay)}
.sc{text-align:right;font-variant-numeric:tabular-nums;font-size:13.5px}.sc.ours{color:var(--clay-d);font-weight:600}
.stack{display:flex;height:24px;border-radius:7px;overflow:hidden;background:var(--sand)}
.stack div{height:100%;font-size:11.5px;font-weight:500;display:flex;align-items:center;justify-content:center;overflow:hidden;white-space:nowrap}
.ourstack{box-shadow:0 0 0 2px var(--card),0 0 0 3.5px var(--clay)}
.crowds{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:16px;margin-top:18px;padding-top:16px;border-top:1px solid var(--line)}
.crowds h4{margin:0 0 6px;font:500 14px var(--serif)}.crowds h4.ours{color:var(--clay-d)}
.dots{display:grid;grid-template-columns:repeat(20,8px);gap:3px}.dots i{width:8px;height:8px;border-radius:50%}
.flag{display:inline-block;margin-top:16px;padding:5px 11px;border-radius:999px;font-size:13px}
.flag.askp{background:#F6E6D2;color:#7A4A1E}.flag.ok{background:#E7ECDF;color:#46602F}
.note{color:var(--mute);font-size:13px;text-align:center;max-width:640px;margin:22px auto 0}
.banner{background:#F6E6D2;color:#6B4219;border-radius:14px;padding:12px 16px;margin:0 0 14px;font-size:14px}
.ev{font-size:13px;color:var(--mute);margin-top:12px}
.meta{font-size:13px;color:var(--mute);margin-top:10px}
.err{color:#A63D2F;background:#F8E3DE;border-radius:12px;padding:10px 14px}
.empty{color:var(--mute);text-align:center;padding:30px 0}
@media (max-width:560px){.hero h1{font-size:30px}.panels{grid-template-columns:1fr}.rows{grid-template-columns:96px 1fr 44px}}
@media (prefers-reduced-motion:no-preference){.card{animation:in .25s ease-out}@keyframes in{from{opacity:0;transform:translateY(4px)}to{opacity:1}}}
</style></head><body>
<div class="top"><div class="mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><span class="brand">Crowdcast</span></div>
<div class="wrap">
  <div class="hero"><h1>How would this group answer?</h1><p>Describe a group of people and ask a multiple-choice question. Crowdcast predicts how they would split, in about a second.</p></div>
  <div id="bnr"></div>
  <div class="composer">
    <div class="lab" style="margin-top:0">Question</div>
    <input id="question" placeholder="Would you pay $9.99 a month for an ad-free version of your favorite social media app?">
    <div class="lab">Answer options, one per line</div>
    <textarea id="options" rows="4" placeholder="Definitely yes&#10;Probably yes&#10;Probably no&#10;Definitely no"></textarea>
    <div class="lab">Who to ask, one group per line</div>
    <textarea id="groups" rows="2" placeholder="The year is 2026. You are from the United States. You are in the 18-29 age group."></textarea>
    <div class="bar"><span id="st" class="src"></span><button id="go" class="send" onclick="predict()">Ask the crowd</button></div>
  </div>
  <div class="chips" id="examples"></div>
  <div id="p2" class="panels"></div>
  <p class="note">These are predictions, not polls. Use them to decide which questions are worth asking real people.</p>
</div>
<script>
const PAL=[["#5B7FA6","#fff"],["#D4A27F","#3A2A1C"],["#788C5D","#fff"],["#B0624B","#fff"],["#8E7CB5","#fff"],["#4E8C86","#fff"],["#C9A85C","#3A2E12"],["#87867F","#fff"]];
const col=i=>PAL[i%PAL.length][0],fg=i=>PAL[i%PAL.length][1];
const el=(t,c,txt)=>{const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e};
const EXAMPLES=[
 {label:"Ad-free subscription, young vs old",q:"Would you pay $9.99 a month for an ad-free version of your favorite social media app?",o:["Definitely yes","Probably yes","Probably no","Definitely no"],g:["The year is 2026. You are from the United States. You are in the 18-29 age group.","The year is 2026. You are from the United States. You are in the 65+ age group."]},
 {label:"Four-day work week, two countries",q:"Would you support a law that makes the standard work week four days instead of five?",o:["Strongly support","Somewhat support","Somewhat oppose","Strongly oppose"],g:["The year is 2026. You are from Germany.","The year is 2026. You are from Japan."]},
 {label:"Trust in AI answers, by education",q:"How much do you trust answers from AI chatbots?",o:["A great deal","Somewhat","Not much","Not at all"],g:["The year is 2026. You are from the United States. You have a college degree.","The year is 2026. You are from the United States. You did not finish high school."]}];
function fill(x){document.getElementById('question').value=x.q;document.getElementById('options').value=x.o.join('\n');document.getElementById('groups').value=x.g.join('\n')}
(function(){const E=document.getElementById('examples');EXAMPLES.forEach(x=>{const b=el('button','chip',x.label);b.onclick=()=>{fill(x);predict()};E.appendChild(b)});fill(EXAMPLES[0])})();
function stack(dist,ours){const s=el('div','stack'+(ours?' ourstack':''));dist.forEach((p,i)=>{const d=el('div',null,p>=.09?Math.round(100*p)+'%':'');d.style.width=(100*p)+'%';d.style.background=col(i);d.style.color=fg(i);d.title=Math.round(100*p)+'%';s.appendChild(d)});return s}
function dots(dist){const g=el('div','dots');const counts=dist.map(p=>Math.floor(100*p));const rem=100-counts.reduce((a,b)=>a+b,0);
 const order=dist.map((p,i)=>[100*p-Math.floor(100*p),i]).sort((a,b)=>b[0]-a[0]);for(let k=0;k<rem;k++)counts[order[k%order.length][1]]++;
 counts.forEach((c,i)=>{for(let k=0;k<c;k++){const d=el('i');d.style.background=col(i);g.appendChild(d)}});return g}
function legend(options){const L=el('div','legend');options.forEach((o,i)=>{const s=el('span');const w=el('i','sw');w.style.background=col(i);s.appendChild(w);s.appendChild(document.createTextNode(o));L.appendChild(s)});return L}
function rows(list){const R=el('div','rows');list.forEach(x=>{R.appendChild(el('div','nm',x.name));R.appendChild(stack(x.dist,false));R.appendChild(el('div','sc',''))});return R}
let INFO=null;async function info(){try{const r=await fetch('/api/info');INFO=await r.json()}catch(e){return}const B=document.getElementById('bnr');B.innerHTML='';
 if(!INFO.live_ready){B.appendChild(el('div','banner','Crowdcast can\'t answer yet: '+INFO.live_error));document.getElementById('go').disabled=true;return}
 if(INFO.partial)B.appendChild(el('div','banner','Preview mode: our fine-tuned model isn\'t deployed yet, so answers come from DeepSeek V4.1 Flash plus the survey memory. The full version adds the fine-tune.'))}
info();
async function predict(){const btn=document.getElementById('go'),st=document.getElementById('st'),P=document.getElementById('p2');
 const body={groups:document.getElementById('groups').value.split('\n'),question:document.getElementById('question').value,options:document.getElementById('options').value.split('\n')};
 if(!body.question.trim()){st.textContent='Write a question first.';return}
 if(body.options.filter(x=>x.trim()).length<2){st.textContent='Add at least two answer options.';return}
 if(!body.groups.filter(x=>x.trim()).length){st.textContent='Describe at least one group.';return}
 btn.disabled=true;st.textContent='Asking the crowd…';P.innerHTML='';
 try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();
  if(d.error){P.appendChild(el('div','err',d.error));st.textContent='';return}
  st.textContent='';const names={F:'Our fine-tune',O:'DeepSeek V4.1 Flash',M:'Similar past surveys'};
  d.results.forEach(g=>{const c=el('div','card');c.appendChild(el('div','who',g.group));c.appendChild(legend(d.options));
   const dist=d.labels.map(l=>g.prediction[l]);const big=el('div','big');big.appendChild(stack(dist,false));c.appendChild(big);
   const cr=el('div','crowds');cr.appendChild(crowd('If you asked 100 of them',dist));c.appendChild(cr);
   if(g.ask_real_people!=null)c.appendChild(el('span','flag '+(g.ask_real_people?'askp':'ok'),g.ask_real_people?'Worth asking real people: the models disagree':'The models agree on this one'));
   c.appendChild(el('div','meta','Answered in '+g.latency_s.toFixed(1)+' s'+(g.cost_usd!=null?' for $'+g.cost_usd.toFixed(5):'')));
   for(const [k,e] of Object.entries(g.member_errors||{}))c.appendChild(el('div','err',names[k]+' didn\'t answer, so it counts as an even split: '+e));
   const det=el('details');det.appendChild(el('summary',null,'How this was computed'));
   const list=[];for(const k of ['F','O','M']){if(g.members[k])list.push({name:names[k],dist:d.labels.map(l=>g.members[k][l])})}
   det.appendChild(rows(list));det.appendChild(el('div','ev','Blend weights: '+Object.entries(d.meta.weights_used).filter(([k,v])=>v>0).map(([k,v])=>(names[k]||k)+' '+Math.round(100*v)+'%').join(', ')));
   if(g.evidence&&g.evidence.length)det.appendChild(el('div','ev','Closest past survey question: '+g.evidence[0].question));
   c.appendChild(det);P.appendChild(c)})}
 catch(e){P.appendChild(el('div','err','Could not reach the local server: '+e));st.textContent=''}finally{btn.disabled=false}}
function crowd(title,dist){const c=el('div');c.appendChild(el('h4',null,title));c.appendChild(dots(dist));return c}
if(location.hash==='#run')predict();
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    print("loading the survey memory and model settings ...")
    get_system()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Crowdcast UI on http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
