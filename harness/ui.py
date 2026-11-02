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
:root{--paper:#FAF9F5;--card:#FFFFFF;--sand:#F0EEE6;--line:#E8E6DC;--ink:#141413;--mute:#73726C;--clay:#D97757;--clay-d:#C15F3C;
  --serif:"Copernicus","Tiempos Headline","Iowan Old Style","Charter","Georgia",serif;
  --sans:"Styrene B","Söhne",ui-sans-serif,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
button,input,textarea{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--clay);outline-offset:2px}
.wrap{max-width:720px;margin:0 auto;padding:64px 24px 80px}
h1{font:400 38px/1.15 var(--serif);letter-spacing:-.02em;text-align:center;margin:0 0 28px}
.box{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:16px 18px;box-shadow:0 2px 10px rgba(20,20,19,.04)}
.lab{font-size:13px;color:var(--mute);margin:12px 0 2px}.lab:first-child{margin-top:0}
input,textarea{width:100%;border:0;outline:0;background:transparent;resize:vertical;padding:2px 0}
.go{display:flex;justify-content:space-between;align-items:center;margin-top:12px;gap:12px}
.hint{font-size:12.5px;color:var(--mute)}
button{border:0;background:var(--clay);color:#fff;border-radius:10px;padding:9px 16px;cursor:pointer;font-weight:500;white-space:nowrap}
button:hover{background:var(--clay-d)}button:disabled{opacity:.5;cursor:default}
#out{margin-top:24px;display:grid;gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:18px 20px}
.who{font:500 17px/1.35 var(--serif);margin:0 0 12px}
.opt{display:grid;grid-template-columns:minmax(110px,34%) 1fr 44px;gap:12px;align-items:center;margin:7px 0}
.opt span{font-size:14px}.pct{text-align:right;font-variant-numeric:tabular-nums}
.track{height:12px;background:var(--sand);border-radius:6px;overflow:hidden}.fill{height:100%;background:var(--clay);border-radius:6px}
.err{color:#A63D2F;background:#F8E3DE;border-radius:12px;padding:10px 14px}
</style></head><body>
<div class="wrap">
  <h1>How would this group answer?</h1>
  <div class="box">
    <div class="lab">Question</div>
    <input id="question" value="Would you pay $9.99 a month for an ad-free version of your favorite social media app?">
    <div class="lab">Answers, one per line</div>
    <textarea id="options" rows="4">Definitely yes
Probably yes
Probably no
Definitely no</textarea>
    <div class="lab">Groups, one per line</div>
    <textarea id="groups" rows="2">The year is 2026. You are from the United States. You are in the 18-29 age group.
The year is 2026. You are from the United States. You are in the 65+ age group.</textarea>
    <div class="go"><span id="hint" class="hint"></span><button id="btn" onclick="ask()">Ask</button></div>
  </div>
  <div id="out"></div>
</div>
<script>
const el=(t,c,x)=>{const e=document.createElement(t);if(c)e.className=c;if(x!=null)e.textContent=x;return e};
const lines=id=>document.getElementById(id).value.split('\n').map(s=>s.trim()).filter(Boolean);
fetch('/api/info').then(r=>r.json()).then(i=>{if(i.partial)document.getElementById('hint').textContent='Preview: the fine-tuned model is not deployed yet.'}).catch(()=>{});
async function ask(){const btn=document.getElementById('btn'),out=document.getElementById('out');
 const body={question:document.getElementById('question').value.trim(),options:lines('options'),groups:lines('groups')};
 out.innerHTML='';if(!body.question||body.options.length<2||!body.groups.length){out.appendChild(el('div','err','Add a question, at least two answers and one group.'));return}
 btn.disabled=true;btn.textContent='Asking…';
 try{const d=await (await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})).json();
  if(d.error){out.appendChild(el('div','err',d.error));return}
  d.results.forEach(g=>{const c=el('div','card');c.appendChild(el('div','who',g.group));
   d.labels.forEach((l,i)=>{const p=g.prediction[l],r=el('div','opt');r.appendChild(el('span',null,d.options[i]));
    const t=el('div','track'),f=el('div','fill');f.style.width=(100*p)+'%';t.appendChild(f);r.appendChild(t);r.appendChild(el('span','pct',Math.round(100*p)+'%'));c.appendChild(r)});
   out.appendChild(c)})}
 catch(e){out.appendChild(el('div','err','Could not reach the local server.'))}
 finally{btn.disabled=false;btn.textContent='Ask'}}
if(location.hash==='#run')ask();
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
