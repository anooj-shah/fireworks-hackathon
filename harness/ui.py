#!/usr/bin/env python3
"""Crowdcast local web UI. Standard library only; serves http://127.0.0.1:8050

  ../.venv/bin/python harness/ui.py            # then open http://127.0.0.1:8050
  ../.venv/bin/python harness/ui.py --port 9000

Tab 1, "Held-out questions": REPLAYS the scored runs on the locked SimBench test set (no API calls).
  Shows real people vs Crowdcast vs Claude Opus 5.5 vs the fine-tune alone vs the untrained base, with each
  system's SimBench score on that row. This is exactly the evaluated system (crowdcast:test-v1).
Tab 2, "Ask a new question": LIVE, through harness/cli.py predict() (same code as the CLI). Until the fine-tune is
  deployed (eval/models.json -> fw-ft), it runs in partial mode (DeepSeek + survey memory) and says so on screen.
"""
import argparse
import json
import random
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "harness"))
from tasks import simbench as T  # noqa: E402
import crowdcast as C  # noqa: E402
import cli  # noqa: E402

REPLAY_RUNS = [  # (key, display name, run spec): the committed test runs
    ("crowdcast", "Crowdcast (ours)", "crowdcast:test-v1"),
    ("opus", "Claude Opus 5.5", "claude-opus-5-5:test"),
    ("finetune", "Fine-tune alone", "ft-qwen:q36-3k-1e4-test"),
    ("base", "Untrained Qwen", "qwen-base:test-base"),
]
OPT_RE = re.compile(r"^\(([A-Z])\):\s*(.*)$")


def load_jsonl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def split_question(item):
    q = item["question"]
    stem, _, opts = q.partition("\n\nOptions:\n")
    names = {}
    for line in opts.splitlines():
        m = OPT_RE.match(line.strip())
        if m:
            names[m.group(1)] = m.group(2)
    return stem, [names.get(l, l) for l in item["labels"]]


class Replay:
    def __init__(self):
        self.items = {it["id"]: it for it in load_jsonl(ROOT / "data" / "test.jsonl")}
        self.runs = {}
        for key, _, spec in REPLAY_RUNS:
            row = C.find_summary_row(spec)
            self.runs[key] = {r["id"]: r for r in load_jsonl(ROOT / row["results_file"])}
        self.ids = sorted(self.items)
        self.index = [(i, (self.items[i]["persona"] + " " + self.items[i]["question"]).lower()) for i in self.ids]
        try:
            self.demo = json.loads((ROOT / "results" / "simbench" / "demo_draw.json").read_text())["chosen"]
        except Exception:
            self.demo = None

    def panel(self, iid):
        it = self.items[iid]
        stem, options = split_question(it)
        systems = []
        for key, name, spec in REPLAY_RUNS:
            r = self.runs[key].get(iid)
            if r is None:
                continue
            d = T.parse_dist(it, r.get("output")) if r.get("ok", True) else None
            d = d or C.uniform(it["labels"])
            systems.append({"key": key, "name": name, "run": spec, "dist": [d[l] for l in it["labels"]],
                            "score": r.get("simbench", 100 * r.get("score", 0.0))})
        cc = self.runs["crowdcast"].get(iid, {})
        return {"id": iid, "subset": it["subset"], "dataset": it["dataset"], "persona": it["persona"], "stem": stem,
                "labels": it["labels"], "options": options, "human": [it["human"][l] for l in it["labels"]],
                "group_size": it.get("group_size"), "systems": systems, "ask_real_people": cc.get("ask_real_people")}

    def search(self, q, subset, limit=60):
        words = [w for w in q.lower().split() if w]
        out = []
        for iid, text in self.index:
            if subset and not iid.startswith(subset + "-"):
                continue
            if all(w in text for w in words):
                it = self.items[iid]
                out.append({"id": iid, "dataset": it["dataset"], "persona": it["persona"][:90],
                            "stem": split_question(it)[0][:140]})
                if len(out) >= limit:
                    break
        return out


STATE = {"replay": None, "system": None, "system_error": None}
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
        qs = {k: v[0] for k, v in parse_qs(u.query).items()}
        rp = STATE["replay"]
        try:
            if u.path == "/":
                return self.send(200, PAGE, "text/html")
            if u.path == "/api/info":
                sysm, err = get_system()
                return self.send(200, {"live_ready": sysm is not None, "live_error": err,
                                       "partial": bool(sysm and sysm["partial"]),
                                       "weights": sysm["weights"] if sysm else None,
                                       "o_model": sysm["o_cfg"]["model"] if sysm else None,
                                       "demo": bool(rp.demo), "n_test": len(rp.ids)})
            if u.path == "/api/search":
                return self.send(200, rp.search(qs.get("q", ""), qs.get("subset", "")))
            if u.path == "/api/replay":
                ids = [i for i in qs.get("ids", "").split(",") if i in rp.items][:6]
                return self.send(200, [rp.panel(i) for i in ids])
            if u.path == "/api/random":
                sub = qs.get("subset", "")
                pool = [i for i in rp.ids if not sub or i.startswith(sub + "-")]
                return self.send(200, [rp.panel(random.choice(pool))])
            if u.path == "/api/demo":
                return self.send(200, [rp.panel(i) for i in (rp.demo or {}).get("ids", []) if i in rp.items])
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
:root{--ink:#16181d;--mute:#6b7280;--line:#e5e7eb;--bg:#fafaf9;--card:#fff;--acc:#2d5bff}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,sans-serif;color:var(--ink);background:var(--bg)}
header{padding:18px 28px;border-bottom:1px solid var(--line);background:var(--card);display:flex;align-items:baseline;gap:16px}
header h1{font-size:20px;margin:0;letter-spacing:-.01em}header .sub{color:var(--mute)}
nav{display:flex;gap:4px;padding:0 28px;background:var(--card);border-bottom:1px solid var(--line)}
nav button{border:0;background:none;padding:12px 14px;font:inherit;color:var(--mute);border-bottom:2px solid transparent;cursor:pointer}
nav button.on{color:var(--ink);border-color:var(--acc);font-weight:600}
main{padding:20px 28px;max-width:1280px}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-bottom:12px}
input,textarea,select{font:inherit;border:1px solid var(--line);border-radius:8px;padding:8px 10px;background:#fff}
textarea{width:100%;resize:vertical}
.btn{border:1px solid var(--line);background:#fff;border-radius:8px;padding:8px 12px;cursor:pointer;font:inherit}
.btn.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
.list{max-height:220px;overflow:auto;border:1px solid var(--line);border-radius:8px;background:#fff;margin-bottom:14px}
.list div{padding:7px 10px;border-bottom:1px solid var(--line);cursor:pointer}.list div:hover{background:#f3f4f6}
.list small{color:var(--mute)}
.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px}
.persona{font-weight:600}.stem{margin:6px 0 10px}.meta{color:var(--mute);font-size:12px}
.legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin:8px 0 12px;font-size:12px}
.legend span{display:inline-flex;align-items:center;gap:6px}.sw{width:10px;height:10px;border-radius:2px;display:inline-block}
.sys{display:grid;grid-template-columns:140px 1fr 64px;gap:10px;align-items:center;margin:6px 0}
.sys .nm{font-size:13px}.sys .sc{text-align:right;font-variant-numeric:tabular-nums;font-size:13px}
.stack{display:flex;height:22px;border-radius:5px;overflow:hidden;background:#f3f4f6}
.stack div{height:100%;color:#fff;font-size:11px;display:flex;align-items:center;justify-content:center;overflow:hidden;white-space:nowrap}
.human .nm{font-weight:700}.ours .nm{font-weight:700;color:var(--acc)}
.dots{display:grid;grid-template-columns:repeat(20,10px);gap:3px;margin-top:4px}.dots i{width:10px;height:10px;border-radius:50%}
.crowds{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin-top:12px}
.crowds h4{margin:0 0 2px;font-size:12px;font-weight:600}
.flag{display:inline-block;margin-top:10px;padding:3px 8px;border-radius:999px;font-size:12px}
.flag.ask{background:#fff4e5;color:#9a5b00}.flag.ok{background:#e8f6ee;color:#1b6b3a}
.banner{background:#fff4e5;border:1px solid #f5d39a;color:#7a4b00;border-radius:8px;padding:10px 12px;margin-bottom:12px}
.note{color:var(--mute);font-size:12px;margin-top:8px}
.ev{font-size:12px;color:var(--mute);margin-top:8px}
.err{color:#b42318}
</style></head><body>
<header><h1>Crowdcast</h1><span class="sub">a focus group in one second &middot; open models on Fireworks</span></header>
<nav><button id="t1" class="on" onclick="tab(1)">Held-out questions (scored)</button><button id="t2" onclick="tab(2)">Ask a new question (live)</button></nav>
<main>
<section id="s1">
  <div class="row">
    <button class="btn pri" onclick="demo()">Rule-drawn demo question</button>
    <button class="btn" onclick="rand('grouped')">Random demographic question</button>
    <button class="btn" onclick="rand('pop')">Random population question</button>
    <input id="q" placeholder="search the locked test set (e.g. adopt, trust, Kenya)" size="40" onkeydown="if(event.key==='Enter')search()">
    <select id="sub"><option value="">all</option><option value="pop">population</option><option value="grouped">demographic</option></select>
    <button class="btn" onclick="search()">Search</button>
  </div>
  <div id="list" class="list" style="display:none"></div>
  <div id="p1" class="panels"></div>
  <p class="note">Replay of the committed scored runs on the locked SimBench test set. No API calls. Score = official SimBench score on that row (100 = exactly the real split, 0 = no better than an even split). One row can't show a trend: use the proof card for claims (<code>results/simbench/prove__crowdcast__test-v1__test.md</code>).</p>
</section>
<section id="s2" style="display:none">
  <div id="bnr"></div>
  <label>Groups (one per line)</label>
  <textarea id="groups" rows="3">The year is 2026. You are from the United States. You are in the 18-29 age group.
The year is 2026. You are from the United States. You are in the 65+ age group.</textarea>
  <div class="row" style="margin-top:10px"><input id="question" size="90" value="Would you pay $9.99 a month for an ad-free version of your favorite social media app?"></div>
  <label>Options (one per line)</label>
  <textarea id="options" rows="4">Definitely yes
Probably yes
Probably no
Definitely no</textarea>
  <div class="row" style="margin-top:10px"><button id="go" class="btn pri" onclick="predict()">Run Crowdcast</button><span id="st" class="meta"></span></div>
  <div id="p2" class="panels"></div>
</section>
</main>
<script>
const COLORS=["#2d5bff","#f59e0b","#10b981","#ef4444","#8b5cf6","#06b6d4","#ec4899","#84cc16","#64748b","#a16207"];
const col=i=>COLORS[i%COLORS.length];
const el=(t,c,txt)=>{const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e};
function tab(n){for(const k of [1,2]){document.getElementById('s'+k).style.display=k===n?'':'none';document.getElementById('t'+k).className=k===n?'on':''}if(n===2)info()}
function stack(dist){const s=el('div','stack');dist.forEach((p,i)=>{const d=el('div',null,p>=.08?Math.round(100*p)+'%':'');d.style.width=(100*p)+'%';d.style.background=col(i);d.title=Math.round(100*p)+'%';s.appendChild(d)});return s}
function dots(dist){const g=el('div','dots');let counts=dist.map(p=>Math.floor(100*p)),rem=100-counts.reduce((a,b)=>a+b,0);
 const order=dist.map((p,i)=>[100*p-Math.floor(100*p),i]).sort((a,b)=>b[0]-a[0]);for(let k=0;k<rem;k++)counts[order[k%order.length][1]]++;
 counts.forEach((c,i)=>{for(let k=0;k<c;k++){const d=el('i');d.style.background=col(i);g.appendChild(d)}});return g}
function legend(labels,options){const L=el('div','legend');labels.forEach((l,i)=>{const s=el('span');const w=el('i','sw');w.style.background=col(i);s.appendChild(w);s.appendChild(document.createTextNode('('+l+') '+options[i]));L.appendChild(s)});return L}
function sysRow(name,dist,score,cls){const r=el('div','sys '+(cls||''));r.appendChild(el('div','nm',name));r.appendChild(stack(dist));r.appendChild(el('div','sc',score==null?'':score.toFixed(1)));return r}
function crowd(title,dist){const c=el('div');c.appendChild(el('h4',null,title));c.appendChild(dots(dist));return c}
function replayCard(p){const c=el('div','card');
 c.appendChild(el('div','meta',p.dataset+' · '+p.id+(p.group_size?' · '+p.group_size+' real respondents':'')));
 c.appendChild(el('div','persona',p.persona));c.appendChild(el('div','stem',p.stem));c.appendChild(legend(p.labels,p.options));
 const h=el('div','sys');h.appendChild(el('div','meta',''));h.appendChild(el('div','meta',''));h.appendChild(el('div','meta','score'));c.appendChild(h);
 c.appendChild(sysRow('Real people',p.human,null,'human'));
 p.systems.forEach(s=>c.appendChild(sysRow(s.name,s.dist,s.score,s.key==='crowdcast'?'ours':'')));
 const cr=el('div','crowds');cr.appendChild(crowd('100 real people',p.human));
 const ours=p.systems.find(s=>s.key==='crowdcast'),op=p.systems.find(s=>s.key==='opus');
 if(ours)cr.appendChild(crowd('Crowdcast',ours.dist));if(op)cr.appendChild(crowd('Claude Opus 5.5',op.dist));c.appendChild(cr);
 if(p.ask_real_people!=null){const f=el('span','flag '+(p.ask_real_people?'ask':'ok'),p.ask_real_people?'Ask real people: our two models disagree':'Our two models agree');c.appendChild(f)}
 return c}
async function show(url){const r=await fetch(url);const d=await r.json();const P=document.getElementById('p1');P.innerHTML='';if(d.error){P.appendChild(el('div','err',d.error));return}d.forEach(p=>P.appendChild(replayCard(p)))}
function demo(){document.getElementById('list').style.display='none';show('/api/demo')}
function rand(s){document.getElementById('list').style.display='none';show('/api/random?subset='+s)}
async function search(){const q=document.getElementById('q').value,s=document.getElementById('sub').value;const r=await fetch('/api/search?q='+encodeURIComponent(q)+'&subset='+s);const d=await r.json();
 const L=document.getElementById('list');L.innerHTML='';L.style.display='';if(!d.length){L.appendChild(el('div','meta','no matches'));return}
 d.forEach(x=>{const row=el('div');row.appendChild(el('div',null,x.stem));row.appendChild(el('small',null,x.dataset+' · '+x.persona));row.onclick=()=>show('/api/replay?ids='+x.id);L.appendChild(row)})}
let INFO=null;async function info(){if(INFO)return;const r=await fetch('/api/info');INFO=await r.json();const B=document.getElementById('bnr');B.innerHTML='';
 if(!INFO.live_ready){B.appendChild(el('div','banner','Live mode unavailable: '+INFO.live_error));document.getElementById('go').disabled=true;return}
 if(INFO.partial)B.appendChild(el('div','banner','Partial mode: the fine-tune is not deployed yet, so this runs DeepSeek V4.1 Flash + survey memory only ('+Object.entries(INFO.weights).map(([k,v])=>k+' '+v.toFixed(2)).join(', ')+'). This is NOT the system scored on the test set. Deploy the fine-tune and set fw-ft in eval/models.json for the full harness.'))}
async function predict(){const btn=document.getElementById('go'),st=document.getElementById('st'),P=document.getElementById('p2');btn.disabled=true;st.textContent='running…';P.innerHTML='';
 const body={groups:document.getElementById('groups').value.split('\n'),question:document.getElementById('question').value,options:document.getElementById('options').value.split('\n')};
 try{const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const d=await r.json();
  if(d.error){P.appendChild(el('div','err',d.error));return}
  const opts=d.options;st.textContent=(d.meta.partial_run?'partial mode · ':'')+'weights '+Object.entries(d.meta.weights_used).map(([k,v])=>k+' '+v.toFixed(2)).join(', ');
  d.results.forEach(g=>{const c=el('div','card');c.appendChild(el('div','persona',g.group));c.appendChild(legend(d.labels,opts));
   c.appendChild(sysRow('Crowdcast',d.labels.map(l=>g.prediction[l]),null,'ours'));
   const names={F:'Fine-tune',O:'DeepSeek V4.1 Flash',M:'Survey memory'};
   for(const k of ['F','O','M']){if(k in g.members&&g.members[k])c.appendChild(sysRow(names[k],d.labels.map(l=>g.members[k][l]),null))}
   const cr=el('div','crowds');cr.appendChild(crowd('100 simulated respondents',d.labels.map(l=>g.prediction[l])));c.appendChild(cr);
   if(g.ask_real_people!=null)c.appendChild(el('span','flag '+(g.ask_real_people?'ask':'ok'),g.ask_real_people?'Ask real people: the models disagree':'Models agree'));
   c.appendChild(el('div','meta',g.latency_s.toFixed(2)+' s'+(g.cost_usd!=null?' · $'+g.cost_usd.toFixed(5):'')));
   for(const [k,e] of Object.entries(g.member_errors||{}))c.appendChild(el('div','err','member '+k+' failed (counted as an even split): '+e));
   if(g.evidence&&g.evidence.length){const e=g.evidence[0];c.appendChild(el('div','ev','Closest past survey question (similarity '+e.similarity+'): '+e.question))}
   P.appendChild(c)})}
 catch(e){P.appendChild(el('div','err',String(e)))}finally{btn.disabled=false}}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8050)
    a = ap.parse_args()
    print("loading the scored test runs ...")
    STATE["replay"] = Replay()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"Crowdcast UI on http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
