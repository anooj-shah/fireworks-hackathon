#!/usr/bin/env python3
"""SF Ballot Predictor — enter your demographics and see how you'd likely vote on SF's
5 most contested 2026 ballot measures. Runs on http://127.0.0.1:8051

  python harness/sf_ballot_ui.py
"""
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
import crowdcast as C             # noqa: E402
import cli                        # noqa: E402

# Top 5 most controversial SF November 2026 props — election day is Nov 3, 2026; no results yet
PROPS = [
    {
        "prop": "G",
        "title": "Prop G — Reopen Upper Great Highway to Cars",
        "short": "Reopen the Upper Great Highway to private vehicles on weekdays, reversing the park closure voters approved in June.",
        "question": "Shall the City close Sunset Dunes Park from Monday at 4:00 a.m. through Friday at 6:00 p.m., unless a holiday falls on a weekday, in order to reopen the roadway known as the Upper Great Highway to private vehicles during that time?",
        "options": ["Yes", "No"],
    },
    {
        "prop": "H",
        "title": "Prop H — Muni Parcel Tax",
        "short": "New annual parcel tax ($129–$400k/parcel) raising ~$177M/year for 15 years to fund Muni bus and rail service.",
        "question": "Shall the City fund Muni service through a new annual parcel tax on real estate at rates ranging from $129 per parcel up to a maximum of $400,000 per parcel, depending on the type of property and its size and adjusted annually for inflation, for estimated revenue of $177 million a year for 15 years?",
        "options": ["Yes", "No"],
    },
    {
        "prop": "B",
        "title": "Prop B — Public Bank",
        "short": "Authorize the City to establish a municipal finance corporation and a public bank.",
        "question": "Shall the City amend the Charter to authorize the City to establish a municipal finance corporation and a public bank?",
        "options": ["Yes", "No"],
    },
    {
        "prop": "D",
        "title": "Prop D — Ballot Measure Signature Rules",
        "short": "Raise the signature threshold for citizen initiatives from 2% to 8% of registered voters (~10,600 → 42,500 signatures).",
        "question": "Shall the City amend the Charter to increase the requirements for signature initiatives from 2% to 8% of registered voters in San Francisco (currently from about 10,600 to about 42,500 signatures), increase the requirements for calling special elections, allow signature initiatives to be withdrawn before an election, and remove the ability of a minority of Board of Supervisors members or the Mayor to place measures on the ballot?",
        "options": ["Yes", "No"],
    },
    {
        "prop": "I",
        "title": "Prop I — Transfer Tax for Affordable Housing",
        "short": "Permanently dedicate half of the transfer tax on $10M+ properties (5.5–6% rate) to affordable housing.",
        "question": "Shall the City permanently dedicate half of the transfer tax revenues collected when certain real estate with a value or sales price of more than $10,000,000 changes hands, which will continue to be taxed at rates of 5.5% to 6%, to be spent only on affordable housing, including social housing developments and housing assistance, and exempt certain sales from the dedicated portion of the tax, for an estimated annual revenue decrease of approximately $1 million?",
        "options": ["Yes", "No"],
    },
]

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


def build_persona(age, housing, employment, district, extra):
    parts = ["The year is 2026. You are a voter in San Francisco."]
    if age:
        parts.append(f"You are in the {age} age group.")
    if housing:
        parts.append(f"You are a {housing}.")
    if employment:
        parts.append(f"You work in {employment}.")
    if district:
        parts.append(f"You live in Supervisorial District {district}.")
    if extra and extra.strip():
        parts.append(extra.strip())
    return " ".join(parts)


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
        try:
            if urlparse(self.path).path == "/":
                return self.send(200, PAGE, "text/html")
            if urlparse(self.path).path == "/api/props":
                return self.send(200, [{"prop": p["prop"], "title": p["title"], "short": p["short"]} for p in PROPS])
            return self.send(404, {"error": "not found"})
        except Exception as e:
            return self.send(500, {"error": str(e)})

    def do_POST(self):
        if urlparse(self.path).path != "/api/predict":
            return self.send(404, {"error": "not found"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            persona = build_persona(
                body.get("age", ""),
                body.get("housing", ""),
                body.get("employment", ""),
                body.get("district", ""),
                body.get("extra", ""),
            )
            sysm, err = get_system()
            if sysm is None:
                return self.send(400, {"error": err or "system not ready"})

            results = []
            for p in PROPS:
                result = cli.predict(sysm, [persona], p["question"], p["options"])
                g = result["results"][0]
                pred_yes = g["prediction"].get("A", 0.5)
                results.append({
                    "prop": p["prop"],
                    "title": p["title"],
                    "short": p["short"],
                    "pred_yes": pred_yes,
                    "latency_s": g["latency_s"],
                    "ask_real_people": g.get("ask_real_people"),
                })

            return self.send(200, {"persona": persona, "results": results})
        except Exception as e:
            import traceback; traceback.print_exc()
            return self.send(500, {"error": str(e)})


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>SF Ballot Predictor — Crowdcast</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{
  --paper:#FAF9F5;--card:#FFFFFF;--sand:#F0EEE6;--line:#E8E6DC;--ink:#141413;--mute:#73726C;
  --clay:#D97757;--clay-d:#C15F3C;--yes:#4E8C5A;--no:#A63D2F;
  --serif:"Copernicus","Tiempos Headline","Georgia",serif;
  --sans:"Styrene B","Söhne",ui-sans-serif,-apple-system,sans-serif;
}
*{box-sizing:border-box}
html,body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 var(--sans);-webkit-font-smoothing:antialiased}
button,input,textarea,select{font:inherit;color:inherit}
:focus-visible{outline:2px solid var(--clay);outline-offset:2px}
.top{display:flex;align-items:center;gap:10px;padding:18px 28px;border-bottom:1px solid var(--line)}
.mark{display:grid;grid-template-columns:repeat(3,5px);gap:2px}.mark i{width:5px;height:5px;border-radius:50%;background:var(--clay)}
.mark i:nth-child(2n){opacity:.55}
.brand{font:500 17px var(--serif);letter-spacing:-.01em}
.wrap{max-width:760px;margin:0 auto;padding:8px 24px 80px}
.hero{text-align:center;margin:32px 0 28px}
.hero h1{font:400 38px/1.15 var(--serif);letter-spacing:-.02em;margin:0 0 10px}
.hero p{color:var(--mute);margin:0;max-width:520px;margin:0 auto}
.form-card{background:var(--card);border:1px solid var(--line);border-radius:20px;padding:22px 24px;box-shadow:0 2px 10px rgba(20,20,19,.04);margin-bottom:24px}
.form-card h2{font:500 17px var(--serif);margin:0 0 18px;letter-spacing:-.01em}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font-size:13px;color:var(--mute);font-weight:500}
.field select,.field input{border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--paper);width:100%}
.full{grid-column:1/-1}
.field textarea{border:1px solid var(--line);border-radius:10px;padding:8px 10px;background:var(--paper);width:100%;resize:vertical;min-height:72px}
.actions{display:flex;justify-content:flex-end;margin-top:18px}
.btn{border:0;background:var(--clay);color:#fff;border-radius:10px;padding:10px 22px;cursor:pointer;font-weight:500;font-size:15px}
.btn:hover{background:var(--clay-d)}.btn:disabled{opacity:.5;cursor:default}
.persona-box{background:var(--sand);border-radius:12px;padding:10px 14px;font-size:13px;color:var(--mute);margin-bottom:20px;line-height:1.5}
.persona-box strong{color:var(--ink)}
.prop-card{background:var(--card);border:1px solid var(--line);border-radius:16px;padding:20px 22px;margin-bottom:14px;box-shadow:0 1px 4px rgba(20,20,19,.04)}
.prop-card h3{font:500 16px var(--serif);margin:0 0 4px;letter-spacing:-.01em}
.prop-card .desc{font-size:13.5px;color:var(--mute);margin:0 0 16px}
.bar-row{margin-bottom:10px}
.bar-label{display:flex;justify-content:space-between;font-size:13px;margin-bottom:4px}
.bar-label .name{font-weight:500}
.bar-label .pct{font-variant-numeric:tabular-nums}
.bar-track{height:22px;border-radius:7px;background:var(--sand);overflow:hidden;display:flex}
.bar-yes{height:100%;background:var(--yes);border-radius:7px 0 0 7px;transition:width .4s ease}
.bar-no{height:100%;background:var(--no);flex:1;border-radius:0 7px 7px 0}
.bar-track.actual .bar-yes{opacity:.4}.bar-track.actual .bar-no{opacity:.4}
.verdict{display:inline-block;margin-top:10px;padding:4px 11px;border-radius:999px;font-size:13px;font-weight:500}
.verdict.yes{background:#E5F0E7;color:#2D6B37}
.verdict.no{background:#F5E5E3;color:#8B2B22}
.verdict.lean-yes{background:#EEF5EF;color:#3A7844}
.verdict.lean-no{background:#F8EDEC;color:#9B3628}
.flag{display:inline-block;margin-top:8px;margin-left:8px;padding:4px 11px;border-radius:999px;font-size:12px;background:#F6E6D2;color:#7A4A1E}
.legend{display:flex;gap:16px;font-size:13px;color:var(--mute);margin-bottom:16px}
.legend span{display:flex;align-items:center;gap:6px}
.legend .dot{width:10px;height:10px;border-radius:3px}
.err{color:#A63D2F;background:#F8E3DE;border-radius:12px;padding:12px 16px;margin-bottom:16px}
.note{color:var(--mute);font-size:13px;text-align:center;max-width:600px;margin:24px auto 0}
.spinner{display:inline-block;width:16px;height:16px;border:2px solid #fff;border-top-color:transparent;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
@media(max-width:560px){.grid{grid-template-columns:1fr}.hero h1{font-size:28px}}
@media(prefers-reduced-motion:no-preference){.prop-card{animation:in .2s ease-out}@keyframes in{from{opacity:0;transform:translateY(4px)}to{opacity:1}}}
</style></head><body>
<div class="top"><div class="mark" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div><span class="brand">Crowdcast</span></div>
<div class="wrap">
  <div class="hero">
    <h1>How will you vote on SF's Nov 2026 ballot?</h1>
    <p>Enter your demographics and Crowdcast predicts your likely position on the 5 most contested November 3, 2026 measures — before results are in.</p>
  </div>
  <div class="form-card">
    <h2>Tell us about yourself</h2>
    <div class="grid">
      <div class="field">
        <label for="age">Age group</label>
        <select id="age">
          <option value="">Prefer not to say</option>
          <option value="18-29">18–29</option>
          <option value="30-44">30–44</option>
          <option value="45-59">45–59</option>
          <option value="60-74">60–74</option>
          <option value="75+">75+</option>
        </select>
      </div>
      <div class="field">
        <label for="housing">Housing status</label>
        <select id="housing">
          <option value="">Prefer not to say</option>
          <option value="homeowner">Homeowner</option>
          <option value="renter">Renter</option>
        </select>
      </div>
      <div class="field">
        <label for="employment">Industry</label>
        <select id="employment">
          <option value="">Prefer not to say</option>
          <option value="technology">Technology</option>
          <option value="healthcare">Healthcare</option>
          <option value="education">Education</option>
          <option value="finance">Finance</option>
          <option value="government or public service">Government / Public service</option>
          <option value="retail or hospitality">Retail / Hospitality</option>
          <option value="construction or trades">Construction / Trades</option>
          <option value="arts or media">Arts / Media</option>
          <option value="a nonprofit">Nonprofit</option>
          <option value="transportation">Transportation</option>
        </select>
      </div>
      <div class="field">
        <label for="district">Supervisorial District</label>
        <select id="district">
          <option value="">Any / Don't know</option>
          <option value="1">D1 — Richmond</option>
          <option value="2">D2 — Marina / Pacific Heights</option>
          <option value="3">D3 — Chinatown / North Beach</option>
          <option value="4">D4 — Sunset</option>
          <option value="5">D5 — Haight / Western Addition</option>
          <option value="6">D6 — SoMa / Tenderloin</option>
          <option value="7">D7 — West Portal / Glen Park</option>
          <option value="8">D8 — Castro / Noe Valley</option>
          <option value="9">D9 — Mission / Bernal Heights</option>
          <option value="10">D10 — Bayview / Potrero</option>
          <option value="11">D11 — Excelsior / Ingleside</option>
        </select>
      </div>
      <div class="field full">
        <label for="extra">Anything else about you? <span style="font-weight:400">(optional)</span></label>
        <textarea id="extra" placeholder="e.g. I have kids in SFUSD. I commute by bike. I moved here 2 years ago. I'm a small business owner."></textarea>
      </div>
    </div>
    <div class="actions"><button class="btn" id="go" onclick="predict()">Predict my votes</button></div>
  </div>
  <div id="results"></div>
  <p class="note">Predictions are from Crowdcast, an AI ensemble trained on real survey data. This is not a poll. Election is November 3, 2026 — results not yet available.</p>
</div>
<script>
function verdict(pct){
  if(pct>=0.6) return ['yes','Strong Yes'];
  if(pct>=0.5) return ['lean-yes','Lean Yes'];
  if(pct>=0.4) return ['lean-no','Lean No'];
  return ['no','Strong No'];
}
async function predict(){
  const btn=document.getElementById('go'), R=document.getElementById('results');
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Predicting…';
  R.innerHTML='';
  const body={
    age:document.getElementById('age').value,
    housing:document.getElementById('housing').value,
    employment:document.getElementById('employment').value,
    district:document.getElementById('district').value,
    extra:document.getElementById('extra').value
  };
  try{
    const r=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const d=await r.json();
    if(d.error){R.innerHTML='<div class="err">'+d.error+'</div>';return;}

    // Persona box
    const pb=document.createElement('div');
    pb.className='persona-box';
    pb.innerHTML='<strong>Your profile:</strong> '+d.persona;
    R.appendChild(pb);

    // Legend
    const leg=document.createElement('div');
    leg.className='legend';
    leg.innerHTML='<span><span class="dot" style="background:#4E8C5A"></span>Predicted Yes</span><span><span class="dot" style="background:#A63D2F"></span>Predicted No</span>';
    R.appendChild(leg);

    d.results.forEach(p=>{
      const card=document.createElement('div');
      card.className='prop-card';
      const [vcls,vtxt]=verdict(p.pred_yes);
      const flag=p.ask_real_people?'<span class="flag">Models disagree — uncertain</span>':'';
      card.innerHTML=`
        <h3>${p.title}</h3>
        <div class="desc">${p.short}</div>
        <div class="bar-row">
          <div class="bar-label"><span class="name">Predicted vote</span><span class="pct">${Math.round(p.pred_yes*100)}% Yes / ${Math.round((1-p.pred_yes)*100)}% No</span></div>
          <div class="bar-track">
            <div class="bar-yes" style="width:${p.pred_yes*100}%"></div>
            <div class="bar-no"></div>
          </div>
        </div>
        <span class="verdict ${vcls}">${vtxt}</span>${flag}
      `;
      R.appendChild(card);
    });
  }catch(e){
    R.innerHTML='<div class="err">Could not reach server: '+e+'</div>';
  }finally{
    btn.disabled=false;
    btn.innerHTML='Predict my votes';
  }
}
</script></body></html>"""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8051)
    a = ap.parse_args()
    print("Loading survey memory and model settings...")
    get_system()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    print(f"SF Ballot Predictor on http://127.0.0.1:{a.port}  (Ctrl-C to stop)")
    srv.serve_forever()


if __name__ == "__main__":
    main()
