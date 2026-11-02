"""Verify the Fireworks API key works, and list available models — without ever printing the key itself."""
import sys

import requests
from dotenv import load_dotenv

sys.path.insert(0, ".")
load_dotenv()
from simbench.fireworks_client import get_api_key

key = get_api_key()
if not key:
    print("No API key found in environment or .env (looked for FIREWORKS_API_KEY, SECRET_KEY)")
    sys.exit(1)

print(f"Found an API key (length {len(key)}, prefix '{key[:3]}***') — testing it against Fireworks...")

resp = requests.get(
    "https://api.fireworks.ai/inference/v1/models",
    headers={"Authorization": f"Bearer {key}"},
    timeout=15,
)
print(f"GET /v1/models -> HTTP {resp.status_code}")
if resp.status_code == 200:
    data = resp.json()
    models = data.get("data", [])
    print(f"{len(models)} models visible to this key")
    for m in models[:20]:
        print(" -", m.get("id"))
else:
    print(resp.text[:1000])
