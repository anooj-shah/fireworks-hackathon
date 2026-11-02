"""Create + upload SimBench SFT datasets to Fireworks via the management API."""
import sys

import requests
from dotenv import load_dotenv

sys.path.insert(0, ".")
load_dotenv()
from simbench.fireworks_client import get_api_key

ACCOUNT_ID = "thisisanooj"
BASE = f"https://api.fireworks.ai/v1/accounts/{ACCOUNT_ID}/datasets"
key = get_api_key()
headers_json = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
headers_auth = {"Authorization": f"Bearer {key}"}


def create_and_upload(dataset_id: str, display_name: str, file_path: str):
    with open(file_path) as f:
        example_count = sum(1 for _ in f)
    resp = requests.post(
        BASE,
        headers=headers_json,
        json={
            "datasetId": dataset_id,
            "dataset": {"displayName": display_name, "format": "CHAT", "exampleCount": str(example_count)},
        },
    )
    print(f"create {dataset_id}: {resp.status_code}")
    if resp.status_code not in (200, 201) and "already exists" not in resp.text.lower():
        print(resp.text[:500])
        return
    elif resp.status_code not in (200, 201):
        print("(dataset already exists, continuing to upload)")

    with open(file_path, "rb") as f:
        up = requests.post(f"{BASE}/{dataset_id}:upload", headers=headers_auth, files={"file": f})
    print(f"upload {dataset_id}: {up.status_code}")
    print(up.text[:500])


if __name__ == "__main__":
    create_and_upload("simbench-sft-train", "SimBench SFT train", "data/sft_train.jsonl")
    create_and_upload("simbench-sft-val", "SimBench SFT val", "data/sft_val.jsonl")
