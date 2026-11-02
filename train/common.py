"""Shared presets for our Fireworks serverless LoRA runs.

Model IDs, tokenizers and renderer names come from the fw-ai/cookbook source
(training/renderer/_model_info.py, training/utils/supervised.py). Thinking is
turned off: our targets are plain JSON answers with no reasoning.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COOKBOOK = ROOT / "cookbook" / "training"
sys.path.insert(0, str(ROOT / "cookbook"))   # makes `training.*` importable
sys.path.insert(0, str(ROOT / "eval"))       # makes `tasks.simbench` importable

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

SERVERLESS_URL = "https://api.fireworks.ai/training/v1/serverless"

BASES = {
    "qwen": {
        "base_model": "accounts/fireworks/models/qwen3p6-35b-a3b",
        "tokenizer": "Qwen/Qwen3.6-35B-A3B",
        "renderer": "qwen3_5_disable_thinking",
    },
    "nem": {
        "base_model": "accounts/fireworks/models/nvidia-nemotron-3p5-super-vl-120b-a12b-bf16",
        "tokenizer": "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
        "renderer": "nemotron35_vl_disable_thinking",
    },
}


def api_key():
    k = os.environ.get("FIREWORKS_API_KEY")
    if not k:
        sys.exit("FIREWORKS_API_KEY missing (put it in .env)")
    return k
