"""Stage1 orchestrator — 在指定 GPU 上依序跑多個 open source model 做 expansion。

每個 model 開獨立 subprocess(qe.vllm_worker),跑完釋放 GPU 再換下一個。
結果快取在 cache/expand/,key 含 dataset + model + prompt hash,改 prompt(APO)
會自動 miss、重生成;沒改就直接讀快取。

用法:
    python -m qe.stage1 --gpu 1 --dataset beir/scifact/test
    python -m qe.stage1 --gpu 1 --models Qwen/Qwen2.5-7B-Instruct,unsloth/mistral-7b-instruct-v0.3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from qe.expand import DEFAULT_EXPAND_PROMPT

DEFAULT_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "NousResearch/Meta-Llama-3.1-8B-Instruct",
    "unsloth/mistral-7b-instruct-v0.3",
]

CACHE_DIR = Path("cache/expand")


def _short(model: str) -> str:
    return model.split("/")[-1]


def _cache_path(dataset: str, model: str, prompt: str, limit: int) -> Path:
    h = hashlib.sha1(f"{prompt}|{limit}".encode()).hexdigest()[:8]
    safe_ds = dataset.replace("/", "-")
    return CACHE_DIR / f"{safe_ds}__{_short(model)}__{h}.json"


def run_stage1(
    dataset: str,
    gpu: int,
    models: list[str],
    prompt: str = DEFAULT_EXPAND_PROMPT,
    limit: int = 0,
    force: bool = False,
) -> dict[str, str]:
    """回傳 {model -> cache_file_path};各檔內容是 {query_id -> expansion}。"""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prompt_file = CACHE_DIR / "_prompt.txt"
    prompt_file.write_text(prompt)

    paths: dict[str, str] = {}
    for model in models:
        out = _cache_path(dataset, model, prompt, limit)
        if out.exists() and not force:
            print(f"[stage1] cache hit: {out}")
            paths[model] = str(out)
            continue

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [
            sys.executable, "-m", "qe.vllm_worker",
            "--model", model,
            "--dataset", dataset,
            "--prompt-file", str(prompt_file),
            "--out", str(out),
        ]
        if limit:
            cmd += ["--limit", str(limit)]
        print(f"[stage1] running {model} on GPU {gpu} ...")
        subprocess.run(cmd, env=env, check=True)
        paths[model] = str(out)
    return paths


def load_expansions(cache_file: str) -> dict[str, str]:
    with open(cache_file) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/scifact/test")
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    paths = run_stage1(args.dataset, args.gpu, models, limit=args.limit, force=args.force)
    print("\n[stage1] done:")
    for m, p in paths.items():
        print(f"  {m} -> {p}")


if __name__ == "__main__":
    main()
