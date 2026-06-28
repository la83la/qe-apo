"""Stage2 LLM 融合 — RRF 的替代方案(文字層)。

Stage2 預設用 RRF(結果層, qe.ensemble.rrf)。本模組提供另一條路:用一個 LLM
把每個 qid 的「N 份 Stage1 expansion + 原 query」合成「一份」融合後的擴展
passage,再做一次 BM25 檢索。對齊 qe.ensemble.fuse_text 的文字層融合精神,
但融合動作交給模型而非串接。

模型以本地 vLLM 為主(沿用 qe.vllm_worker,跑完釋放 GPU);無 GPU 時退化成
mock 串接,讓整條 pipeline 仍可跑通。融合用的 prompt 也是一個欄位,之後可交給
qe.apo 優化(Stage2 的 APO)。
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from qe.expand import CachedExpander, Expander

# Stage2 GPU 模式下未指定 --stage2-model 時的預設融合模型
DEFAULT_FUSE_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# 融合 prompt;{query} 為原查詢,{expansions} 為各 model expansion 的編號清單
DEFAULT_FUSE_PROMPT = (
    "You are merging several candidate passages written by different models for "
    "one search query. Synthesize a single, information-dense passage that keeps "
    "the most relevant and specific terminology from all candidates and drops "
    "redundancy or off-topic content.\n\n"
    "Query: {query}\n\n"
    "Candidate passages:\n{expansions}\n\n"
    "Merged passage:"
)

CACHE_DIR = Path("cache/fuse")


def _short(model: str) -> str:
    return model.split("/")[-1]


def collect_expansions(
    expanders: list[Expander], queries: dict[str, str]
) -> dict[str, list[str]]:
    """從各 expander 取出每個 qid 的 expansion 文字 -> {qid: [exp_1, exp_2, ...]}。

    CachedExpander 直接讀其 _by_qid;其他(如 MockExpander)用 expand(query)。
    """
    per_qid: dict[str, list[str]] = {}
    for qid, q in queries.items():
        exps: list[str] = []
        for e in expanders:
            if isinstance(e, CachedExpander):
                exps.append(e._by_qid.get(qid, ""))
            else:
                exps.append(e.expand(q))
        per_qid[qid] = exps
    return per_qid


def _build_messages(
    queries: dict[str, str], per_qid: dict[str, list[str]], prompt: str
) -> dict[str, str]:
    msgs: dict[str, str] = {}
    for qid, q in queries.items():
        numbered = "\n".join(
            f"[{i + 1}] {exp.strip()}" for i, exp in enumerate(per_qid[qid]) if exp.strip()
        )
        msgs[qid] = prompt.format(query=q, expansions=numbered)
    return msgs


def _to_query2doc(query: str, passage: str, repeat_orig: int = 5) -> str:
    """Query2Doc 風格:原 query 重複數次保權重,接上融合後 passage。"""
    return (" ".join([query] * repeat_orig) + " " + passage).strip()


def _cache_path(dataset: str, model: str, prompt: str, msgs: dict[str, str]) -> Path:
    # key 含融合輸入(msgs)的指紋,Stage1 expansion 或 prompt 一改就 miss、重生成
    fingerprint = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
    h = hashlib.sha1(f"{prompt}|{fingerprint}".encode()).hexdigest()[:8]
    safe_ds = dataset.replace("/", "-")
    return CACHE_DIR / f"{safe_ds}__fuse-{_short(model)}__{h}.json"


def fuse_llm(
    queries: dict[str, str],
    expanders: list[Expander],
    model: str = "",
    gpu: int = -1,
    dataset: str = "stage2",
    prompt: str = DEFAULT_FUSE_PROMPT,
    force: bool = False,
    gpu_mem_util: float = 0.70,
    max_model_len: int = 4096,
) -> dict[str, str]:
    """Stage2 LLM 融合,回傳 {qid: 融合後的擴展 query}(可直接丟進 BM25 檢索)。

    gpu < 0  -> mock:把各 expansion 串接當融合 passage(無 GPU 也能跑)。
    gpu >= 0 -> 用 qe.vllm_worker 跑 model 生成融合 passage。
    """
    per_qid = collect_expansions(expanders, queries)
    msgs = _build_messages(queries, per_qid, prompt)

    if gpu < 0:
        # mock 融合:串接非空 expansion
        fused = {
            qid: _to_query2doc(queries[qid], " ".join(e for e in per_qid[qid] if e.strip()))
            for qid in queries
        }
        return fused

    model = model or DEFAULT_FUSE_MODEL
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = _cache_path(dataset, model, prompt, msgs)

    if out.exists() and not force:
        print(f"[stage2] cache hit: {out}")
        with open(out) as f:
            passages = json.load(f)
    else:
        msgs_file = CACHE_DIR / f"_messages-{_short(model)}.json"
        with open(msgs_file, "w") as f:
            json.dump(msgs, f, ensure_ascii=False)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        cmd = [
            sys.executable, "-m", "qe.vllm_worker",
            "--model", model,
            "--messages-file", str(msgs_file),
            "--out", str(out),
            "--gpu-mem-util", str(gpu_mem_util),
            "--max-model-len", str(max_model_len),
        ]
        print(f"[stage2] LLM fuse with {model} on GPU {gpu} ...")
        subprocess.run(cmd, env=env, check=True)
        with open(out) as f:
            passages = json.load(f)

    return {qid: _to_query2doc(queries[qid], passages.get(qid, "")) for qid in queries}
