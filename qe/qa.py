"""QA end-to-end driver — 兩階段檢索後接 reader, 量 EM/F1。

pipeline(對齊 EXPERIMENT_PLAN 的 QA task):
    question
      ─[Stage1: multi-model expansion]─▶ 各 model 的 expanded query
      ─[Stage2: ensemble(RRF)]────────▶ 在「該題候選池」內檢索 + 融合
      ─ top-k passages ─[Reader]──────▶ answer ─ EM/F1 vs 金答案

檢索採 per-query pool:每題只在自己的 ~10 段候選 passages 內檢索/重排
(MS MARCO v1.1 RC 原生設定),不建全域 corpus。

方法:
    closed   無 passage,reader 只靠內建知識(floor)
    B0       BM25 原問題 top-k(檢索基線)
    M1       單 model expansion + 檢索 top-k
    M2       多 model expansion + RRF top-k
    oracle   金標 passage(is_selected=1)餵 reader(ceiling)

用法:
    # smoke(mock reader, 無 GPU)
    python -m qe.qa --split validation --limit 20 --methods closed,B0,oracle --reader mock
    # 真分數(vLLM reader 在 GPU5;M1/M2 需先在 GPU5 產 expansion)
    python -m qe.qa --split validation --limit 100 --k 3 \
        --methods closed,B0,M1,M2,oracle --reader vllm --gpu 5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from qe import ensemble, expand, qa_metrics, reader
from qe.expand import DEFAULT_EXPAND_PROMPT
from qe.qa_data import QAExample, load_qa, summary
from qe.retrieve import BM25Index

# 16GB 卡(RTX 4080)跑得下、且跨家族的縮小版 expansion 三人組
DEFAULT_QA_EXPAND_MODELS = [
    "Qwen/Qwen2.5-3B-Instruct",
    "unsloth/Llama-3.2-1B-Instruct",
    "google/gemma-4-E2B-it",
]
EXPAND_CACHE_DIR = Path("cache/expand")


# ---------- Stage1: 對 QA 問題做 expansion(messages-file 模式) ----------

def _qa_expand_one(examples: list[QAExample], model: str, gpu: int, tag: str,
                   gpu_mem_util: float, max_model_len: int, force: bool) -> dict[str, str]:
    """用一個本地模型對所有問題生成 expansion,結果快取。回傳 {qid: expansion}。"""
    EXPAND_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    msgs = {e.qid: DEFAULT_EXPAND_PROMPT.format(query=e.question) for e in examples}
    short = model.split("/")[-1]
    h = hashlib.sha1(f"{DEFAULT_EXPAND_PROMPT}|{tag}".encode()).hexdigest()[:8]
    out = EXPAND_CACHE_DIR / f"{tag}__{short}__{h}.json"

    if out.exists() and not force:
        print(f"[qa-stage1] cache hit: {out}")
        with open(out) as f:
            return json.load(f)

    msgs_file = EXPAND_CACHE_DIR / f"_qa-msgs-{short}.json"
    with open(msgs_file, "w") as f:
        json.dump(msgs, f, ensure_ascii=False)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
    cmd = [
        sys.executable, "-m", "qe.vllm_worker",
        "--model", model, "--messages-file", str(msgs_file), "--out", str(out),
        "--max-tokens", "256", "--temperature", "0.7",
        "--gpu-mem-util", str(gpu_mem_util), "--max-model-len", str(max_model_len),
    ]
    print(f"[qa-stage1] expand with {model} on GPU {gpu}: {len(msgs)} questions ...")
    subprocess.run(cmd, env=env, check=True)
    with open(out) as f:
        return json.load(f)


def build_qa_expanders(examples: list[QAExample], models: list[str], gpu: int, tag: str,
                       gpu_mem_util: float = 0.85, max_model_len: int = 2048,
                       force: bool = False) -> list[expand.CachedExpander]:
    out = []
    for m in models:
        by_qid = _qa_expand_one(examples, m, gpu, tag, gpu_mem_util, max_model_len, force)
        out.append(expand.CachedExpander(name=m.split("/")[-1], _by_qid=by_qid))
    return out


# ---------- 每題候選池檢索 ----------

def _pool_index(ex: QAExample) -> BM25Index:
    return BM25Index({str(i): p for i, p in enumerate(ex.passages)})


def _ranked_passages(ex: QAExample, run_for_q: dict[str, float], k: int) -> list[str]:
    top = sorted(run_for_q.items(), key=lambda x: x[1], reverse=True)[:k]
    return [ex.passages[int(doc_id)] for doc_id, _ in top]


def _expanded_query(e: expand.Expander, qid: str, question: str) -> str:
    if isinstance(e, expand.CachedExpander):
        return e.expand_query_for_qid(qid, question)
    return e.expand_query(question)


def _select_passages(ex: QAExample, method: str, k: int, idx: BM25Index,
                     expanders: list[expand.Expander]) -> list[str]:
    if method == "closed":
        return []
    if method == "oracle":
        gold = [p for p, s in zip(ex.passages, ex.selected) if s == 1]
        return gold[:k] if gold else ex.passages[:1]
    if method == "B0":
        run = idx.search({ex.qid: ex.question}, k=len(ex.passages))
        return _ranked_passages(ex, run[ex.qid], k)
    if method == "M1":
        eq = _expanded_query(expanders[0], ex.qid, ex.question)
        run = idx.search({ex.qid: eq}, k=len(ex.passages))
        return _ranked_passages(ex, run[ex.qid], k)
    if method == "M2":
        runs = [idx.search({ex.qid: _expanded_query(e, ex.qid, ex.question)},
                           k=len(ex.passages)) for e in expanders]
        fused = ensemble.rrf(runs, top_k=len(ex.passages))
        return _ranked_passages(ex, fused[ex.qid], k)
    raise ValueError(f"unknown method {method!r}")


def run_qa_eval(dataset, split, limit, k, methods, reader_kind, reader_model,
                gpu, expand_models, force=False, gpu_mem_util=0.45):
    """跑 QA 方法,回傳 {method: {em, f1}}。CLI(main)與彙整 script 共用。

    gpu_mem_util: vLLM 的 gpu_memory_utilization。共享 GPU 上請壓低(預設 0.45),
    避免吃滿整張卡把別人的 process 擠掉/OOM。16GB 卡 0.45≈7.2GB。
    """
    print(f"載入 QA {dataset} [{split}] ...")
    examples = load_qa(dataset, split=split, limit=limit)
    print(" ", summary(examples))

    expanders: list[expand.Expander] = []
    if {"M1", "M2"} & set(methods):
        models = [m for m in expand_models.split(",") if m] or DEFAULT_QA_EXPAND_MODELS
        tag = f"qa-{split}-{limit or 'full'}"
        if reader_kind == "vllm":
            print(f"Stage1: 在 GPU {gpu} 對 QA 問題跑 expansion {models}")
            expanders = build_qa_expanders(examples, models, gpu, tag,
                                           gpu_mem_util=gpu_mem_util, force=force)
        else:
            print("Stage1: reader=mock → M1/M2 改用 mock expander(非真 expansion)")
            expanders = expand.build_expanders(None)

    rdr = reader.build_reader(reader_kind, model=reader_model, gpu=gpu,
                              gpu_mem_util=gpu_mem_util) \
        if reader_kind == "vllm" else reader.build_reader(reader_kind)
    print(f"reader: {rdr.name}  methods: {methods}")

    items: dict[str, tuple[str, list[str]]] = {}
    golds: dict[str, list[str]] = {}
    keymap: dict[str, list[str]] = {m: [] for m in methods}
    for ex in examples:
        if not ex.passages:
            continue
        idx = _pool_index(ex)
        golds[ex.qid] = ex.answers
        for m in methods:
            top = _select_passages(ex, m, k, idx, expanders)
            key = f"{m}||{ex.qid}"
            items[key] = (ex.question, top)
            keymap[m].append(key)

    print(f"reader 生成 {len(items)} 筆答案 ...")
    answers = rdr.read_many(items)

    results: dict[str, dict[str, float]] = {}
    for m in methods:
        preds = {key.split("||", 1)[1]: answers.get(key, "") for key in keymap[m]}
        results[m] = qa_metrics.score(preds, golds)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Lala8383/ms-marco-qa-10k")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 題")
    ap.add_argument("--k", type=int, default=3, help="餵 reader 的 top-k passages")
    ap.add_argument("--methods", default="closed,B0,M1,M2,oracle")
    ap.add_argument("--reader", default="mock", help="mock 或 vllm")
    ap.add_argument("--reader-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--gpu", type=int, default=5, help="vLLM reader / expansion 用的 GPU")
    ap.add_argument("--expand-models", default="", help="逗號分隔;留空用 DEFAULT_QA_EXPAND_MODELS")
    ap.add_argument("--gpu-mem-util", type=float, default=0.45,
                    help="vLLM gpu_memory_utilization;共享卡壓低免擠掉別人(16GB 卡 0.45≈7.2GB)")
    ap.add_argument("--force", action="store_true", help="忽略快取重跑")
    args = ap.parse_args()

    methods = args.methods.split(",")
    results = run_qa_eval(args.dataset, args.split, args.limit, args.k, methods,
                          args.reader, args.reader_model, args.gpu,
                          args.expand_models, args.force, args.gpu_mem_util)

    print("\n===== QA 對照表 =====")
    names = ["em", "f1"]
    print(f"{'method':10}" + "".join(f"{n:>10}" for n in names))
    for m in methods:
        sc = results[m]
        print(f"{m:10}" + "".join(f"{sc.get(n, 0):>10.4f}" for n in names))


if __name__ == "__main__":
    main()
