"""End-to-end driver: 跑 baseline 與本方法, 印出對照表。

用法:
    python -m qe.run --dataset beir/scifact/test --k 100
    python -m qe.run --methods B0,M2,M5            # 只跑指定方法

方法代號對應 EXPERIMENT_PLAN.md:
    B0  BM25 無 expansion
    M1  單一 model + expansion (無 ensemble)
    M2  多 model + RRF (無 APO)
    M5  多 model + RRF + APO (本方法雛形)
"""
from __future__ import annotations

import argparse

from qe import apo, ensemble, expand, metrics
from qe.data import load_ir
from qe.retrieve import BM25Index


def _expand_queries(exp: expand.Expander, queries: dict[str, str]) -> dict[str, str]:
    if isinstance(exp, expand.CachedExpander):
        return {qid: exp.expand_query_for_qid(qid, q) for qid, q in queries.items()}
    return {qid: exp.expand_query(q) for qid, q in queries.items()}


def run_b0(index, queries, qrels, k):
    return metrics.score(qrels, index.search(queries, k=k))


def run_m1(index, queries, qrels, k, expanders):
    exp = expanders[0]
    eq = _expand_queries(exp, queries)
    return metrics.score(qrels, index.search(eq, k=k))


def run_m2(index, queries, qrels, k, expanders):
    runs = []
    for exp in expanders:
        eq = _expand_queries(exp, queries)
        runs.append(index.search(eq, k=k))
    fused = ensemble.rrf(runs, top_k=k)
    return metrics.score(qrels, fused)


def run_m5(index, queries, qrels, k, expanders, dev_queries, dev_qrels):
    """對每個 expander 用 APO 在 dev 上挑 prompt, 再在 test 上 RRF。
    mock expander 不吃 prompt, 所以 APO 在此只是示範流程; 接上 Anthropic 後才會真正生效。
    """
    optimized = []
    for exp in expanders:
        if isinstance(exp, expand.AnthropicExpander):
            def reward(p: str, exp=exp) -> float:
                exp.prompt = p
                eq = _expand_queries(exp, dev_queries)
                return metrics.score(dev_qrels, index.search(eq, k=k)).get("ndcg@10", 0.0)

            res = apo.optimize(reward, exp.prompt, rounds=2, candidates_per_round=2)
            exp.prompt = res.best_prompt
            print(f"  [APO] {exp.name}: dev ndcg@10 -> {res.best_score:.4f}")
        optimized.append(exp)
    runs = [index.search(_expand_queries(e, queries), k=k) for e in optimized]
    fused = ensemble.rrf(runs, top_k=k)
    return metrics.score(qrels, fused)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/scifact/test")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--methods", default="B0,M1,M2,M5")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 個 query (除錯用)")
    ap.add_argument("--models", default="", help="逗號分隔的 Anthropic model id; 留空用 mock")
    ap.add_argument("--local-gpu", type=int, default=-1,
                    help="設了就用本地 open source model 跑 Stage1(在該 GPU),expander 改用 vLLM 快取")
    ap.add_argument("--local-models", default="",
                    help="本地 Stage1 的 HF model id(逗號分隔);留空用 qe.stage1.DEFAULT_MODELS")
    args = ap.parse_args()

    print(f"載入 {args.dataset} ...")
    ds = load_ir(args.dataset)
    print(" ", ds.summary())

    queries = ds.queries
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])

    print("建 BM25 index ...")
    index = BM25Index(ds.corpus)

    if args.local_gpu >= 0:
        from qe import stage1
        local_models = [m for m in args.local_models.split(",") if m] or stage1.DEFAULT_MODELS
        print(f"Stage1: 本地 vLLM 在 GPU {args.local_gpu} 跑 {local_models}")
        paths = stage1.run_stage1(args.dataset, args.local_gpu, local_models, limit=args.limit)
        expanders = expand.load_cached_expanders(paths)
    else:
        model_ids = [m for m in args.models.split(",") if m] or None
        expanders = expand.build_expanders(model_ids)
    print(f"expanders: {[e.name for e in expanders]}")

    methods = args.methods.split(",")
    results: dict[str, dict[str, float]] = {}
    for m in methods:
        print(f"\n>>> {m}")
        if m == "B0":
            results[m] = run_b0(index, queries, ds.qrels, args.k)
        elif m == "M1":
            results[m] = run_m1(index, queries, ds.qrels, args.k, expanders)
        elif m == "M2":
            results[m] = run_m2(index, queries, ds.qrels, args.k, expanders)
        elif m == "M5":
            # 簡化: 用同一份 queries 當 dev (正式實驗請改成獨立 dev split)
            results[m] = run_m5(index, queries, ds.qrels, args.k, expanders, queries, ds.qrels)
        else:
            print(f"  (未知方法 {m}, 跳過)")
            continue
        print("  ", results[m])

    print("\n===== 對照表 =====")
    metric_names = metrics.DEFAULT_METRICS
    print(f"{'method':8}" + "".join(f"{mn:>14}" for mn in metric_names))
    for m, sc in results.items():
        print(f"{m:8}" + "".join(f"{sc.get(mn, 0):>14.4f}" for mn in metric_names))


if __name__ == "__main__":
    main()
