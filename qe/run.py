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
from dataclasses import dataclass

from qe import apo, ensemble, expand, metrics
from qe.data import load_ir
from qe.retrieve import BM25Index


@dataclass
class Stage2Config:
    """Stage2 融合設定。mode='rrf'(結果層,預設) 或 'llm'(文字層, 用模型融合)。"""

    mode: str = "rrf"
    model: str = ""
    gpu: int = -1
    dataset: str = "stage2"
    # 大模型(如 Qwen2.5-14B bf16)在 32GB 卡需要較高 mem-util + 較短 ctx 才塞得下
    gpu_mem_util: float = 0.70
    max_model_len: int = 4096


def _expand_queries(exp: expand.Expander, queries: dict[str, str]) -> dict[str, str]:
    if isinstance(exp, expand.CachedExpander):
        return {qid: exp.expand_query_for_qid(qid, q) for qid, q in queries.items()}
    return {qid: exp.expand_query(q) for qid, q in queries.items()}


def _ensemble_run(index, queries, k, expanders, stage2: Stage2Config):
    """依 Stage2 設定融合多 model 的 expansion, 回傳融合後的 run(dict[qid->{doc->score}])。"""
    if stage2.mode == "llm":
        from qe import stage2 as s2

        fused_queries = s2.fuse_llm(
            queries, expanders, model=stage2.model, gpu=stage2.gpu, dataset=stage2.dataset,
            gpu_mem_util=stage2.gpu_mem_util, max_model_len=stage2.max_model_len,
        )
        return index.search(fused_queries, k=k)
    runs = [index.search(_expand_queries(e, queries), k=k) for e in expanders]
    return ensemble.rrf(runs, top_k=k)


def run_b0(index, queries, qrels, k):
    return metrics.score(qrels, index.search(queries, k=k))


def run_m1(index, queries, qrels, k, expanders):
    exp = expanders[0]
    eq = _expand_queries(exp, queries)
    return metrics.score(qrels, index.search(eq, k=k))


def run_m2(index, queries, qrels, k, expanders, stage2: Stage2Config):
    fused = _ensemble_run(index, queries, k, expanders, stage2)
    return metrics.score(qrels, fused)


def run_m5(index, queries, qrels, k, expanders, dev_queries, dev_qrels, stage2: Stage2Config):
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
    fused = _ensemble_run(index, queries, k, optimized, stage2)
    return metrics.score(qrels, fused)


def run_ir(dataset, methods, k, limit, models, local_gpu, stage1_models, stage2: Stage2Config):
    """跑 IR 方法,回傳 {method: {metric: value}}。CLI(main)與彙整 script 共用。"""
    print(f"載入 {dataset} ...")
    ds = load_ir(dataset)
    print(" ", ds.summary())

    queries = ds.queries
    if limit:
        queries = dict(list(queries.items())[:limit])

    print("建 BM25 index ...")
    index = BM25Index(ds.corpus)

    if local_gpu >= 0:
        from qe import stage1
        local_models = [m for m in stage1_models.split(",") if m] or stage1.DEFAULT_MODELS
        print(f"Stage1: 本地 vLLM 在 GPU {local_gpu} 跑 {local_models}")
        paths = stage1.run_stage1(dataset, local_gpu, local_models, limit=limit)
        expanders = expand.load_cached_expanders(paths)
    else:
        model_ids = [m for m in models.split(",") if m] or None
        expanders = expand.build_expanders(model_ids)
    print(f"expanders: {[e.name for e in expanders]}")
    if stage2.mode == "llm":
        where = f"GPU {stage2.gpu}" if stage2.gpu >= 0 else "mock(無 GPU)"
        print(f"Stage2: llm 融合 ({stage2.model or 'DEFAULT_FUSE_MODEL'}) @ {where}")
    else:
        print("Stage2: rrf 結果層融合")

    results: dict[str, dict[str, float]] = {}
    for m in methods:
        print(f"\n>>> {m}")
        if m == "B0":
            results[m] = run_b0(index, queries, ds.qrels, k)
        elif m == "M1":
            results[m] = run_m1(index, queries, ds.qrels, k, expanders)
        elif m == "M2":
            results[m] = run_m2(index, queries, ds.qrels, k, expanders, stage2)
        elif m == "M5":
            # 簡化: 用同一份 queries 當 dev (正式實驗請改成獨立 dev split)
            results[m] = run_m5(index, queries, ds.qrels, k, expanders, queries, ds.qrels, stage2)
        else:
            print(f"  (未知方法 {m}, 跳過)")
            continue
        print("  ", results[m])
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beir/scifact/test")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--methods", default="B0,M1,M2,M5")
    ap.add_argument("--limit", type=int, default=0, help="只取前 N 個 query (除錯用)")
    ap.add_argument("--models", default="", help="逗號分隔的 Anthropic model id; 留空用 mock")
    ap.add_argument("--local-gpu", type=int, default=-1,
                    help="設了就用本地 open source model 跑 Stage1(在該 GPU),expander 改用 vLLM 快取")
    ap.add_argument("--stage1-models", "--local-models", dest="stage1_models", default="",
                    help="Stage1 expansion 的本地 HF model id(逗號分隔);留空用 qe.stage1.DEFAULT_MODELS")
    ap.add_argument("--stage2-mode", choices=["rrf", "llm"], default="rrf",
                    help="Stage2 融合方式: rrf(結果層,預設) 或 llm(用模型做文字層融合)")
    ap.add_argument("--stage2-model", default="",
                    help="Stage2 llm 融合用的本地 HF model id;留空用 qe.stage2.DEFAULT_FUSE_MODEL")
    ap.add_argument("--stage2-gpu", type=int, default=None,
                    help="Stage2 llm 融合跑在哪張 GPU;留空沿用 --local-gpu")
    ap.add_argument("--stage2-gpu-mem-util", type=float, default=0.70,
                    help="Stage2 融合模型的 vLLM gpu_memory_utilization;大模型(14B bf16)需調高")
    ap.add_argument("--stage2-max-model-len", type=int, default=4096,
                    help="Stage2 融合模型的 max_model_len;塞不下時可調短(融合輸入不長)")
    args = ap.parse_args()

    stage2 = Stage2Config(
        mode=args.stage2_mode,
        model=args.stage2_model,
        gpu=args.stage2_gpu if args.stage2_gpu is not None else args.local_gpu,
        dataset=args.dataset,
        gpu_mem_util=args.stage2_gpu_mem_util,
        max_model_len=args.stage2_max_model_len,
    )

    methods = args.methods.split(",")
    results = run_ir(args.dataset, methods, args.k, args.limit, args.models,
                     args.local_gpu, args.stage1_models, stage2)

    print("\n===== 對照表 =====")
    metric_names = metrics.DEFAULT_METRICS
    print(f"{'method':8}" + "".join(f"{mn:>14}" for mn in metric_names))
    for m, sc in results.items():
        print(f"{m:8}" + "".join(f"{sc.get(mn, 0):>14.4f}" for mn in metric_names))


if __name__ == "__main__":
    main()
