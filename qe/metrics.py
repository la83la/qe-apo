"""IR 評估指標 (ranx)。"""
from __future__ import annotations

from ranx import Qrels, Run, evaluate

DEFAULT_METRICS = ["ndcg@10", "recall@100", "mrr@10"]


def score(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
) -> dict[str, float]:
    metrics = metrics or DEFAULT_METRICS
    # 只評估 run 與 qrels 都有的 query
    common = {q: qrels[q] for q in qrels if q in run}
    q = Qrels(common)
    r = Run({qid: run[qid] for qid in common})
    res = evaluate(q, r, metrics)
    if isinstance(res, dict):
        return {m: float(v) for m, v in res.items()}
    return {metrics[0]: float(res)}
