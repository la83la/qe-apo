"""IR 評估指標 (ranx)。

不同 dataset 用不同 metric 與相關門檻:
    MS MARCO dev : MRR@10, Recall@1000          (qrels 二元, rel>=1)
    TREC-DL 19/20: MAP, nDCG@10, Recall@1000    (qrels 分級 0-3)

TREC-DL 慣例(對齊 trec_eval / Pyserini, 也是 Exp4Fuse 用的設定):
    - nDCG 用「分級」relevance(gain 隨等級遞增)
    - MAP / Recall / MRR 等二元指標, relevance >= 2 才算相關
所以本模組對二元指標會先把 qrels 以 rel_threshold 二值化, nDCG 則用原始分級。
"""
from __future__ import annotations

from ranx import Qrels, Run, evaluate

DEFAULT_METRICS = ["ndcg@10", "recall@100", "mrr@10"]

# 用「分級」relevance 的 metric 家族(其餘視為二元指標, 需先二值化)
_GRADED_FAMILIES = {"ndcg", "dcg"}


def _family(metric: str) -> str:
    return metric.split("@", 1)[0].strip().lower()


def _evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]],
              metrics: list[str]) -> dict[str, float]:
    # 只留下「有非空 qrels」且「run 有結果」的 query, 否則 ranx 會報錯
    common = {q: qrels[q] for q in qrels if qrels.get(q) and q in run}
    if not common:
        return {m: 0.0 for m in metrics}
    q = Qrels(common)
    r = Run({qid: run[qid] for qid in common})
    res = evaluate(q, r, metrics)
    if isinstance(res, dict):
        return {m: float(v) for m, v in res.items()}
    return {metrics[0]: float(res)}


def score(
    qrels: dict[str, dict[str, int]],
    run: dict[str, dict[str, float]],
    metrics: list[str] | None = None,
    rel_threshold: int = 1,
) -> dict[str, float]:
    """評估。graded(nDCG)用原始 relevance; 二元指標用 rel>=rel_threshold 二值化。"""
    metrics = metrics or DEFAULT_METRICS
    graded = [m for m in metrics if _family(m) in _GRADED_FAMILIES]
    binary = [m for m in metrics if _family(m) not in _GRADED_FAMILIES]

    out: dict[str, float] = {}
    if graded:
        out.update(_evaluate(qrels, run, graded))
    if binary:
        bin_qrels = {
            q: {d: 1 for d, rel in rels.items() if rel >= rel_threshold}
            for q, rels in qrels.items()
        }
        out.update(_evaluate(bin_qrels, run, binary))
    # 依輸入 metric 順序回傳
    return {m: out.get(m, 0.0) for m in metrics}
