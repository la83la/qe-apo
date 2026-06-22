"""Ensemble: 把多個 model 的 expansion 合起來。

兩種層級 (對應 EXPERIMENT_PLAN 的 A / B):
    fuse_text   : 文字層融合 — 把多份 expansion 文字串接成一個 query (再做一次檢索)
    rrf         : 結果層融合 — 多份 ranked list 用 Reciprocal Rank Fusion 合併 (主力)
"""
from __future__ import annotations


def fuse_text(expansions: list[str], weights: list[float] | None = None) -> str:
    """文字層融合。weights 用整數倍重複近似加權 (BM25 對 term 頻次敏感)。"""
    if weights is None:
        return " ".join(expansions)
    parts: list[str] = []
    for text, w in zip(expansions, weights):
        parts.extend([text] * max(1, round(w)))
    return " ".join(parts)


def rrf(
    runs: list[dict[str, dict[str, float]]],
    weights: list[float] | None = None,
    k: int = 60,
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    """Reciprocal Rank Fusion。

    runs: 多個 run, 每個是 dict[qid -> dict[doc_id -> score]]。
    回傳融合後同格式的 run。weights 可被 APO 優化。
    """
    if weights is None:
        weights = [1.0] * len(runs)
    qids: set[str] = set()
    for r in runs:
        qids.update(r.keys())

    fused: dict[str, dict[str, float]] = {}
    for qid in qids:
        acc: dict[str, float] = {}
        for run, w in zip(runs, weights):
            ranked = sorted(run.get(qid, {}).items(), key=lambda x: x[1], reverse=True)
            for rank, (doc_id, _score) in enumerate(ranked, start=1):
                acc[doc_id] = acc.get(doc_id, 0.0) + w * (1.0 / (k + rank))
        fused[qid] = dict(sorted(acc.items(), key=lambda x: x[1], reverse=True)[:top_k])
    return fused
