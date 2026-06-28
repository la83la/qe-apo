"""QA 評估指標 — SQuAD 風格 EM / F1。

normalize: 小寫、去標點、去冠詞(a/an/the)、壓空白,再比對。
一題有多個金答案時取 max(對任一金答案最好的分數)。
"""
from __future__ import annotations

import re
import string
from collections import Counter


def normalize(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def _em(pred: str, gold: str) -> float:
    return float(normalize(pred) == normalize(gold))


def _f1(pred: str, gold: str) -> float:
    p_toks = normalize(pred).split()
    g_toks = normalize(gold).split()
    if not p_toks or not g_toks:
        # 兩邊都空才算 1(空答案對空金答案)
        return float(p_toks == g_toks)
    common = Counter(p_toks) & Counter(g_toks)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(p_toks)
    recall = n_same / len(g_toks)
    return 2 * precision * recall / (precision + recall)


def score_one(pred: str, golds: list[str]) -> tuple[float, float]:
    """回傳該題的 (EM, F1) — 對所有金答案取 max。"""
    golds = [g for g in golds if g.strip()] or [""]
    em = max(_em(pred, g) for g in golds)
    f1 = max(_f1(pred, g) for g in golds)
    return em, f1


def score(preds: dict[str, str], golds: dict[str, list[str]]) -> dict[str, float]:
    """preds/golds 以 qid 對齊,回傳 {'em':, 'f1':} 平均(只算兩邊都有的 qid)。"""
    common = [q for q in golds if q in preds]
    if not common:
        return {"em": 0.0, "f1": 0.0}
    ems, f1s = [], []
    for q in common:
        em, f1 = score_one(preds[q], golds[q])
        ems.append(em)
        f1s.append(f1)
    return {"em": sum(ems) / len(ems), "f1": sum(f1s) / len(f1s)}
