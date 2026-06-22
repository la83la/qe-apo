"""Automatic Prompt Optimization (APO) 迴圈。

這裡放一個「方法無關」的最小迭代式 optimizer:
    1. 從目前最佳 prompt 產生若干變體 (mutate)
    2. 在 dev set 上用 reward_fn 評分
    3. 留最佳, 反覆

預設 mutate 是「把候選 prompt 句庫挑著組合」的佔位實作; 接上 Claude 後
可改成讓 LLM 依 dev 表現改寫 prompt (即 ProTeGi / OPRO / APE 的作法),
或整段換成 DSPy 的 MIPROv2。介面刻意保持簡單以便替換。

reward_fn(prompt) -> float : 給一個 prompt, 回 dev set 上的指標 (越大越好)。
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

# 候選指令片段; LLM-based mutate 上線後可棄用
_INSTRUCTION_BANK = [
    "Write a short passage that answers the following query.",
    "Generate a detailed, factual passage relevant to the query.",
    "Produce expansion terms and a concise passage for the query.",
    "Write an expert-level passage with precise domain terminology.",
    "Answer the query with key entities, synonyms, and related concepts.",
]
_SUFFIX_BANK = [
    "Use precise, topic-relevant terminology.",
    "Include specific technical terms.",
    "Be concise but information-dense.",
    "Focus on terms a relevant document would contain.",
]


@dataclass
class APOResult:
    best_prompt: str
    best_score: float
    history: list[tuple[str, float]]


def _mutate(_base: str, rng: random.Random) -> str:
    instr = rng.choice(_INSTRUCTION_BANK)
    suffix = rng.choice(_SUFFIX_BANK)
    return f"{instr} {suffix}\n\nQuery: {{query}}\n\nPassage:"


def optimize(
    reward_fn: Callable[[str], float],
    init_prompt: str,
    rounds: int = 4,
    candidates_per_round: int = 3,
    seed: int = 0,
) -> APOResult:
    rng = random.Random(seed)
    best_prompt = init_prompt
    best_score = reward_fn(init_prompt)
    history = [(init_prompt, best_score)]

    for _ in range(rounds):
        for _ in range(candidates_per_round):
            cand = _mutate(best_prompt, rng)
            s = reward_fn(cand)
            history.append((cand, s))
            if s > best_score:
                best_prompt, best_score = cand, s
    return APOResult(best_prompt=best_prompt, best_score=best_score, history=history)
