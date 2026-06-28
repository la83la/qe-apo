"""載入 QA dataset(MS MARCO v1.1 RC 風格)成統一格式。

每題自帶一小池候選 passages(~10 段),其中 is_selected=1 的含答案。
QA task 的檢索就在「每題自己的候選池」內做(per-query pool),取 top-k 餵 reader。

回傳 list[QAExample]:
    qid      : query_id(str)
    question : 問題文字
    passages : list[str]  該題的候選 passages(順序即原始順序)
    answers  : list[str]  金答案(可能為空或 ["No Answer Present."])
    selected : list[int]  與 passages 對齊;1 表示該段含答案(評估/分析用)
"""
from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_QA_DATASET = "Lala8383/ms-marco-qa-10k"


@dataclass
class QAExample:
    qid: str
    question: str
    passages: list[str]
    answers: list[str]
    selected: list[int] = field(default_factory=list)

    @property
    def has_answer(self) -> bool:
        """是否為可答題(有非空、非 'No Answer Present.' 的金答案)。"""
        return any(a.strip() and a.strip() != "No Answer Present." for a in self.answers)


def load_qa(name: str = DEFAULT_QA_DATASET, split: str = "validation", limit: int = 0) -> list[QAExample]:
    """用 HF datasets 載入 MS MARCO v1.1 QA 子集。

    split: train / validation / test。test 通常無金答案,評估請用 validation。
    limit: 只取前 N 題(除錯/smoke test 用)。
    """
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    out: list[QAExample] = []
    for ex in ds:
        passages = ex["passages"]["passage_text"]
        selected = ex["passages"].get("is_selected", [0] * len(passages))
        out.append(
            QAExample(
                qid=str(ex["query_id"]),
                question=ex["query"],
                passages=list(passages),
                answers=list(ex["answers"]),
                selected=list(selected),
            )
        )
    return out


def summary(examples: list[QAExample]) -> str:
    n = len(examples)
    n_ans = sum(1 for e in examples if e.has_answer)
    avg_p = sum(len(e.passages) for e in examples) / max(1, n)
    return f"{n} questions ({n_ans} answerable), avg {avg_p:.1f} passages/q"


if __name__ == "__main__":
    exs = load_qa(limit=5)
    print(summary(exs))
    e = exs[0]
    print("sample:", e.qid, "::", e.question)
    print("  #passages:", len(e.passages), "  gold:", e.answers[:1])
