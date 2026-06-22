"""載入 IR dataset 成統一格式。

回傳:
    corpus  : dict[doc_id -> text]
    queries : dict[query_id -> text]
    qrels   : dict[query_id -> dict[doc_id -> relevance(int)]]

預設用 BEIR/SciFact (小, ~5k docs, 適合快速 iterate)。
透過 ir_datasets 載入, 第一次會自動下載並快取。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IRDataset:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]

    def summary(self) -> str:
        n_q_with_rel = sum(1 for q in self.queries if self.qrels.get(q))
        return (
            f"corpus={len(self.corpus)} docs, queries={len(self.queries)} "
            f"({n_q_with_rel} with qrels)"
        )


def load_ir(name: str = "beir/scifact/test") -> IRDataset:
    """用 ir_datasets 載入。常用 name:
    beir/scifact/test, beir/nfcorpus/test, beir/fiqa/test, beir/trec-covid
    """
    import ir_datasets

    ds = ir_datasets.load(name)

    corpus: dict[str, str] = {}
    for doc in ds.docs_iter():
        # ir_datasets 的 doc 欄位依 dataset 而異; SciFact 有 title + text
        title = getattr(doc, "title", "") or ""
        text = getattr(doc, "text", "") or ""
        corpus[doc.doc_id] = (title + " " + text).strip()

    queries: dict[str, str] = {q.query_id: q.text for q in ds.queries_iter()}

    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    # 只保留有 qrels 的 query, 評估才有意義
    queries = {qid: t for qid, t in queries.items() if qid in qrels}
    return IRDataset(corpus=corpus, queries=queries, qrels=qrels)


if __name__ == "__main__":
    ds = load_ir()
    print(ds.summary())
    qid = next(iter(ds.queries))
    print("sample query:", qid, "->", ds.queries[qid])
