"""BM25 檢索 (bm25s, 純 Python, 不需 Java)。

用法:
    idx = BM25Index(corpus)
    runs = idx.search({qid: query_text, ...}, k=100)
    # runs: dict[qid -> dict[doc_id -> score]]  (ranx run 格式)
"""
from __future__ import annotations

import bm25s


class BM25Index:
    def __init__(self, corpus: dict[str, str], stemmer: str = "english"):
        self.doc_ids = list(corpus.keys())
        texts = [corpus[d] for d in self.doc_ids]
        try:
            import Stemmer

            self._stemmer = Stemmer.Stemmer(stemmer)
        except Exception:
            self._stemmer = None
        tokens = bm25s.tokenize(texts, stemmer=self._stemmer, show_progress=False)
        self.retriever = bm25s.BM25()
        self.retriever.index(tokens, show_progress=False)

    def search(self, queries: dict[str, str], k: int = 100) -> dict[str, dict[str, float]]:
        qids = list(queries.keys())
        q_tokens = bm25s.tokenize(
            [queries[q] for q in qids], stemmer=self._stemmer, show_progress=False
        )
        results, scores = self.retriever.retrieve(
            q_tokens, k=min(k, len(self.doc_ids)), show_progress=False
        )
        runs: dict[str, dict[str, float]] = {}
        for i, qid in enumerate(qids):
            runs[qid] = {
                self.doc_ids[results[i, j]]: float(scores[i, j])
                for j in range(results.shape[1])
            }
        return runs
