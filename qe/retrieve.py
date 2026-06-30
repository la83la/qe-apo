"""BM25 檢索 (bm25s, 純 Python, 不需 Java)。

用法:
    idx = BM25Index(corpus)
    runs = idx.search({qid: query_text, ...}, k=100)
    # runs: dict[qid -> dict[doc_id -> score]]  (ranx run 格式)

大 corpus(如 MS MARCO 8.8M passages)建索引很貴, 用 load_or_build 把索引快取到
磁碟, 之後 dev / dl19 / dl20 共用同一份索引就不必重建:
    idx = BM25Index.load_or_build(corpus, "cache/bm25_index/msmarco-passage")

BM25 參數預設對齊 Pyserini 在 MS MARCO passage 的設定(Lucene, k1=0.9, b=0.4),
方便與用 Pyserini 的論文(如 Exp4Fuse)做 baseline 比較。
"""
from __future__ import annotations

import json
from pathlib import Path

import bm25s


class BM25Index:
    def __init__(
        self,
        corpus: dict[str, str],
        stemmer: str = "english",
        method: str = "lucene",
        k1: float = 0.9,
        b: float = 0.4,
        _skip_index: bool = False,
    ):
        self.doc_ids = list(corpus.keys())
        try:
            import Stemmer

            self._stemmer = Stemmer.Stemmer(stemmer)
        except Exception:
            self._stemmer = None
        if _skip_index:
            self.retriever = None
            return
        texts = [corpus[d] for d in self.doc_ids]
        tokens = bm25s.tokenize(texts, stemmer=self._stemmer, show_progress=False)
        self.retriever = bm25s.BM25(method=method, k1=k1, b=b)
        self.retriever.index(tokens, show_progress=False)

    # ---------- 索引磁碟快取(大 corpus 用) ----------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.retriever.save(str(path))
        with open(path / "doc_ids.json", "w") as f:
            json.dump(self.doc_ids, f)

    @classmethod
    def load(cls, path: str | Path, stemmer: str = "english") -> "BM25Index":
        path = Path(path)
        obj = cls({}, stemmer=stemmer, _skip_index=True)
        obj.retriever = bm25s.BM25.load(str(path), load_corpus=False)
        with open(path / "doc_ids.json") as f:
            obj.doc_ids = json.load(f)
        return obj

    @classmethod
    def load_or_build(
        cls,
        corpus: dict[str, str],
        path: str | Path,
        stemmer: str = "english",
        method: str = "lucene",
        k1: float = 0.9,
        b: float = 0.4,
    ) -> "BM25Index":
        path = Path(path)
        if (path / "doc_ids.json").exists():
            print(f"[bm25] 載入已快取索引: {path}")
            return cls.load(path, stemmer=stemmer)
        print(f"[bm25] 建索引({len(corpus)} docs) -> 快取到 {path} ...")
        obj = cls(corpus, stemmer=stemmer, method=method, k1=k1, b=b)
        obj.save(path)
        return obj

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
