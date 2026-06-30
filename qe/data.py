"""載入 IR dataset 成統一格式。

回傳:
    corpus  : dict[doc_id -> text]
    queries : dict[query_id -> text]
    qrels   : dict[query_id -> dict[doc_id -> relevance(int)]]

預設用 BEIR/SciFact (小, ~5k docs, 適合快速 iterate)。
透過 ir_datasets 載入, 第一次會自動下載並快取。

對齊 Exp4Fuse (arXiv:2506.04760) 的 in-domain 設定, 另支援三個友善代號:
    msmarco-dev  -> msmarco-passage/dev/small           (MRR@10, R@1k)
    dl19         -> msmarco-passage/trec-dl-2019/judged  (MAP, nDCG@10, R@1k; rel>=2)
    dl20         -> msmarco-passage/trec-dl-2020/judged  (MAP, nDCG@10, R@1k; rel>=2)
這三者共用同一份 8.8M passage corpus, 用 BM25Index.load_or_build 快取索引避免重建。
"""
from __future__ import annotations

from dataclasses import dataclass, field

# 友善代號 -> (ir_datasets id, eval 設定)
# eval 設定: metrics 為 ranx metric 名; rel_threshold 為計算二元指標(map/recall/mrr)
# 時 relevance >= 此值才算相關(TREC-DL 慣例為 2; MS MARCO dev 二元 qrels 為 1)。
_REGISTRY: dict[str, tuple[str, dict]] = {
    "msmarco-dev": (
        "msmarco-passage/dev/small",
        {"metrics": ["mrr@10", "recall@1000"], "rel_threshold": 1},
    ),
    "dl19": (
        "msmarco-passage/trec-dl-2019/judged",
        {"metrics": ["map@1000", "ndcg@10", "recall@1000"], "rel_threshold": 2},
    ),
    "dl20": (
        "msmarco-passage/trec-dl-2020/judged",
        {"metrics": ["map@1000", "ndcg@10", "recall@1000"], "rel_threshold": 2},
    ),
    # --- Exp4Fuse out-of-domain: 7 個 BEIR 低資源資料集(Table 2; 只報 nDCG@10) ---
    # 額外帶 recall@100 當自家診斷。nDCG 用分級 relevance, 不受 rel_threshold 影響。
    "dbpedia": (
        "beir/dbpedia-entity/test",
        {"metrics": ["ndcg@10", "recall@100"], "rel_threshold": 1},
    ),
    "fiqa": (
        "beir/fiqa/test",
        {"metrics": ["ndcg@10", "recall@100"], "rel_threshold": 1},
    ),
    "nq": (
        "beir/nq",
        {"metrics": ["ndcg@10", "recall@100"], "rel_threshold": 1},
    ),
    "touche2020": (
        "beir/webis-touche2020/v2",
        {"metrics": ["ndcg@10", "recall@100"], "rel_threshold": 1},
    ),
    "scifact": (
        "beir/scifact/test",
        {"metrics": ["ndcg@10", "recall@100"], "rel_threshold": 1},
    ),
    # Robust04 / News(TREC-NEWS) 在 BEIR 不重發語料; 底層用 TREC 原始語料,
    # 需自備授權語料(Robust04→TREC disks 4&5; News→Washington Post v2)。
    # queries/qrels 可自 NIST 下載,但 docs 不在 ir_datasets 自動下載範圍。
    "robust04": (
        "disks45/nocr/trec-robust-2004",
        {"metrics": ["ndcg@10", "recall@100"], "rel_threshold": 1},
    ),
    "news": (
        "wapo/v2/trec-news-2019",
        {"metrics": ["ndcg@10", "recall@100"], "rel_threshold": 1},
    ),
}

# Exp4Fuse Appendix A.1 的 per-dataset zero-shot expansion 指令。
# 只對 OOD 代號套用(faithful 對齊論文); 未列者(含 in-domain dl19/dl20/msmarco-dev
# 與原始 ir_datasets id)回退 DEFAULT_EXPAND_PROMPT, 不動既有快取。
_EXPAND_INSTRUCTIONS: dict[str, str] = {
    "dbpedia": "Please write a passage to answer the question. {query}",
    "nq": "Please write a passage to answer the question. {query}",
    "fiqa": "Please write a financial article passage to answer the question. {query}",
    "news": "Please write a news passage about the topic. {query}",
    "robust04": "Please write a news passage about the topic. {query}",
    "touche2020": "Please write a counter argument for the passage. {query}",
    "scifact": "Please write a scientific paper passage to support/refute the claim. {query}",
}

# 共用同一 corpus 的 dataset 給同一個索引快取名(避免 dev/dl19/dl20 各建一次 8.8M 索引)
_CORPUS_KEY: dict[str, str] = {
    "msmarco-dev": "msmarco-passage",
    "dl19": "msmarco-passage",
    "dl20": "msmarco-passage",
}

DEFAULT_EVAL = {"metrics": ["ndcg@10", "recall@100", "mrr@10"], "rel_threshold": 1}


def resolve(name: str) -> str:
    """友善代號 -> ir_datasets id; 非代號則原樣回傳。"""
    return _REGISTRY[name][0] if name in _REGISTRY else name


def eval_config(name: str) -> dict:
    """回傳該 dataset 的 {metrics, rel_threshold}; 未登記者用 DEFAULT_EVAL。"""
    return _REGISTRY[name][1] if name in _REGISTRY else DEFAULT_EVAL


def corpus_cache_key(name: str) -> str:
    """索引磁碟快取用的 corpus 名(共用 corpus 的 dataset 回傳同一個)。"""
    return _CORPUS_KEY.get(name, resolve(name).replace("/", "_"))


def expand_instruction(name: str) -> str:
    """回傳該 dataset 的 zero-shot expansion 指令(含 {query} 佔位)。
    OOD 代號用 Exp4Fuse Appendix A.1 的專屬指令; 其餘回退 DEFAULT_EXPAND_PROMPT。"""
    from qe.expand import DEFAULT_EXPAND_PROMPT

    return _EXPAND_INSTRUCTIONS.get(name, DEFAULT_EXPAND_PROMPT)


@dataclass
class IRDataset:
    corpus: dict[str, str]
    queries: dict[str, str]
    qrels: dict[str, dict[str, int]]
    name: str = ""
    eval: dict = field(default_factory=lambda: dict(DEFAULT_EVAL))

    def summary(self) -> str:
        n_q_with_rel = sum(1 for q in self.queries if self.qrels.get(q))
        return (
            f"corpus={len(self.corpus)} docs, queries={len(self.queries)} "
            f"({n_q_with_rel} with qrels)"
        )


def load_ir(name: str = "beir/scifact/test", with_corpus: bool = True) -> IRDataset:
    """用 ir_datasets 載入。接受友善代號(msmarco-dev/dl19/dl20)或原始 id
    (beir/scifact/test, beir/nfcorpus/test, beir/fiqa/test, ...)。

    with_corpus=False: 跳過(可能很大的)corpus 掃描, 只載 queries/qrels —
    當 BM25 索引已快取在磁碟、不需重建時用, 省去掃 8.8M passages 的時間。
    """
    import ir_datasets

    ds = ir_datasets.load(resolve(name))

    corpus: dict[str, str] = {}
    if with_corpus:
        for doc in ds.docs_iter():
            # ir_datasets 的 doc 欄位依 dataset 而異; SciFact 有 title+text,
            # MS MARCO passage 只有 text。
            title = getattr(doc, "title", "") or ""
            # SciFact/NQ/MS MARCO 用 .text; TREC 系(Robust04 disks45, News wapo)用 .body
            text = getattr(doc, "text", "") or getattr(doc, "body", "") or ""
            corpus[doc.doc_id] = (title + " " + text).strip()

    queries: dict[str, str] = {q.query_id: q.text for q in ds.queries_iter()}

    qrels: dict[str, dict[str, int]] = {}
    for qrel in ds.qrels_iter():
        qrels.setdefault(qrel.query_id, {})[qrel.doc_id] = int(qrel.relevance)

    # 只保留有 qrels 的 query, 評估才有意義
    queries = {qid: t for qid, t in queries.items() if qid in qrels}
    return IRDataset(
        corpus=corpus, queries=queries, qrels=qrels, name=name, eval=eval_config(name)
    )


if __name__ == "__main__":
    ds = load_ir()
    print(ds.summary())
    qid = next(iter(ds.queries))
    print("sample query:", qid, "->", ds.queries[qid])
