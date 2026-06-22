"""qe: query expansion + ensemble + APO 實驗骨架。

模組:
    data      載入 IR dataset (corpus / queries / qrels)
    retrieve  BM25 檢索 (bm25s, 純 Python, 不需 Java)
    expand    multi-model query expansion (provider 可插拔; 無 API key 時用 mock)
    ensemble  RRF 結果層融合 + 文字層融合
    metrics   nDCG / Recall / MRR (ranx)
    apo       automatic prompt optimization 迴圈 (簡單實作, 之後可換 DSPy)
    run       end-to-end driver / baselines
"""
