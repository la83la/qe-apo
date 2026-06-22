# Query Expansion + Ensemble + APO — 實驗計畫

## 1. 方法總覽

```
query
  │
  ├─[Stage 1: Multi-model Expansion]───────────────┐
  │   model_A ─APO_A─▶ expansion_A                  │
  │   model_B ─APO_B─▶ expansion_B                  │  每個 model 有自己被 APO 過的 prompt
  │   model_C ─APO_C─▶ expansion_C                  │
  │                                                 │
  ├─[Stage 2: Ensemble]────────────────────────────┘
  │   (A) 文字層融合: 合併 expansion 文字 → 一次檢索
  │   (B) 結果層融合: 各自檢索 → RRF 合併 ranked lists   ◀ 主力、最 robust
  │   ─APO_ensemble─▶ (融合權重 / 融合用 prompt 也可被優化)
  │
  └─▶ retrieval (BM25 / dense) ─▶ ranked docs ─▶ (QA: reader) ─▶ answer
```

兩個 task 共用同一套 expansion，下游只換評估:
- **IR task**: 量 ranked list 品質 (nDCG@10, Recall@100, MRR@10)
- **QA task**: 量最終答案 (EM, F1) 或 retrieval-for-QA 的 answer-recall@k

## 2. Reward signal（APO 的優化目標）

| 階段 | reward 用什麼 | 備註 |
|---|---|---|
| Stage1 各 model APO | 該 model 單獨 expansion 後的 nDCG@10 (IR) / answer-F1 (QA) | dev set 上算 |
| Stage2 ensemble APO | 融合後的 nDCG@10 / F1 | **這才是 pipeline 真正目標** |
| (進階) 聯合優化 | 用 Stage2 reward 反傳影響 Stage1 prompt | DSPy MIPROv2 可做 pipeline-level |

> 注意 credit assignment：Stage1 各自最佳 ≠ ensemble 後最佳。實驗中要保留「分階段優化」vs「聯合優化」的對照。

## 3. 實驗矩陣

方法（rows） × dataset（cols） × metric，每格報 dev/test。

### 方法軸（baseline → 完整方法 → ablation）

| # | 方法 | 類別 | 想證明的事 |
|---|---|---|---|
| B0 | BM25（無 expansion） | floor baseline | 地板 |
| B1 | Dense retriever（如 Contriever/E5）無 expansion | floor baseline | 第二地板 |
| B2 | RM3 / Rocchio（傳統 PRF） | 經典 baseline | 非 LLM 對照 |
| B3 | Query2Doc（單 model，固定 prompt） | LLM 競品 | SOTA-ish 對照 |
| B4 | HyDE（單 model，固定 prompt） | LLM 競品 | SOTA-ish 對照 |
| B5 | **ExpandSearch**（RL 訓練單一模型生多變體，QA only） | LLM 競品（RL 路線） | training-based 對照；見 §7.3 |
| B6 | **Exp4Fuse**（原 query vs 單模型擴寫，modified RRF 融合） | LLM 競品（fusion 路線） | 與本方法 Stage2 最像；見 §7.2 |
| M1 | 單一最佳 model + APO（無 ensemble） | ablation | ensemble 的增益 |
| M2 | 多 model + RRF（**無 APO**） | ablation | APO 的增益 |
| M3 | 多 model + ensemble + 人工 prompt | ablation | APO vs manual |
| M4 | 多 model + ensemble + 只 Stage1 APO | ablation | Stage2 APO 的增益 |
| **M5** | **多 model + ensemble + 全階段 APO（本方法）** | **proposed** | 最終 |
| M6 | M5 但聯合優化 (DSPy) | proposed+ | credit assignment |

關鍵對照：
- M5 − M1 → ensemble 貢獻
- M5 − M2 → APO 貢獻
- M5 − M3 → 自動 vs 人工 prompt
- M5 − M4 → Stage2 APO 是否必要
- M5 vs B3/B4 → 打不打得贏現有 LLM 方法
- **M5 vs (B0+RRF of B3,B4)** → 你的 ensemble 是否只是把現成方法 RRF 一下
- **M5 vs B5（ExpandSearch）** → training-free APO ensemble vs RL single-model 的路線對比（只在 QA task 比；ExpandSearch 不做 IR ranking metric）
- **M5 vs B6（Exp4Fuse）** → 最關鍵對照：兩者都是「LLM 擴寫 + RRF」，差別只在本方法用**多異質模型 + APO**。贏 B6 才能證明多模型/APO 的價值，而非 RRF 本身；M2 vs B6 可單獨 isolate「多模型」貢獻、M5 vs M2 isolate「APO」貢獻

### Dataset 軸

**IR**
- BEIR / SciFact（小，5k docs，適合快速 iterate）← 開發用
- BEIR / FiQA、NFCorpus、TREC-COVID（領域多樣）
- MS MARCO passage + TREC DL19/20（大規模、人工判定）

**QA**
- Natural Questions (NQ-open)
- TriviaQA
- HotpotQA（multi-hop，expansion 最能展現價值）
- 你已備的 MS MARCO v1.1 QA 子集（reading comprehension 風格）

> NQ / HotpotQA / MS MARCO 同時有 IR 版與 QA 版 → 拿來當「兩個 task 共用 backbone」最乾淨。

### Metric 軸

- IR: **nDCG@10**（主）, Recall@100, MRR@10
- QA: **EM / F1**（主）, answer-Recall@k（檢索面）

## 4. 資料切分與防過擬合

- train / dev / test 嚴格分開；**APO 只看 dev，數字只報 test**。
- APO 每個方法跑 ≥3 seeds，報 mean ± std。
- 報 LLM 呼叫次數 / 成本（APO 很貴，這是審稿會問的）。
- 顯著性檢定：對 per-query metric 做 paired t-test 或 bootstrap。

## 5. 風險與緩解

| 風險 | 緩解 |
|---|---|
| APO overfit dev set | 嚴格 test-only 報數、多 seed |
| ensemble 只是 RRF 既有方法 | 加 M5 vs RRF(B3,B4) 對照 |
| 多 model 成本爆炸 | 先在 SciFact 等小集 iterate，定案再上大集 |
| credit assignment 不清 | 保留 M4/M6 對照 |
| 各 model expansion 高度重疊 → ensemble 無增益 | 量 expansion 的多樣性（如 distinct-n / 兩兩 Jaccard） |

## 6. 落地順序

1. 先把 B0（BM25）+ metrics 在 SciFact 跑通 ← 已可跑（見 `qe/run.py`）
2. 接上單 model expansion（M1 雛形）
3. 多 model + RRF（M2）
4. 套 APO（M5）
5. 換大 dataset、補 baseline、跑 QA task

## 7. Related Work

本方法 = **多模型 (inter-model) expansion × rank-level ensemble × 全階段 APO**。下面依主題盤點，每篇標出與本方法的關係。**核心 gap：沒有任何前作同時做到「跨異質模型 expansion」+「rank-level 融合」+「兩階段 APO」這三者的交集。**

### 7.1 生成式 query expansion（pre-LLM → LLM）

- **GAR** — Generation-Augmented Retrieval for Open-Domain QA (Mao et al., ACL 2021, arXiv:2009.08553)。用 fine-tuned seq2seq (BART) 生成 query context（答案 / 含答案句 / passage title）擴展 query 再 BM25。**生成式擴展的前身**，但靠監督式 fine-tune、單一生成器、無 LLM zero-shot、無 ensemble。
- **Query2Doc** (Wang et al., EMNLP 2023, arXiv:2303.07678)。LLM 生 pseudo-doc，原 query 重複數次接上去。→ 本 plan 的 **baseline B3**，也是本方法 `expand_query` 的預設形態。
- **Query Expansion by Prompting LLMs** (Jagerman et al., Google, 2023, arXiv:2305.03653)。系統比較 zero-shot / few-shot / **CoT** prompt，發現 CoT 最強（逐步拆解產生大量相關 term）。MS-MARCO + BEIR 上勝傳統 PRF。→ baseline B3 的 prompt 變體來源；**它證明「prompt 形態」對 QE 影響大 → 正是本方法用 APO 自動搜尋 prompt 的動機**。

### 7.2 多生成 + 融合 / 去噪（與本方法最接近的一支）

- **MuGI** — Exploring the Best Practices of QE with LLMs / Multi-Text Generation Integration (Zhang et al., EMNLP Findings 2024, arXiv:2401.06311)。**單一 LLM** 生**多個** pseudo-reference，用 adaptive reweighting（避免 query 被 pseudo-doc 稀釋）+ feature pooling + query calibration 整合；sparse/dense 皆可，BM25+MuGI 在 TREC DL 勝強 dense retriever。**差異**：MuGI 的「多」是同一模型的多次生成（intra-model）且融合在 **query/feature 層**；本方法的「多」是**跨不同模型**且融合在 **rank 層 (RRF)**，且 prompt 靠 APO 而非手調。
- **Exp4Fuse** (ACL Findings 2025, arXiv:2506.04760)。**與本方法 Stage2 機制最像**：原 query 與 LLM 擴寫 query 各自用 sparse retriever 檢索，再用 **modified RRF** 融合兩條 ranked list。**差異**：Exp4Fuse 只有「原 vs 擴寫」**兩條路、單一 LLM、固定 prompt**；本方法是**多個異質模型 × N 條路 + APO**。→ Exp4Fuse 應列為**強對照**，本方法要證明「多模型 + APO」勝過「原+單模型擴寫的 RRF」（即 plan 中 M5 vs RRF(B3,B4) 的加強版）。
- **GOLFer** (ACL Findings 2025, arXiv:2506.04762)。針對**小模型**（如 LLaMA-3-8B）擴展易幻覺的問題，加 hallucination filter（刪非事實/不一致句）+ combiner（weight vector 平衡 query 與生成內容）。**差異**：GOLFer 重點在**單模型去噪**，無跨模型 ensemble、無 APO。→ 它的 filter 可當本方法 Stage1 後的**可選去噪模組**借用。

### 7.3 學習 / RL-based expansion

- **ExpandSearch** — Train your LLM for QE with RL (Zhao, Yu, Xu, NVIDIA × PSU, 2025, arXiv:2510.10009)。**單一**模型用 **RL (PPO)** 訓練一次生 n=3 變體（paraphrase / 子問題 / keyword），reward = EM + 格式 (λ=0.2)；"**expand-then-squeeze**" 把檢索 chunks 丟給 frozen squeezer LLM (LLaMA-4-17B) 蒸餾再給 reader；agentic 多輪。只做 QA（EM），7 benchmark（NQ/TriviaQA/PopQA + HotpotQA/2Wiki/MuSiQue/Bamboogle），平均 +4.4% EM。→ baseline **B5**（見下表）。
- (相關) **LLM-QE** (arXiv:2502.17057, 2025)：用 DPO 把 LLM 的擴展對齊 ranking preference。同屬「訓練式對齊」路線，與本方法 training-free 形成對比，可一句帶過。

**ExpandSearch vs 本方法**（最關鍵的路線對比）：

| 面向 | ExpandSearch | 本方法 |
|---|---|---|
| 多樣性來源 | 單模型生多變體 (intra-model) | 多個異質模型 (inter-model) |
| 優化對象 | 改**權重**（RL/PPO） | 改 **prompt**（APO，training-free） |
| 前提/成本 | 要可訓練 open weights + RL infra + GPU | model-agnostic，**API 閉源也能用** |
| Stage2 聚合 | squeeze（context-level 蒸餾） | RRF（rank-level 融合） |
| Task | QA / EM only | **IR（nDCG…）+ QA 兩者** |

### 7.4 Automatic Prompt Optimization（本方法的核心工具）

- **A Systematic Survey of APO Techniques** (Ramnath et al., EMNLP 2025, arXiv:2502.16923)。給 APO 形式化定義 + 5-part 統一框架。→ 本方法 APO 模組的**設計依據與用語對齊**。
- **APO via Heuristic Search: A Survey** (Cui et al., ACL Findings 2025, arXiv:2502.18746)。依「在哪優化 / 優化什麼 / 準則 / 產生算子 / 搜尋演算法」分類（OPRO / ProTeGi / EvoPrompt 等）。→ 選 APO 演算法的依據（`qe/apo.py` 之後接 DSPy MIPROv2 或 OPRO）。

### 7.5 Survey

- **Query Expansion in the Age of Pre-trained and LLMs: A Comprehensive Survey** (2025, arXiv:2509.07794)。沿四個設計維度組織 QE：擴展注入位置、與 corpus evidence 的 grounding、如何學習/對齊、是否納入結構化知識。→ 本方法定位用：屬「decoder-only LLM zero-shot reasoning-driven expansion + instruction-tuned format adherence」，創新在**跨模型 + rank 融合 + APO** 的組合維度。

### 7.6 本方法的 novelty gap（一句話）

> 7.2 證明「多生成 + 融合」有效但停在**單模型**（MuGI/Exp4Fuse）或**去噪**（GOLFer）；7.3 用 **RL 改權重**換取 intra-model 多樣性（ExpandSearch）。**沒有人把「跨異質模型的 inter-model 多樣性」與「training-free 的兩階段 APO」結合，並在 IR + QA 兩個 task 上用 rank-level ensemble 驗證** —— 這就是本方法卡的位。
