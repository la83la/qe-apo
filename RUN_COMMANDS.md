# 跑實驗的指令speedsheet（IR + QA）

> 所有可調項都走命令列參數，不用改程式碼。
> 先進環境：`source merge_venv/bin/activate`（或 `source /mnt/raid0/home/lalako/MERGE/merge_venv/bin/activate`）

---

## 0. 先決：挑 GPU 與模型

- 用哪張卡：`nvidia-smi` 看哪張空，把它的 index 填到 `--local-gpu`（IR）或 `--gpu`（QA）。
- 模型要塞得下該卡 VRAM：
  - **≥24GB**：可用原始異質 7B 三人組（見下方「大卡」變體）。
  - **16GB（如 RTX 4080）**：用縮小三人組 `Qwen/Qwen2.5-3B-Instruct,unsloth/Llama-3.2-1B-Instruct,google/gemma-4-E2B-it`，reader 用 `Qwen/Qwen2.5-3B-Instruct`。
- 結果都會快取（`cache/`），同參數重跑是秒回；改 prompt/模型/limit 會自動 miss 重生成。加 `--force` 強制重跑。
- vLLM 在 Blackwell+CUDA13 的環境設定已寫死在 `qe/vllm_worker.py`（torch 須 cu130、`CUDA_DEVICE_ORDER=PCI_BUS_ID` 等）。

---

## 1. IR task（檢索，指標 nDCG@10 / Recall@100 / MRR@10）

驅動程式：`qe/run.py`。方法：`B0`(BM25) / `M1`(單模型 expansion) / `M2`(多模型+RRF) / `M5`(多模型+RRF+APO)。

```bash
# (a) 無 GPU 健檢：BM25 vs mock-ensemble（先確認 pipeline 通）
python -m qe.run --dataset beir/scifact/test --methods B0,M2 --limit 50

# (b) 真實：本地 vLLM 在 GPU 1 跑 Stage1 多模型 expansion + RRF（小卡縮小三人組）
python -m qe.run --dataset beir/scifact/test --local-gpu 1 \
    --stage1-models Qwen/Qwen2.5-3B-Instruct,unsloth/Llama-3.2-1B-Instruct,google/gemma-4-E2B-it \
    --methods B0,M1,M2 --k 100 --limit 200

# (c) 大卡（≥24GB）：原始異質 7B 三人組
python -m qe.run --dataset beir/scifact/test --local-gpu 1 \
    --stage1-models Qwen/Qwen2.5-7B-Instruct,NousResearch/Meta-Llama-3.1-8B-Instruct,unsloth/mistral-7b-instruct-v0.3 \
    --methods B0,M1,M2 --k 100

# (d) Stage2 用 LLM 文字層融合（而非 RRF）
python -m qe.run --dataset beir/scifact/test --local-gpu 1 \
    --methods B0,M2 --stage2-mode llm --stage2-model Qwen/Qwen2.5-3B-Instruct --stage2-gpu 1
```

其他可換 dataset：`beir/nfcorpus/test`、`beir/fiqa/test`、`beir/trec-covid`。

---

## 1b. 對齊 Exp4Fuse (arXiv:2506.04760) 的 in-domain 設定（為了跟它比）

論文用 **MS MARCO passage** 全 corpus（8.8M），metric 與相關門檻我們程式已自動對齊：

| dataset 代號 | ir_datasets | metric | 相關門檻 |
|---|---|---|---|
| `msmarco-dev` | `msmarco-passage/dev/small` | MRR@10, R@1k | rel≥1 |
| `dl19` | `msmarco-passage/trec-dl-2019/judged` | MAP, nDCG@10, R@1k | rel≥2 |
| `dl20` | `msmarco-passage/trec-dl-2020/judged` | MAP, nDCG@10, R@1k | rel≥2 |

- BM25 參數已設成 Pyserini 預設（Lucene, k1=0.9, b=0.4）；檢索深度用 `--k 1000`（要 R@1k）。
- TREC-DL 的 MAP/Recall/MRR 自動用 rel≥2 二值化、nDCG 用分級（trec_eval 慣例），不必手動處理。
- **索引只建一次**：三個 dataset 共用同一份 8.8M 索引，快取在 `cache/bm25_index/msmarco-passage`；
  第一個跑的會建（吃時間/記憶體），之後 dev/dl19/dl20 秒載、且跳過掃 corpus。

```bash
# (a) 先建索引 + BM25 baseline（第一次會下載 ~3GB corpus 並建 8.8M 索引, 較久）
python -m qe.run --dataset dl19 --methods B0 --k 1000

# (b) 三個 dataset 的 BM25 baseline（對齊論文 Table 1 的 BM25 列）
for D in msmarco-dev dl19 dl20; do python -m qe.run --dataset $D --methods B0 --k 1000; done

# (c) 我們的方法（本地開源 expansion + RRF）。論文開源點是 LLaMA3-8B-Instruct，
#     大卡建議用回 7B/8B 三人組才跟它同級
python -m qe.run --dataset dl19 --methods B0,M1,M2 --k 1000 --local-gpu 1 \
    --stage1-models Qwen/Qwen2.5-7B-Instruct,NousResearch/Meta-Llama-3.1-8B-Instruct,unsloth/mistral-7b-instruct-v0.3
```

**論文 BM25 baseline 參考值**（你重現時應接近）：
dev MRR@10 18.4 / R@1k 85.7；DL19 MAP 30.1 / nDCG@10 50.6 / R@1k 75.0；DL20 28.6 / 48.0 / 78.6。
（開源 LLaMA3-8B + Exp4Fuse：dev MRR@10 18.9、DL19 nDCG@10 59.7。）

> 注意：論文 Exp4Fuse 是「原始 query route + 單一 expansion route」做 RRF；我們的 M2 是「多模型 expansion」做 RRF，
> 概念相近但不完全相同。要逐項可比，之後可再加一個 Exp4Fuse 風格方法（orig+expanded 兩路 RRF）。

---

## 1c. 對齊 Exp4Fuse 的 out-of-domain 設定（7 個 BEIR 低資源資料集）

論文 Table 2 只報 **nDCG@10**（我們額外帶 recall@100 當診斷）。已註冊友善代號：

| 代號 | ir_datasets id | #queries | 語料來源 |
|---|---|---|---|
| `dbpedia` | `beir/dbpedia-entity/test` | 400 | BEIR 公開（自動下載，~640MB）|
| `fiqa` | `beir/fiqa/test` | 648 | BEIR 公開（自動下載）|
| `nq` | `beir/nq` | 3452 | BEIR 公開（自動下載，較大）|
| `touche2020` | `beir/webis-touche2020/v2` | 49 | BEIR 公開（自動下載）|
| `scifact` | `beir/scifact/test` | 300 | BEIR 公開（自動下載，最小）|
| `robust04` | `disks45/nocr/trec-robust-2004` | 249 | **需自備 TREC disks 4&5**（授權語料）|
| `news` | `wapo/v2/trec-news-2019` | 57 | **需自備 Washington Post v2**（授權語料）|

- nDCG 用分級 relevance（不受 rel_threshold 影響），對齊 trec_eval / BEIR。
- 每個 BEIR 資料集**各建一份 BM25 索引**（不共用），快取在 `cache/bm25_index/<key>`。
- **expansion 指令自動套用** Exp4Fuse Appendix A.1 的 per-dataset zero-shot 指令
  （如 SciFact=support/refute claim、Touche2020=counter argument、FiQA=financial、
  News/Robust04=news topic），不用手動指定。in-domain（dl19/dl20/msmarco-dev）仍用原 DEFAULT。

```bash
# (a) BM25 baseline：5 個可自動下載的資料集（對齊論文 Table 2 的 BM25 列）
for D in scifact fiqa touche2020 dbpedia nq; do
  python -m qe.run --dataset $D --methods B0 --k 1000
done

# (b) 我們的方法（本地開源 expansion + RRF）。大卡建議 7B/8B 三人組
python -m qe.run --dataset scifact --methods B0,M1,M2 --k 1000 --local-gpu 1 \
    --stage1-models Qwen/Qwen2.5-7B-Instruct,NousResearch/Meta-Llama-3.1-8B-Instruct,unsloth/mistral-7b-instruct-v0.3
```

**論文 BM25 baseline 參考值（nDCG@10，你重現時應接近）**：
DBPedia 31.8 / FiQA 23.6 / News 39.5 / NQ 30.6 / Robust04 40.7 / Touche2020 44.2 / SciFact 67.9。
（已驗證 SciFact 重現 = 67.6，吻合。）

> **Robust04 / News 需授權語料**：BEIR 不重發這兩個語料，ir_datasets 也只能自動下到
> queries/qrels，docs 要自備——Robust04 用 TREC disks 4&5、News 用 WaPo v2，都需向
> NIST/LDC 申請。取得後把語料放到 `~/.ir_datasets/`（disks45 / wapo）對應位置即可跑。
> 在拿到語料前，這兩個資料集會在建索引時失敗（其餘 5 個不受影響）。

---

## 2. QA task（讀答，指標 EM / F1）

驅動程式：`qe/qa.py`。方法：
`closed`(無passage,floor) / `B0`(BM25) / `M1`(單模型expansion) / `M2`(多模型+RRF) / `oracle`(金標passage,ceiling)。

檢索採 **per-query 候選池**（每題只在自己 ~8 段 passages 內重排）。

```bash
# (a) 無 GPU 健檢：mock reader，驗證 wiring + EM/F1 算得出（絕對分數無意義）
python -m qe.qa --split validation --limit 20 --methods closed,B0,oracle --reader mock

# (b) 真分數：vLLM reader 在 GPU 5（小卡縮小三人組）
#     M1/M2 會先在同張 GPU 對 QA 問題產 expansion，再用 reader 批次作答
python -m qe.qa --split validation --limit 200 --k 3 \
    --methods closed,B0,M1,M2,oracle \
    --reader vllm --reader-model Qwen/Qwen2.5-3B-Instruct --gpu 5 \
    --expand-models Qwen/Qwen2.5-3B-Instruct,unsloth/Llama-3.2-1B-Instruct,google/gemma-4-E2B-it

# (c) 大卡（≥24GB）：reader 用 7B、原始異質三人組 expansion
python -m qe.qa --split validation --limit 200 --k 3 \
    --methods closed,B0,M1,M2,oracle \
    --reader vllm --reader-model Qwen/Qwen2.5-7B-Instruct --gpu 1 \
    --expand-models Qwen/Qwen2.5-7B-Instruct,NousResearch/Meta-Llama-3.1-8B-Instruct,unsloth/mistral-7b-instruct-v0.3

# (d) 只想先看 reader 端基線（不跑 expansion，最快拿到真分數）
python -m qe.qa --split validation --limit 200 --methods closed,B0,oracle \
    --reader vllm --reader-model Qwen/Qwen2.5-3B-Instruct --gpu 5
```

可換 `--split test`（注意 test 多無金答案，評估用 `validation`）、調 `--k`（餵 reader 的 passage 數）、`--limit 0`（跑全部）。

---

## 3. 一次跑 IR + QA 並輸出 CSV（彙整）

驅動程式：`run_all.py`（repo 根目錄）。內部呼叫 `qe.run` 與 `qe.qa`，合併成一份 CSV。
IR 與 QA 的 GPU 可分開指定（`--ir-gpu` / `--qa-gpu`）。

輸出為**長格式 CSV**，欄位 `task,dataset,method,metric,value`（各 task 指標不同，長格式最乾淨；用 pandas/Excel pivot 成寬表）。

```bash
# (a) 無 GPU 健檢：確認彙整流程通（mock，分數無意義）
python run_all.py --out results.csv --limit 20 \
    --ir-methods B0,M2 --qa-methods closed,B0,oracle --reader mock

# (b) 真跑：IR Stage1 在 GPU 1、QA reader/expansion 在 GPU 5（小卡縮小三人組）
python run_all.py --out results.csv --limit 200 --k 100 --qa-k 3 \
    --ir-methods B0,M1,M2 --qa-methods closed,B0,M1,M2,oracle \
    --reader vllm --reader-model Qwen/Qwen2.5-3B-Instruct \
    --ir-gpu 1 --qa-gpu 5 \
    --expand-models Qwen/Qwen2.5-3B-Instruct,unsloth/Llama-3.2-1B-Instruct,google/gemma-4-E2B-it

# (c) 只跑其中一個：--skip-qa 或 --skip-ir
python run_all.py --out ir_only.csv --skip-qa --ir-gpu 1 \
    --ir-stage1-models Qwen/Qwen2.5-3B-Instruct,unsloth/Llama-3.2-1B-Instruct,google/gemma-4-E2B-it
```

常用旗標：`--limit`（兩 task 共用題數，0=全部）、`--k`（IR 檢索深度）、`--qa-k`（QA top-k passages）、
`--ir-stage1-models` / `--expand-models`（分別給 IR / QA 的 expansion 模型）、`--force`（忽略快取）。

---

## 4. 常見狀況

- **OOM**：模型太大塞不下該卡 → 換更小模型，或調 `qe/vllm_worker.py` 的 `--gpu-mem-util`（QA reader 預設 0.85；可在 `VLLMReader` 改）。實務上 16GB 卡別放 7B/8B。
- **gemma-4 載入失敗**：若該 vLLM 版本不支援 gemma 架構，把三人組第三個換成 `Qwen/Qwen2.5-1.5B-Instruct` 之類。
- **想重跑覆蓋快取**：加 `--force`。
- **看單題輸出**：expansion 在 `cache/expand/`，QA reader 答案在 `cache/qa/`，都是可讀 JSON。
