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
