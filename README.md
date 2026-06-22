# qe-apo — Multi-model Query Expansion + Ensemble + APO

研究骨架:**Stage1** 用多個（異質）model 做 query expansion，**Stage2** 對結果做
rank-level ensemble（RRF），每階段都套 **APO (automatic prompt optimization)**。
IR 與 QA 兩個 task 都評估。

完整實驗設計、baseline、ablation 與 related work 見 [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md)。

## 套件結構 `qe/`

| 模組 | 功能 |
|---|---|
| `data.py` | 載入 IR dataset（ir_datasets / BEIR），統一成 corpus/queries/qrels |
| `retrieve.py` | BM25 檢索（`bm25s`，純 Python，不需 Java） |
| `expand.py` | multi-model expansion；provider 可插拔（Anthropic / 本地 vLLM 快取 / mock） |
| `ensemble.py` | RRF 結果層融合 + 文字層融合 |
| `metrics.py` | nDCG / Recall / MRR（ranx） |
| `apo.py` | APO 迴圈（簡單實作，介面可換 DSPy / OPRO） |
| `stage1.py` | Stage1 orchestrator：依序在指定 GPU 上跑多個本地模型，結果快取 |
| `vllm_worker.py` | 單一本地模型的 vLLM expansion worker（subprocess） |
| `run.py` | end-to-end driver，跑 baseline 與本方法對照表 |

## 環境

```bash
python3 -m venv merge_venv && source merge_venv/bin/activate
uv pip install bm25s PyStemmer ranx ir_datasets anthropic datasets
# 本地開源模型（vLLM）— Blackwell (RTX 5090) 需 cu130：
uv pip install -U vllm --torch-backend=cu130
```

## 用法

```bash
# 1) baseline + mock expander（無 GPU，先驗 pipeline）
python -m qe.run --dataset beir/scifact/test --methods B0,M2

# 2) Stage1 用本地開源模型（GPU 1）+ RRF ensemble
CUDA_DEVICE_ORDER=PCI_BUS_ID python -m qe.run \
    --dataset beir/scifact/test --local-gpu 1 --methods B0,M2 --limit 50
```

### Blackwell (RTX 5090, sm_120) + CUDA 13 注意事項

`qe/vllm_worker.py` 已內建以下設定（import vllm 前）：
- torch 必須 **cu130**（非 cu128，否則缺 `libcudart.so.13`）
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`（混卡機器對齊 nvidia-smi index）
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN`、`VLLM_USE_FLASHINFER_SAMPLER=0`（FlashInfer 在此 arch JIT check 會誤判失敗）
- `gpu_memory_utilization=0.7`（共享卡留餘裕）

## 初步結果（50 題 SciFact）

| method | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| B0（BM25，無 expansion） | 0.808 | 0.960 | 0.786 |
| M2（3 模型 expansion + RRF） | 0.842 | 0.960 | 0.820 |

模型：`Qwen2.5-7B-Instruct` / `Meta-Llama-3.1-8B-Instruct` / `mistral-7b-instruct-v0.3`。
