# qe-apo — Multi-model Query Expansion + Ensemble + APO

Research scaffold: **Stage1** performs query expansion with multiple (heterogeneous)
models, **Stage2** applies rank-level ensemble (RRF) over the results, and **APO
(automatic prompt optimization)** is layered onto each stage. Both IR and QA tasks
are evaluated.

See [`EXPERIMENT_PLAN.md`](EXPERIMENT_PLAN.md) for the full experimental design,
baselines, ablations, and related work.

## Package layout `qe/`

| Module | Purpose |
|---|---|
| `data.py` | Load IR datasets (ir_datasets / BEIR), normalized into corpus/queries/qrels |
| `retrieve.py` | BM25 retrieval (`bm25s`, pure Python, no Java required) |
| `expand.py` | Multi-model expansion; pluggable providers (Anthropic / local vLLM cache / mock) |
| `ensemble.py` | RRF result-level fusion + text-level fusion |
| `metrics.py` | nDCG / Recall / MRR (ranx) |
| `apo.py` | APO loop (simple implementation; interface swappable with DSPy / OPRO) |
| `stage1.py` | Stage1 orchestrator: runs multiple local models in sequence on the assigned GPU, with result caching |
| `vllm_worker.py` | vLLM expansion worker for a single local model (subprocess) |
| `run.py` | End-to-end driver; runs the baseline-vs-method comparison table |

## Environment

```bash
python3 -m venv merge_venv && source merge_venv/bin/activate
uv pip install bm25s PyStemmer ranx ir_datasets anthropic datasets
# Local open-source models (vLLM) — Blackwell (RTX 5090) requires cu130:
uv pip install -U vllm --torch-backend=cu130
```

## Usage

```bash
# 1) Baseline + mock expander (no GPU; validate the pipeline first)
python -m qe.run --dataset beir/scifact/test --methods B0,M2

# 2) Stage1 with local open-source models (GPU 1) + RRF ensemble
CUDA_DEVICE_ORDER=PCI_BUS_ID python -m qe.run \
    --dataset beir/scifact/test --local-gpu 1 --methods B0,M2 --limit 50
```

### Notes for Blackwell (RTX 5090, sm_120) + CUDA 13

`qe/vllm_worker.py` already applies the following settings (before importing vllm):
- torch must be **cu130** (not cu128, otherwise `libcudart.so.13` is missing)
- `CUDA_DEVICE_ORDER=PCI_BUS_ID` (align with nvidia-smi index on mixed-GPU machines)
- `VLLM_ATTENTION_BACKEND=FLASH_ATTN`, `VLLM_USE_FLASHINFER_SAMPLER=0` (FlashInfer's JIT check misfires on this arch)
- `gpu_memory_utilization=0.7` (leave headroom on shared GPUs)

## Preliminary results (50 SciFact queries)

| method | nDCG@10 | Recall@100 | MRR@10 |
|---|---|---|---|
| B0 (BM25, no expansion) | 0.808 | 0.960 | 0.786 |
| M2 (3-model expansion + RRF) | 0.842 | 0.960 | 0.820 |

Models: `Qwen2.5-7B-Instruct` / `Meta-Llama-3.1-8B-Instruct` / `mistral-7b-instruct-v0.3`.
