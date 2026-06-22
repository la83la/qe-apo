"""vLLM expansion worker — 一次跑「一個」open source model。

設計:每個 model 用獨立 subprocess 跑(本檔),載入 → batch 生成所有 query 的
expansion → 寫快取 JSON → 結束(完整釋放 GPU),再換下一個 model。避免 vLLM
在同一 process 內 load/unload 多模型的記憶體釋放問題。

用法(由 qe.stage1 orchestrator 呼叫,一般不手動跑):
    CUDA_VISIBLE_DEVICES=1 python -m qe.vllm_worker \
        --model Qwen/Qwen2.5-7B-Instruct \
        --dataset beir/scifact/test \
        --prompt-file prompts/default.txt \
        --out cache/expand/scifact__Qwen2.5-7B__<hash>.json \
        [--limit 50]

輸出 JSON: {query_id: expansion_passage}
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# 必須在 import vllm 之前設定:
# - PCI_BUS_ID: 混卡機器上讓 CUDA_VISIBLE_DEVICES 對齊 nvidia-smi 的 index
# - FLASH_ATTN: 避開在此 Blackwell(sm_120)+CUDA13 組合上會炸的 FlashInfer backend
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
# FlashInfer 的 JIT sampler 在此機器的 arch check 會誤判失敗,改用原生 sampler
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

from qe.data import load_ir
from qe.expand import DEFAULT_EXPAND_PROMPT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="beir/scifact/test")
    ap.add_argument("--prompt-file", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--gpu-mem-util", type=float, default=0.70)  # 共享 GPU,留餘裕給別人
    ap.add_argument("--max-model-len", type=int, default=4096)
    args = ap.parse_args()

    prompt_tmpl = DEFAULT_EXPAND_PROMPT
    if args.prompt_file and Path(args.prompt_file).exists():
        prompt_tmpl = Path(args.prompt_file).read_text()

    ds = load_ir(args.dataset)
    queries = ds.queries
    if args.limit:
        queries = dict(list(queries.items())[: args.limit])
    qids = list(queries.keys())

    # 延遲 import,讓 --help 不需要 vLLM
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=args.max_model_len,
        enforce_eager=True,  # 新硬體先求穩,確認可跑後可關掉換 cudagraph 加速
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    # 套各 model 自己的 chat template
    prompts = []
    for qid in qids:
        user_msg = prompt_tmpl.format(query=queries[qid])
        prompts.append(
            tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        )

    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens, seed=0)
    outputs = llm.generate(prompts, sp)

    result = {qid: out.outputs[0].text.strip() for qid, out in zip(qids, outputs)}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"[worker] {args.model}: wrote {len(result)} expansions -> {args.out}")


if __name__ == "__main__":
    main()
