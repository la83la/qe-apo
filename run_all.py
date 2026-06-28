"""彙整 driver — 一次跑 IR + QA, 把結果寫成 CSV(長格式)。

長格式 CSV 欄位: task,dataset,method,metric,value
(各 task 指標不同, 長格式最乾淨, 之後用 pandas/excel pivot 成寬表)。

用法:
    # 無 GPU 健檢(mock,確認彙整流程通)
    python run_all.py --out results.csv --limit 20 \
        --ir-methods B0,M2 --qa-methods closed,B0,oracle --reader mock

    # 真跑(IR 在 GPU N1 做 expansion;QA reader/expansion 在 GPU N2)
    python run_all.py --out results.csv --limit 200 --k 3 \
        --ir-methods B0,M1,M2 --qa-methods closed,B0,M1,M2,oracle \
        --reader vllm --reader-model Qwen/Qwen2.5-3B-Instruct \
        --ir-gpu 1 --qa-gpu 5 \
        --expand-models Qwen/Qwen2.5-3B-Instruct,unsloth/Llama-3.2-1B-Instruct,google/gemma-4-E2B-it

只想跑其中一個: --skip-qa 或 --skip-ir。
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    # 共用
    ap.add_argument("--out", default="results.csv", help="輸出 CSV 路徑")
    ap.add_argument("--limit", type=int, default=0, help="兩個 task 都只取前 N 題(0=全部)")
    ap.add_argument("--k", type=int, default=100, help="IR 檢索深度 / 也作 QA 預設 top-k")
    ap.add_argument("--skip-ir", action="store_true")
    ap.add_argument("--skip-qa", action="store_true")
    # IR
    ap.add_argument("--ir-dataset", default="beir/scifact/test")
    ap.add_argument("--ir-methods", default="B0,M1,M2")
    ap.add_argument("--ir-gpu", type=int, default=-1, help="IR Stage1 本地 vLLM GPU(-1=mock)")
    ap.add_argument("--ir-stage1-models", default="", help="留空用 stage1.DEFAULT_MODELS")
    ap.add_argument("--ir-models", default="", help="Anthropic model id(逗號分隔);留空 mock")
    # QA
    ap.add_argument("--qa-dataset", default="Lala8383/ms-marco-qa-10k")
    ap.add_argument("--qa-split", default="validation")
    ap.add_argument("--qa-methods", default="closed,B0,M1,M2,oracle")
    ap.add_argument("--qa-k", type=int, default=3, help="QA 餵 reader 的 top-k passages")
    ap.add_argument("--qa-gpu", type=int, default=5, help="QA reader/expansion 的 GPU")
    ap.add_argument("--reader", default="mock", help="QA reader: mock 或 vllm")
    ap.add_argument("--reader-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--expand-models", default="", help="QA expansion 模型;留空用 QA 預設三人組")
    ap.add_argument("--force", action="store_true", help="忽略快取重跑")
    args = ap.parse_args()

    rows: list[dict] = []  # 每筆: task,dataset,method,metric,value

    if not args.skip_ir:
        from qe.run import Stage2Config, run_ir
        print("\n######## IR ########")
        stage2 = Stage2Config(mode="rrf", gpu=args.ir_gpu, dataset=args.ir_dataset)
        ir = run_ir(args.ir_dataset, args.ir_methods.split(","), args.k, args.limit,
                    args.ir_models, args.ir_gpu, args.ir_stage1_models, stage2)
        for method, sc in ir.items():
            for metric, value in sc.items():
                rows.append({"task": "IR", "dataset": args.ir_dataset,
                             "method": method, "metric": metric, "value": value})

    if not args.skip_qa:
        from qe.qa import run_qa_eval
        print("\n######## QA ########")
        qa = run_qa_eval(args.qa_dataset, args.qa_split, args.limit, args.qa_k,
                         args.qa_methods.split(","), args.reader, args.reader_model,
                         args.qa_gpu, args.expand_models, args.force)
        ds_tag = f"{args.qa_dataset}[{args.qa_split}]"
        for method, sc in qa.items():
            for metric, value in sc.items():
                rows.append({"task": "QA", "dataset": ds_tag,
                             "method": method, "metric": metric, "value": value})

    out = Path(args.out)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["task", "dataset", "method", "metric", "value"])
        w.writeheader()
        for r in rows:
            r = dict(r, value=round(r["value"], 4))
            w.writerow(r)

    print(f"\n===== 寫出 {len(rows)} 列 -> {out} =====")
    print(f"{'task':5}{'method':10}{'metric':14}{'value':>8}")
    for r in rows:
        print(f"{r['task']:5}{r['method']:10}{r['metric']:14}{r['value']:>8.4f}")


if __name__ == "__main__":
    main()
