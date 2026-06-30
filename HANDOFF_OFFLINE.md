# 交接：離線 server 補跑 QA 全量（給協作 agent）

> 目的：在**無法連外網**的 server 上，把唯一剩下的實驗——**QA 全量 1000 題的 EM/F1**——補跑完並把分數落地成 CSV。
> 這份檔自含所有需要的指令、前置條件、與「跑完別讓分數掉」的保命步驟。其餘已完成項見最後一節。

---

## 0. 一句話任務

跑 `qe.qa` 全量 1000 題（5 個方法）、reader 用 Qwen2.5-7B、expansion 用 7B/8B 異質三人組，
**跑完務必用 §4 的離線算分腳本把 EM/F1 算出來存檔**（昨晚就是跑完 reader 但 process 在算分前被砍，分數沒落地）。

---

## 1. 離線環境前置條件（最關鍵，沒做會直接失敗）

這台不能連網，所以**模型與資料集必須事先放進本機 HF 快取**，並開離線旗標：

```bash
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**必須預先存在於本機 HF 快取**（`~/.cache/huggingface`，或設 `HF_HOME` 指到已備好的目錄）：

| 用途 | repo id |
|---|---|
| 資料集 | `Lala8383/ms-marco-qa-10k`（validation split） |
| reader + expansion #1 | `Qwen/Qwen2.5-7B-Instruct` |
| expansion #2 | `NousResearch/Meta-Llama-3.1-8B-Instruct` |
| expansion #3 | `unsloth/mistral-7b-instruct-v0.3` |

> 驗證有沒有備齊：`python -c "from datasets import load_dataset; load_dataset('Lala8383/ms-marco-qa-10k', split='validation')"`
> 開了 OFFLINE 旗標後若報 `Couldn't find` / `connection`，就是該 repo 沒先快取好——請先在有網路的機器 `huggingface-cli download <repo>` 再搬過來。

**（可選加速）** 把本機這份 repo 的 `cache/expand/qa-validation-full__*.json`（3 個檔，1000 題 expansion 已算好）一起搬過去放到對應 `cache/expand/`，expansion 階段就會 cache-hit 直接跳過，只剩 reader 要算。沒搬也行，會自動重生成。

---

## 2. 環境 / GPU 注意事項

- 進虛擬環境：`source merge_venv/bin/activate`（若這台是另一套環境，確認已裝 vllm/ranx/bm25s/ir_datasets/datasets）。
- **挑一張真正空的卡**：`nvidia-smi` 先看，把 index 填到 `--gpu`。**不要擠掉別人正在跑的 process**。
- reader 是 7B，**單張至少 ~16GB VRAM**；24GB 以上最穩。`--gpu-mem-util` 預設可給 `0.85`（自己獨佔的卡）；**共享卡請壓到 0.45~0.6 避免擠掉別人**。
- vLLM 在 Blackwell(RTX 5090)+CUDA13 的環境設定已寫死在 `qe/vllm_worker.py`（`CUDA_DEVICE_ORDER=PCI_BUS_ID` 等）。**若這台不是 Blackwell**，那些設定無害可忽略；若 vLLM 起不來看 §5。

---

## 3. 主指令：QA 全量 1000 題

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1

python -m qe.qa \
  --split validation --limit 0 --k 3 \
  --methods closed,B0,M1,M2,oracle \
  --reader vllm --reader-model Qwen/Qwen2.5-7B-Instruct \
  --gpu <空卡INDEX> --gpu-mem-util 0.85 \
  --expand-models Qwen/Qwen2.5-7B-Instruct,NousResearch/Meta-Llama-3.1-8B-Instruct,unsloth/mistral-7b-instruct-v0.3 \
  2>&1 | tee qa_full1k_offline.log
```

- `--limit 0` = 跑全部（約 1000 題）。流程：先對所有問題用三模型各自產 expansion（GPU），再用 7B reader 一次批次作答（GPU），最後算 EM/F1 印表。
- 每個 model 跑完釋放 GPU 再換下一個（subprocess），峰值只占一個 7-8B 模型的量。
- 全程結果快取在 `cache/expand/`（expansion）與 `cache/qa/`（reader 答案）；中斷後重跑會 cache-hit 續跑，不會白做。

**預期最後印出**（這段出現才算成功）：
```
===== QA 對照表 =====
method            em        f1
closed       ...
B0           ...
M1           ...
M2           ...
oracle       ...
```

---

## 4. 保命步驟：跑完一定要把分數算出來存檔

**昨晚的教訓**：reader 全部生成完、快取也寫了，但 process 在「算分印表」前被砍 → 分數沒落地。
只要 reader 快取在，分數就能**離線重算、不必再上 GPU**。把下面存成 `rescore_qa.py` 跑：

```python
# rescore_qa.py — 從 cache/qa 的 reader 答案離線重算 EM/F1（不需 GPU）
import glob, json, os
from collections import defaultdict
from qe.qa_data import load_qa
from qe import qa_metrics

# 找最新、且筆數最多的 7B reader 快取（method||qid 為 key）
cands = sorted(glob.glob("cache/qa/reader-Qwen2.5-7B-Instruct__*.json"),
               key=os.path.getmtime, reverse=True)
assert cands, "找不到 reader 快取，reader 階段沒跑成功"
path = max(cands, key=lambda p: len(json.load(open(p))))  # 挑筆數最多的
ans = json.load(open(path))
print("用快取:", path, " 筆數:", len(ans))

by_method = defaultdict(dict); qids = set()
for key, a in ans.items():
    m, qid = key.split("||", 1); by_method[m][qid] = a; qids.add(qid)

gold = {e.qid: e.answers for e in load_qa(split="validation", limit=0) if e.qid in qids}
print(f"題數={len(qids)}  金答案對到={len(gold)}")

rows = []
print(f"\n{'method':10}{'EM':>10}{'F1':>10}{'n':>8}")
for m in ["closed","B0","M1","M2","oracle"]:
    if m not in by_method: continue
    preds = {qid: by_method[m].get(qid, "") for qid in qids}
    sc = qa_metrics.score(preds, gold)
    print(f"{m:10}{sc['em']:>10.4f}{sc['f1']:>10.4f}{len(preds):>8}")
    rows.append(f"{m},{sc['em']:.4f},{sc['f1']:.4f},{len(preds)},Qwen2.5-7B-Instruct")

with open("qa_full1k_qwen7b.csv", "w") as f:
    f.write("method,em,f1,n,reader\n" + "\n".join(rows) + "\n")
print("\n已寫入 qa_full1k_qwen7b.csv")
```

```bash
python rescore_qa.py
```

> 健全性檢查（數字對才算正常）：`oracle` 的 F1 應最高（ceiling）、`closed` 最低（floor）、
> `B0/M1/M2` 夾在中間且 `M2 ≳ B0`。500 題版參考值：closed F1≈0.166、B0≈0.318、M1≈0.310、M2≈0.325、oracle≈0.437。
> 1000 題數字會接近但不會完全一樣。

---

## 5. 疑難排解

- **`Couldn't find` / 連線錯誤**：模型或資料集沒事先快取好 → 回 §1 補。
- **OOM**：卡太小塞不下 7B → 換更大卡，或把 `--gpu-mem-util` 調高（獨佔卡）/ 換較小 reader（會改變絕對分數）。
- **vLLM 起不來**（非 Blackwell 機）：`qe/vllm_worker.py` 裡 Blackwell 專用的環境變數可保留無妨；若是 torch/CUDA 版本問題，依該機實際 CUDA 版本裝對應 vllm/torch。
- **想完全重算不吃快取**：主指令加 `--force`。
- **看單題輸出 debug**：`cache/expand/`（expansion）、`cache/qa/`（reader 答案）都是可讀 JSON，key 為 `method||qid`。

---

## 6. 其餘狀態（不用這台跑，純背景）

- **已完成**：
  - IR OOD 5 個 BEIR（scifact/fiqa/touche2020/dbpedia/nq）B0/M1/M2 → `exp4fuse_ood.csv`，B0 已對上論文 Table 2。
  - IR in-domain（msmarco-dev/dl19/dl20）→ `exp4fuse_compare.csv`。
  - QA **500 題** Qwen2.5-7B → `qa_500_qwen7b.csv`（已從快取救回）。
- **本次要補**：QA **全量 1000 題**（本檔 §3+§4）。
- **被授權語料卡住（可選，且這台若有 TREC 語料反而適合補）**：
  IR 的 `robust04`（需 TREC disks 4&5）與 `news`（需 WaPo v2）。
  把語料放到 `~/.ir_datasets/{disks45,wapo}` 後即可：
  ```bash
  for D in robust04 news; do
    python -m qe.run --dataset $D --methods B0,M1,M2 --k 1000 --local-gpu <空卡> \
      --stage1-models Qwen/Qwen2.5-7B-Instruct,NousResearch/Meta-Llama-3.1-8B-Instruct,unsloth/mistral-7b-instruct-v0.3
  done
  ```
  論文 nDCG@10 參考：Robust04 40.7、News 39.5。
