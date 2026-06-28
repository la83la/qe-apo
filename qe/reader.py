"""QA Reader — 讀 top-k passages 生成答案。

provider 可插拔(對齊 qe.expand 的精神):
    MockReader   無 GPU 佔位;回傳 top-1 passage 的首句,讓整條 QA pipeline 跑通
    VLLMReader   本地 vLLM(沿用 qe.vllm_worker 的 messages-file 介面),跑完釋放 GPU

介面:read(question, passages) -> answer_str。passages 已是檢索排序後的 top-k。
批次入口 read_many({key: (question, passages)}) -> {key: answer},vLLM 一次載入
生成全部(只 load 一次模型)。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Reader 的 prompt 也是一個欄位,之後可交給 qe.apo 優化(QA 端的 APO)
DEFAULT_READ_PROMPT = (
    "Answer the question using only the passages below. "
    "Reply with a short answer (a few words), no explanation.\n\n"
    "Passages:\n{passages}\n\nQuestion: {question}\n\nAnswer:"
)
# 無 passage(closed-book)時用:只靠模型內建知識作答
CLOSED_BOOK_PROMPT = (
    "Answer the question with a short answer (a few words), no explanation.\n\n"
    "Question: {question}\n\nAnswer:"
)

# 單段 passage 截斷上限(字元),避免 reader prompt 爆 max_model_len
MAX_PASSAGE_CHARS = 1200
CACHE_DIR = Path("cache/qa")


class Reader:
    name: str

    def read(self, question: str, passages: list[str]) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def read_many(self, items: dict[str, tuple[str, list[str]]]) -> dict[str, str]:
        """預設逐題呼叫 read();VLLMReader 覆寫成單次批次生成。"""
        return {key: self.read(q, ps) for key, (q, ps) in items.items()}


def _first_sentence(text: str) -> str:
    text = text.strip()
    m = re.search(r"(.+?[.!?])(\s|$)", text)
    return (m.group(1) if m else text).strip()


def build_message(question: str, passages: list[str],
                  read_prompt: str = DEFAULT_READ_PROMPT,
                  closed_prompt: str = CLOSED_BOOK_PROMPT) -> str:
    """把 (question, passages) 組成餵給 LLM 的 user message。passages 空 → closed-book。"""
    if not passages:
        return closed_prompt.format(question=question)
    numbered = "\n".join(
        f"[{i + 1}] {p.strip()[:MAX_PASSAGE_CHARS]}" for i, p in enumerate(passages) if p.strip()
    )
    return read_prompt.format(passages=numbered, question=question)


@dataclass
class MockReader(Reader):
    """佔位 reader:回傳 top-1 passage 的首句當答案。

    不產生真實理解,但讓檢索 → 生成 → EM/F1 整條 wiring 可驗證:
    檢索越準(top-1 越可能是含答案那段),F1 越高。
    """

    name: str = "mock-reader"

    def read(self, question: str, passages: list[str]) -> str:
        if not passages:
            return ""
        return _first_sentence(passages[0])


@dataclass
class VLLMReader(Reader):
    """本地 vLLM reader。沿用 qe.vllm_worker 的 messages-file 模式,單 subprocess
    一次載入模型、批次生成全部答案、寫出後釋放 GPU。結果以 messages 指紋快取。"""

    model: str = "Qwen/Qwen2.5-3B-Instruct"
    gpu: int = 5
    max_tokens: int = 64           # QA 短答案
    temperature: float = 0.0       # 貪婪解碼,EM/F1 可重現
    gpu_mem_util: float = 0.85
    max_model_len: int = 4096
    read_prompt: str = DEFAULT_READ_PROMPT
    name: str = "vllm-reader"
    force: bool = False

    def read(self, question: str, passages: list[str]) -> str:
        return self.read_many({"_one": (question, passages)})["_one"]

    def read_many(self, items: dict[str, tuple[str, list[str]]]) -> dict[str, str]:
        if not items:
            return {}
        msgs = {
            key: build_message(q, ps, self.read_prompt)
            for key, (q, ps) in items.items()
        }
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        short = self.model.split("/")[-1]
        fingerprint = json.dumps(msgs, sort_keys=True, ensure_ascii=False)
        h = hashlib.sha1(f"{self.model}|{self.max_tokens}|{fingerprint}".encode()).hexdigest()[:8]
        out = CACHE_DIR / f"reader-{short}__{h}.json"

        if out.exists() and not self.force:
            print(f"[reader] cache hit: {out}")
            with open(out) as f:
                return json.load(f)

        msgs_file = CACHE_DIR / f"_reader-msgs-{short}.json"
        with open(msgs_file, "w") as f:
            json.dump(msgs, f, ensure_ascii=False)

        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(self.gpu))
        cmd = [
            sys.executable, "-m", "qe.vllm_worker",
            "--model", self.model,
            "--messages-file", str(msgs_file),
            "--out", str(out),
            "--max-tokens", str(self.max_tokens),
            "--temperature", str(self.temperature),
            "--gpu-mem-util", str(self.gpu_mem_util),
            "--max-model-len", str(self.max_model_len),
        ]
        print(f"[reader] vLLM {self.model} on GPU {self.gpu}: {len(msgs)} prompts ...")
        subprocess.run(cmd, env=env, check=True)
        with open(out) as f:
            return json.load(f)


def build_reader(kind: str = "mock", **kwargs) -> Reader:
    if kind == "mock":
        return MockReader()
    if kind in ("vllm", "vllm-reader"):
        return VLLMReader(**kwargs)
    raise ValueError(f"unknown reader kind: {kind!r} (支援: mock, vllm)")
