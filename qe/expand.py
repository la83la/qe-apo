"""Multi-model query expansion。

每個 Expander 把 query 變成「擴展後的 query 文字」(原 query + 補充 term/pseudo-doc)。
provider 可插拔:
    AnthropicExpander  真的呼叫 Claude (需 ANTHROPIC_API_KEY)
    MockExpander       無 API key 時的佔位, 讓整條 pipeline 能跑通

prompt 是一個欄位, 之後交給 apo 模組去優化。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

# Query2Doc 風格的預設 prompt; APO 會在此基礎上搜尋更好的版本
DEFAULT_EXPAND_PROMPT = (
    "Write a short passage that answers the following query. "
    "Use precise, topic-relevant terminology.\n\nQuery: {query}\n\nPassage:"
)


class Expander:
    name: str

    def expand(self, query: str) -> str:  # pragma: no cover - interface
        raise NotImplementedError

    def expand_query(self, query: str, repeat_orig: int = 5) -> str:
        """Query2Doc 作法: 原 query 重複數次以保權重, 接上生成的 pseudo-doc。"""
        psg = self.expand(query)
        return (" ".join([query] * repeat_orig) + " " + psg).strip()


@dataclass
class MockExpander(Expander):
    """無 API key 時用; 依 model name 加不同的固定佐料, 製造 model 間差異供 ensemble 測試。"""

    name: str = "mock"
    flavor: str = "generic background information context details"

    def expand(self, query: str) -> str:
        return f"{query} {self.flavor}"


@dataclass
class AnthropicExpander(Expander):
    name: str
    model: str
    prompt: str = DEFAULT_EXPAND_PROMPT
    max_tokens: int = 256
    temperature: float = 0.7
    _client: object = field(default=None, repr=False)

    def _get_client(self):
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic()
        return self._client

    def expand(self, query: str) -> str:
        msg = self._get_client().messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": self.prompt.format(query=query)}],
        )
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()


@dataclass
class CachedExpander(Expander):
    """讀 qe.stage1 產生的快取檔(vLLM 跑出來的 expansion),不碰 GPU。

    用於 ensemble / 評估階段:Stage1 已把各 model 的 expansion 寫成 JSON,
    這裡用 query_id 對應取出,組成 Query2Doc 風格的 expanded query。
    """

    name: str
    _by_qid: dict[str, str] = field(default_factory=dict)

    def expand(self, query: str) -> str:  # 不以文字為 key,改用 expand_for_qid
        raise NotImplementedError("CachedExpander 請用 expand_query_for_qid(qid, query)")

    def expand_query_for_qid(self, qid: str, query: str, repeat_orig: int = 5) -> str:
        psg = self._by_qid.get(qid, "")
        return (" ".join([query] * repeat_orig) + " " + psg).strip()


def load_cached_expanders(model_to_file: dict[str, str]) -> list[CachedExpander]:
    import json

    out = []
    for model, path in model_to_file.items():
        with open(path) as f:
            data = json.load(f)
        out.append(CachedExpander(name=model.split("/")[-1], _by_qid=data))
    return out


def build_expanders(model_ids: list[str] | None = None) -> list[Expander]:
    """有 ANTHROPIC_API_KEY 就用真 model, 否則回 mock 群 (仍有 model 間差異)。"""
    if os.getenv("ANTHROPIC_API_KEY") and model_ids:
        return [AnthropicExpander(name=m.split("-")[1] if "-" in m else m, model=m) for m in model_ids]
    return [
        MockExpander(name="mockA", flavor="definition causes mechanism background"),
        MockExpander(name="mockB", flavor="clinical evidence study results outcome"),
        MockExpander(name="mockC", flavor="related concepts synonyms terminology"),
    ]
