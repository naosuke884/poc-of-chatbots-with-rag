"""LLM・埋め込みの呼び出しで共通に使う設定と、トークン数・料金の集計。"""

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ANSWER_MODEL = "claude-sonnet-5-5"
INDEX_MODEL = "claude-haiku-5-5"
JUDGE_MODEL = "claude-opus-5-5"
EMBED_MODEL = "text-embedding-3-large"
RERANK_MODEL = "jev-latest"  # TypeSafe AI の Jev。ベクトル検索の並べ替えに使う

# 100 万トークンあたりの米ドル(入力, 出力, キャッシュ読み込み)
PRICES = {
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "text-embedding-3-large": (0.13, 0.0, 0.0),
    "jev-latest": (0.042, 0.0, 0.0),  # 出力は無料。公開の料金表がなく、紹介記事の値
}


@dataclass
class Usage:
    """モデルごとのトークン数。input にはキャッシュから読んだ分を含めない。"""

    tokens: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, model: str, input: int = 0, output: int = 0, cache_read: int = 0) -> None:
        t = self.tokens.setdefault(model, {"input": 0, "output": 0, "cache_read": 0})
        t["input"] += input
        t["output"] += output
        t["cache_read"] += cache_read

    def add_message(self, model: str, usage) -> None:
        """Anthropic の Message.usage を足す。キャッシュ書き込みは入力として数える。"""
        self.add(
            model,
            input=usage.input_tokens + (usage.cache_creation_input_tokens or 0),
            output=usage.output_tokens,
            cache_read=usage.cache_read_input_tokens or 0,
        )

    def merge(self, other: "Usage") -> None:
        for model, t in other.tokens.items():
            self.add(model, **t)

    @property
    def cost_usd(self) -> float:
        total = 0.0
        for model, t in self.tokens.items():
            pin, pout, pcache = PRICES[model]
            total += (t["input"] * pin + t["output"] * pout + t["cache_read"] * pcache) / 1_000_000
        return total

    def to_dict(self) -> dict:
        return {"tokens": self.tokens, "cost_usd": round(self.cost_usd, 6)}


_anthropic = None
_openai = None


def anthropic_client():
    global _anthropic
    if _anthropic is None:
        import anthropic

        _anthropic = anthropic.Anthropic()
    return _anthropic


def openai_client():
    global _openai
    if _openai is None:
        import openai

        _openai = openai.OpenAI()
    return _openai


_typesafe = None


def typesafe_client():
    global _typesafe
    if _typesafe is None:
        from typesafe_sdk import TypeSafeClient

        _typesafe = TypeSafeClient()
    return _typesafe
