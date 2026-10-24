"""両方式で共通の回答生成と、パイプラインが返す結果の型。

2 方式の違いを検索(コンテキストの集め方)だけにするため、回答生成は同じプロンプト・同じモデルで行う。
"""

from dataclasses import asdict, dataclass, field

from pydantic import BaseModel

from rag_poc.docs import DOCS_BY_ID
from rag_poc.llm import ANSWER_MODEL, Usage, anthropic_client

ANSWER_EFFORT = "medium"

SYSTEM = """あなたは IPA「中小企業の情報セキュリティ対策ガイドライン第4.0版」(本編・付録5 情報セキュリティ関連規程サンプル・付録3 自社診断)について質問に答えるアシスタントです。

- 与えられた資料の抜粋だけを根拠に、日本語で簡潔に答えてください。抜粋にない知識で補わないでください。
- 抜粋に答えが書かれていない場合は、推測せずに answerable を false にし、answer で「資料には記載がありません」と伝えてください。
- sources には、回答の根拠にした抜粋の doc と page を挙げてください。根拠にしなかった抜粋は挙げないでください。"""


class Passage(BaseModel):
    doc: str
    page: int
    text: str


class Source(BaseModel):
    doc: str
    page: int


class _Answer(BaseModel):
    answerable: bool
    answer: str
    sources: list[Source]


@dataclass
class Result:
    """パイプラインが返す結果。API と評価の両方でこの形を使う。"""

    method: str
    question: str
    answer: str
    answerable: bool
    sources: list[dict]  # 回答の根拠として LLM が挙げたページ [{doc, page}]
    retrieved: list[dict]  # 検索で取得してコンテキストに入れたページ [{doc, page}]
    retrieval_ms: int
    generation_ms: int
    usage: Usage = field(default_factory=Usage)
    detail: dict = field(default_factory=dict)  # 方式ごとの補足(PageIndex で選んだ節など)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usage"] = self.usage.to_dict()
        d["total_ms"] = self.retrieval_ms + self.generation_ms
        return d


def format_passages(passages: list[Passage]) -> str:
    parts = []
    for p in passages:
        title = DOCS_BY_ID[p.doc].title
        parts.append(f'<passage doc="{p.doc}" page="{p.page}" title="{title}">\n{p.text}\n</passage>')
    return "\n\n".join(parts)


def generate(question: str, passages: list[Passage], usage: Usage) -> _Answer:
    content = f"<passages>\n{format_passages(passages)}\n</passages>\n\n質問: {question}"
    response = anthropic_client().messages.parse(
        model=ANSWER_MODEL,
        max_tokens=16000,
        system=SYSTEM,
        output_config={"effort": ANSWER_EFFORT},
        messages=[{"role": "user", "content": content}],
        output_format=_Answer,
    )
    usage.add_message(ANSWER_MODEL, response.usage)
    if response.stop_reason == "refusal" or response.parsed_output is None:
        return _Answer(answerable=False, answer=f"回答を生成できませんでした({response.stop_reason})", sources=[])
    return response.parsed_output


def unique_pages(passages: list[Passage]) -> list[dict]:
    seen = []
    for p in passages:
        key = {"doc": p.doc, "page": p.page}
        if key not in seen:
            seen.append(key)
    return seen
