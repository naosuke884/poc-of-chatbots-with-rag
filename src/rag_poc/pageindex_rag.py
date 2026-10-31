"""PageIndex 方式の検索〜回答生成。

3 資料の目次ツリー(節の見出し・ページ範囲・要約)を LLM に渡し、質問に答えるのに読むべき節を推論で選ばせる。
選ばれた節のページ本文をコンテキストにして、ベクトル方式と同じ answer.generate で回答する。
ツリーは毎回同じなので、プロンプトキャッシュを付けて送る。
"""

import time
from functools import cache

from pydantic import BaseModel

from rag_poc.answer import Passage, Result, generate, unique_pages
from rag_poc.docs import CORPUS_DOCS, clean_pages
from rag_poc.llm import ANSWER_MODEL, Usage, anthropic_client
from rag_poc.pageindex_build import load_tree

SELECT_EFFORT = "medium"
# コンテキストに入れるページ数の上限。選ばれた節の順に、上限に達するまでページを足す
MAX_PAGES = 10

SELECT_SYSTEM = """あなたは資料の目次ツリーを辿って、質問に答えるために読むべき節を選ぶアシスタントです。

以下は {n} つの資料の目次ツリーです。各節には id・見出し・ページ範囲・要約があります。
質問に答えるのに必要な記述が載っている節を、関係が深い順に最大 5 つ選んでください。

- 子の節で足りるときは、親ではなく子を選んでください(ページ範囲が狭いほうが読む量が少なく済みます)。
- 複数の資料や節にまたがる質問では、それぞれの節を選んでください。
- 該当しそうな節がなければ、空のリストを返してください。

<trees>
{trees}
</trees>"""


class _Selection(BaseModel):
    reasoning: str
    node_ids: list[str]


@cache
def _nodes() -> dict[str, dict]:
    """「doc:node_id」から節(doc・見出し・開始/終了ページ)への対応。"""
    nodes = {}

    def walk(doc_id, items):
        for n in items:
            nodes[f"{doc_id}:{n['node_id']}"] = {"doc": doc_id, "title": n["title"], "start": n["start_index"], "end": n["end_index"]}
            walk(doc_id, n.get("nodes", []))

    for doc in CORPUS_DOCS:
        walk(doc.id, load_tree(doc.id)["structure"])
    return nodes


@cache
def _trees_text() -> str:
    lines = []

    def walk(doc_id, items, depth):
        for n in items:
            summary = " ".join((n.get("summary") or "").split())
            lines.append(f"{'  ' * depth}- [{doc_id}:{n['node_id']}] {n['title']} (p{n['start_index']}-{n['end_index']}): {summary}")
            walk(doc_id, n.get("nodes", []), depth + 1)

    for doc in CORPUS_DOCS:
        lines.append(f'<tree doc="{doc.id}" title="{doc.title}">')
        walk(doc.id, load_tree(doc.id)["structure"], 0)
        lines.append("</tree>")
    return "\n".join(lines)


@cache
def _pages(doc_id: str) -> list[str]:
    return clean_pages(next(d for d in CORPUS_DOCS if d.id == doc_id))


def select_nodes(question: str, usage: Usage) -> _Selection:
    response = anthropic_client().messages.parse(
        model=ANSWER_MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": SELECT_SYSTEM.format(n=len(CORPUS_DOCS), trees=_trees_text()), "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": SELECT_EFFORT},
        messages=[{"role": "user", "content": f"質問: {question}"}],
        output_format=_Selection,
    )
    usage.add_message(ANSWER_MODEL, response.usage)
    return response.parsed_output or _Selection(reasoning=f"選択に失敗した({response.stop_reason})", node_ids=[])


def search(question: str, usage: Usage) -> tuple[list[Passage], _Selection]:
    selection = select_nodes(question, usage)
    nodes = _nodes()
    pages: list[tuple[str, int]] = []
    for node_id in selection.node_ids:
        node = nodes.get(node_id)
        if node is None:
            continue
        for page in range(node["start"], node["end"] + 1):
            if (node["doc"], page) not in pages and len(pages) < MAX_PAGES:
                pages.append((node["doc"], page))
    passages = [Passage(doc=d, page=p, text=_pages(d)[p - 1]) for d, p in pages]
    return passages, selection


def ask(question: str) -> Result:
    usage = Usage()
    started = time.perf_counter()
    passages, selection = search(question, usage)
    retrieved = time.perf_counter()
    answer = generate(question, passages, usage)
    finished = time.perf_counter()
    return Result(
        method="pageindex",
        question=question,
        answer=answer.answer,
        answerable=answer.answerable,
        sources=[s.model_dump() for s in answer.sources],
        retrieved=unique_pages(passages),
        retrieval_ms=round((retrieved - started) * 1000),
        generation_ms=round((finished - retrieved) * 1000),
        usage=usage,
        detail={
            "reasoning": selection.reasoning,
            "nodes": [{"id": n, **_nodes()[n]} for n in selection.node_ids if n in _nodes()],
        },
    )
