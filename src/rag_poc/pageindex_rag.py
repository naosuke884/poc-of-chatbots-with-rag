"""PageIndex 方式の検索〜回答生成。

3 資料の目次ツリー(節の見出し・ページ範囲・要約)を LLM に渡し、質問に答えるのに読むべき節を推論で選ばせる。
選ばれた節のページ本文をコンテキストにして、ベクトル方式と同じ answer.generate で回答する。
ツリーは毎回同じなので、プロンプトキャッシュを付けて送る。

ask_jev は、節の選択を LLM の代わりに Jev(RERANK_MODEL)で行う。全資料の節(資料名・上位の見出し・ページ範囲・要約)を
Jev に渡し、節ごとに答えが書かれていそうかを判定させて、確率の高い順に MAX_NODES 個の節を選ぶ。
"""

import time
from concurrent.futures import ThreadPoolExecutor
from functools import cache

from pydantic import BaseModel

from rag_poc.answer import Passage, Result, generate, unique_pages
from rag_poc.docs import CORPUS_DOCS, clean_pages
from rag_poc.llm import ANSWER_MODEL, RERANK_MODEL, Usage, anthropic_client, typesafe_client
from rag_poc.pageindex_build import load_tree

SELECT_EFFORT = "medium"
# コンテキストに入れるページ数の上限。選ばれた節の順に、上限に達するまでページを足す
MAX_PAGES = 10
# Jev で選ぶ節の数(LLM に選ばせるときの上限と同じ)
MAX_NODES = 5
# Jev に 1 回で渡す節の数。120 節(約 3.4 万トークン)は通り、160 節は上限を超えた
JEV_BATCH = 100

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
def _trees_text(doc_ids: tuple[str, ...] | None = None) -> str:
    """目次ツリーの文字列。doc_ids を渡すとその資料だけにする。"""
    lines = []

    def walk(doc_id, items, depth):
        for n in items:
            summary = " ".join((n.get("summary") or "").split())
            lines.append(f"{'  ' * depth}- [{doc_id}:{n['node_id']}] {n['title']} (p{n['start_index']}-{n['end_index']}): {summary}")
            walk(doc_id, n.get("nodes", []), depth + 1)

    for doc in CORPUS_DOCS if doc_ids is None else [d for d in CORPUS_DOCS if d.id in doc_ids]:
        lines.append(f'<tree doc="{doc.id}" title="{doc.title}">')
        walk(doc.id, load_tree(doc.id)["structure"], 0)
        lines.append("</tree>")
    return "\n".join(lines)


@cache
def _pages(doc_id: str) -> list[str]:
    return clean_pages(next(d for d in CORPUS_DOCS if d.id == doc_id))


def select_nodes(question: str, usage: Usage, doc_ids: tuple[str, ...] | None = None) -> _Selection:
    """読む節を LLM に選ばせる。doc_ids を渡すとその資料のツリーだけを渡す。

    全資料のツリーは毎回同じなのでキャッシュする。資料を絞ったツリーは質問ごとに変わり、キャッシュしても読まれないので付けない。
    """
    n = len(CORPUS_DOCS) if doc_ids is None else len(doc_ids)
    system = {"type": "text", "text": SELECT_SYSTEM.format(n=n, trees=_trees_text(doc_ids))}
    if doc_ids is None:
        system["cache_control"] = {"type": "ephemeral"}
    response = anthropic_client().messages.parse(
        model=ANSWER_MODEL,
        max_tokens=16000,
        system=[system],
        output_config={"effort": SELECT_EFFORT},
        messages=[{"role": "user", "content": f"質問: {question}"}],
        output_format=_Selection,
    )
    usage.add_message(ANSWER_MODEL, response.usage)
    return response.parsed_output or _Selection(reasoning=f"選択に失敗した({response.stop_reason})", node_ids=[])


def _node_pages(node_ids: list[str]) -> list[Passage]:
    """節の順に、MAX_PAGES に達するまでページを足す。"""
    nodes = _nodes()
    pages: list[tuple[str, int]] = []
    for node_id in node_ids:
        node = nodes.get(node_id)
        if node is None:
            continue
        for page in range(node["start"], node["end"] + 1):
            if (node["doc"], page) not in pages and len(pages) < MAX_PAGES:
                pages.append((node["doc"], page))
    return [Passage(doc=d, page=p, text=_pages(d)[p - 1]) for d, p in pages]


def search(question: str, usage: Usage) -> tuple[list[Passage], _Selection]:
    selection = select_nodes(question, usage)
    return _node_pages(selection.node_ids), selection


@cache
def _sections() -> dict[str, str]:
    """「doc:node_id」から、Jev に渡す節の説明(資料名・上位の見出し・ページ範囲・要約)への対応。"""
    sections = {}

    def walk(doc, items, path):
        for n in items:
            summary = " ".join((n.get("summary") or "").split())
            heading = " > ".join([*path, n["title"]])
            sections[f"{doc.id}:{n['node_id']}"] = f"{doc.title} / {heading} (p{n['start_index']}-{n['end_index']}): {summary}"
            walk(doc, n.get("nodes", []), [*path, n["title"]])

    for doc in CORPUS_DOCS:
        walk(doc, load_tree(doc.id)["structure"], [])
    return sections


def _score_sections(question: str, ids: list[str], usage: Usage) -> dict[str, float]:
    from typesafe_sdk import Noul, NoulCriteria

    criteria = NoulCriteria(true="この節の本文に question への具体的な答えがある", false="話題が近いだけ、または関係がない")
    response = typesafe_client().system_one(
        model=RERANK_MODEL,
        state={"question": question, "sections": {f"s{i}": _sections()[n] for i, n in enumerate(ids)}},
        questions={f"s{i}": Noul(instructions=f"question に答えるために、sections の s{i} の節を読む必要があるか？", criteria=criteria) for i in range(len(ids))},
    )
    usage.add(RERANK_MODEL, input=response.usage.input_tokens, output=response.usage.output_tokens)
    return {n: response.answers[f"s{i}"].noul for i, n in enumerate(ids)}


def select_nodes_jev(question: str, usage: Usage) -> tuple[list[str], list[float]]:
    """全部の節を判定させ、確率の高い順に MAX_NODES 個の節と確率を返す。同じ確率ならページ範囲が狭い節を先にする。

    Jev は 1 回に渡せる量に上限がある(11 冊の 262 節は入らない)ので、JEV_BATCH 節ずつに分けて並列に呼ぶ。
    """
    ids = list(_sections())
    batches = [ids[i : i + JEV_BATCH] for i in range(0, len(ids), JEV_BATCH)]
    usages = [Usage() for _ in batches]
    with ThreadPoolExecutor(len(batches)) as pool:
        results = list(pool.map(lambda b: _score_sections(question, *b), zip(batches, usages)))
    for u in usages:
        usage.merge(u)
    scores = {n: v for r in results for n, v in r.items()}
    nodes = _nodes()
    order = sorted(ids, key=lambda n: (-scores[n], nodes[n]["end"] - nodes[n]["start"]))[:MAX_NODES]
    return order, [scores[n] for n in order]


def search_jev(question: str, usage: Usage) -> tuple[list[Passage], list[str], list[float]]:
    node_ids, scores = select_nodes_jev(question, usage)
    return _node_pages(node_ids), node_ids, scores


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


def ask_jev(question: str) -> Result:
    usage = Usage()
    started = time.perf_counter()
    passages, node_ids, scores = search_jev(question, usage)
    retrieved = time.perf_counter()
    answer = generate(question, passages, usage)
    finished = time.perf_counter()
    return Result(
        method="pageindex_jev",
        question=question,
        answer=answer.answer,
        answerable=answer.answerable,
        sources=[s.model_dump() for s in answer.sources],
        retrieved=unique_pages(passages),
        retrieval_ms=round((retrieved - started) * 1000),
        generation_ms=round((finished - retrieved) * 1000),
        usage=usage,
        detail={"nodes": [{"id": n, **_nodes()[n]} for n in node_ids], "rerank_scores": scores},
    )
