"""ベクトル検索と PageIndex を組み合わせる方式の検索〜回答生成。

1. ベクトル検索で近いチャンクを RERANK_CANDIDATES 件取り、上位に出てきた順に資料を HYBRID_DOCS 冊選ぶ
2. その資料の目次ツリーだけを LLM に渡し、読む節を選ばせる(PageIndex 方式と同じ選び方)
3. 選んだ節のページ(最大 MAX_PAGES)を Jev で判定し、答えが書かれている確率の高い順に HYBRID_PAGES ページを渡す

全資料のツリーを送らないので、資料が増えても節の選択の時間と料金がほぼ一定になる。
"""

import time

from rag_poc.answer import Passage, Result, generate, unique_pages
from rag_poc.llm import Usage
from rag_poc.pageindex_rag import _node_pages, _nodes, select_nodes
from rag_poc.vector_rag import RERANK_CANDIDATES, rerank, search

HYBRID_DOCS = 3
# 回答生成に渡すページ数。Jev で並べ替えるベクトル方式(平均 6 ページ前後)と揃える
HYBRID_PAGES = 6


def retrieve(question: str, usage: Usage) -> tuple[list[Passage], dict]:
    docs = []
    for p in search(question, usage, k=RERANK_CANDIDATES):
        if p.doc not in docs:
            docs.append(p.doc)
    docs = tuple(docs[:HYBRID_DOCS])
    selection = select_nodes(question, usage, doc_ids=docs)
    passages = _node_pages(selection.node_ids)
    scores = []
    if passages:
        passages, scores = rerank(question, passages, usage, k=HYBRID_PAGES)
    nodes = _nodes()
    return passages, {
        "docs": list(docs),
        "reasoning": selection.reasoning,
        "nodes": [{"id": n, **nodes[n]} for n in selection.node_ids if n in nodes],
        "rerank_scores": scores,
    }


def ask(question: str) -> Result:
    usage = Usage()
    started = time.perf_counter()
    passages, detail = retrieve(question, usage)
    retrieved = time.perf_counter()
    answer = generate(question, passages, usage)
    finished = time.perf_counter()
    return Result(
        method="hybrid_jev",
        question=question,
        answer=answer.answer,
        answerable=answer.answerable,
        sources=[s.model_dump() for s in answer.sources],
        retrieved=unique_pages(passages),
        retrieval_ms=round((retrieved - started) * 1000),
        generation_ms=round((finished - retrieved) * 1000),
        usage=usage,
        detail=detail,
    )
