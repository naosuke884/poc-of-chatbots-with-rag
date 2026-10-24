"""ベクトル方式の検索〜回答生成。質問を埋め込み、コサイン距離の近いチャンクを TOP_K 件取ってコンテキストにする。

ask_page(ページ拡張)は、取ったチャンクをそのページの本文全体に置き換えてから渡す。
同じページのチャンクが複数取れたときは 1 ページにまとめる。

埋め込みは 3072 次元で、pgvector の vector 型の索引(HNSW / IVFFlat)は 2000 次元までしか張れない。
チャンクが数百件なので、索引なしの全件走査で十分速い。
"""

import time
from functools import cache

import numpy as np

from rag_poc.answer import Passage, Result, generate, unique_pages
from rag_poc.docs import DOCS_BY_ID, clean_pages
from rag_poc.llm import Usage
from rag_poc.vector_build import connect, embed

TOP_K = 8


def search(question: str, usage: Usage, k: int = TOP_K) -> list[Passage]:
    [q] = embed([question], usage)

    with connect() as conn:
        rows = conn.execute(
            "SELECT doc, page, text FROM chunks ORDER BY embedding <=> %s LIMIT %s",
            (np.array(q), k),
        ).fetchall()
    return [Passage(doc=d, page=p, text=t) for d, p, t in rows]


@cache
def _pages(doc_id: str) -> list[str]:
    return clean_pages(DOCS_BY_ID[doc_id])


def expand_to_pages(passages: list[Passage]) -> list[Passage]:
    return [Passage(doc=p["doc"], page=p["page"], text=_pages(p["doc"])[p["page"] - 1]) for p in unique_pages(passages)]


def ask(question: str, expand: bool = False) -> Result:
    usage = Usage()
    started = time.perf_counter()
    passages = search(question, usage)
    if expand:
        passages = expand_to_pages(passages)
    retrieved = time.perf_counter()
    answer = generate(question, passages, usage)
    finished = time.perf_counter()
    return Result(
        method="vector_page" if expand else "vector",
        question=question,
        answer=answer.answer,
        answerable=answer.answerable,
        sources=[s.model_dump() for s in answer.sources],
        retrieved=unique_pages(passages),
        retrieval_ms=round((retrieved - started) * 1000),
        generation_ms=round((finished - retrieved) * 1000),
        usage=usage,
    )


def ask_page(question: str) -> Result:
    return ask(question, expand=True)
