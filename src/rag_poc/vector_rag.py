"""ベクトル方式の検索〜回答生成。質問を埋め込み、コサイン距離の近いチャンクを TOP_K 件取ってコンテキストにする。

ask_page(ページ拡張)は、取ったチャンクをそのページの本文全体に置き換えてから渡す。
同じページのチャンクが複数取れたときは 1 ページにまとめる。
ask_page_doc(資料ごとの枠)は、ページ拡張に加えて、全体の上位 TOP_K 件とは別に各資料の上位 PER_DOC 件を足す。
似た記述が他の資料にあるとき、本来の資料のチャンクが全体の上位から漏れるのを防ぐ。

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
PER_DOC = 2


def search(question: str, usage: Usage, k: int = TOP_K, per_doc: int = 0) -> list[Passage]:
    [q] = embed([question], usage)
    v = np.array(q)

    with connect() as conn:
        rows = conn.execute("SELECT id, doc, page, text FROM chunks ORDER BY embedding <=> %s LIMIT %s", (v, k)).fetchall()
        if per_doc:
            # 資料ごとの上位 per_doc 件のうち、全体の上位に入っていないものを後ろに足す
            extra = conn.execute(
                """SELECT id, doc, page, text FROM (
                       SELECT *, row_number() OVER (PARTITION BY doc ORDER BY embedding <=> %s) AS rank FROM chunks
                   ) ranked WHERE rank <= %s ORDER BY embedding <=> %s""",
                (v, per_doc, v),
            ).fetchall()
            seen = {r[0] for r in rows}
            rows += [r for r in extra if r[0] not in seen]
    return [Passage(doc=d, page=p, text=t) for _, d, p, t in rows]


@cache
def _pages(doc_id: str) -> list[str]:
    return clean_pages(DOCS_BY_ID[doc_id])


def expand_to_pages(passages: list[Passage]) -> list[Passage]:
    return [Passage(doc=p["doc"], page=p["page"], text=_pages(p["doc"])[p["page"] - 1]) for p in unique_pages(passages)]


def ask(question: str, expand: bool = False, per_doc: int = 0, method: str = "vector") -> Result:
    usage = Usage()
    started = time.perf_counter()
    passages = search(question, usage, per_doc=per_doc)
    if expand:
        passages = expand_to_pages(passages)
    retrieved = time.perf_counter()
    answer = generate(question, passages, usage)
    finished = time.perf_counter()
    return Result(
        method=method,
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
    return ask(question, expand=True, method="vector_page")


def ask_page_doc(question: str) -> Result:
    return ask(question, expand=True, per_doc=PER_DOC, method="vector_page_doc")
