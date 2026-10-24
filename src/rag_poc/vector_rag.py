"""ベクトル方式の検索〜回答生成。質問を埋め込み、コサイン距離の近いチャンクを TOP_K 件取ってコンテキストにする。

埋め込みは 3072 次元で、pgvector の vector 型の索引(HNSW / IVFFlat)は 2000 次元までしか張れない。
チャンクが数百件なので、索引なしの全件走査で十分速い。
"""

import time

import numpy as np

from rag_poc.answer import Passage, Result, generate, unique_pages
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


def ask(question: str) -> Result:
    usage = Usage()
    started = time.perf_counter()
    passages = search(question, usage)
    retrieved = time.perf_counter()
    answer = generate(question, passages, usage)
    finished = time.perf_counter()
    return Result(
        method="vector",
        question=question,
        answer=answer.answer,
        answerable=answer.answerable,
        sources=[s.model_dump() for s in answer.sources],
        retrieved=unique_pages(passages),
        retrieval_ms=round((retrieved - started) * 1000),
        generation_ms=round((finished - retrieved) * 1000),
        usage=usage,
    )
