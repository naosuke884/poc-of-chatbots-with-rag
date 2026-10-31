"""ベクトル方式の検索〜回答生成。

ask(シンプルなベクトル検索)は、質問を埋め込み、コサイン距離の近いチャンクを TOP_K 件取ってそのまま渡す。
ask_jev(Jev で並べ替え)は、近いチャンクを RERANK_CANDIDATES 件取り、Jev(RERANK_MODEL)に各チャンクに
答えが書かれているかを判定させて、その確率の高い順に TOP_K 件を選び、それぞれのページ全体に広げてから渡す。
同じページのチャンクが複数あるときは 1 ページにまとめる。

埋め込みは 3072 次元で、pgvector の vector 型の索引(HNSW / IVFFlat)は 2000 次元までしか張れない。
チャンクが数百件なので、索引なしの全件走査で十分速い。
"""

import time
from functools import cache

import numpy as np

from rag_poc.answer import Passage, Result, generate, unique_pages
from rag_poc.docs import DOCS_BY_ID, clean_pages
from rag_poc.llm import RERANK_MODEL, Usage, typesafe_client
from rag_poc.vector_build import CHUNKS_TABLE, connect, embed

TOP_K = 8
RERANK_CANDIDATES = 30


def search(question: str, usage: Usage, k: int = TOP_K) -> list[Passage]:
    [q] = embed([question], usage)
    with connect() as conn:
        rows = conn.execute(f"SELECT doc, page, text FROM {CHUNKS_TABLE} ORDER BY embedding <=> %s LIMIT %s", (np.array(q), k)).fetchall()
    return [Passage(doc=d, page=p, text=t) for d, p, t in rows]


def rerank(question: str, passages: list[Passage], usage: Usage, k: int = TOP_K) -> tuple[list[Passage], list[float]]:
    """各チャンクに答えが書かれている確率を 1 回の呼び出しでまとめて判定し、高い順に k 件を返す。同じ確率なら元の順。"""
    from typesafe_sdk import Noul

    response = typesafe_client().system_one(
        model=RERANK_MODEL,
        state={"question": question, "passages": {f"p{i}": p.text for i, p in enumerate(passages)}},
        questions={f"p{i}": Noul(instructions=f"passages の p{i} に question の答えが書かれているか？") for i in range(len(passages))},
    )
    usage.add(RERANK_MODEL, input=response.usage.input_tokens, output=response.usage.output_tokens)
    scores = [response.answers[f"p{i}"].noul for i in range(len(passages))]
    order = sorted(range(len(passages)), key=lambda i: -scores[i])[:k]
    return [passages[i] for i in order], [scores[i] for i in order]


@cache
def _pages(doc_id: str) -> list[str]:
    return clean_pages(DOCS_BY_ID[doc_id])


def expand_to_pages(passages: list[Passage]) -> list[Passage]:
    return [Passage(doc=p["doc"], page=p["page"], text=_pages(p["doc"])[p["page"] - 1]) for p in unique_pages(passages)]


def retrieve(question: str, usage: Usage, jev: bool = False) -> tuple[list[Passage], dict]:
    """検索だけを行い、回答生成に渡す抜粋と方式ごとの補足を返す。"""
    if not jev:
        return search(question, usage), {}
    passages, scores = rerank(question, search(question, usage, k=RERANK_CANDIDATES), usage)
    return expand_to_pages(passages), {"rerank_scores": scores}


def ask(question: str, jev: bool = False, method: str = "vector") -> Result:
    usage = Usage()
    started = time.perf_counter()
    passages, detail = retrieve(question, usage, jev=jev)
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
        detail=detail,
    )


def ask_jev(question: str) -> Result:
    return ask(question, jev=True, method="vector_jev")
