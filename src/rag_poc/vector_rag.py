"""ベクトル方式の検索〜回答生成。質問を埋め込み、コサイン距離の近いチャンクを TOP_K 件取ってコンテキストにする。

ask_page(ページ拡張)は、取ったチャンクをそのページの本文全体に置き換えてから渡す。
同じページのチャンクが複数取れたときは 1 ページにまとめる。
ask_page_doc(資料ごとの枠)は、ページ拡張に加えて、全体の上位 TOP_K 件とは別に各資料の上位 PER_DOC 件を足す。
似た記述が他の資料にあるとき、本来の資料のチャンクが全体の上位から漏れるのを防ぐ。
ask_jev(並べ替え)は、近いチャンクを RERANK_CANDIDATES 件取り、Jev(RERANK_MODEL)に各チャンクに答えが
書かれているかを判定させて、その確率の高い順に TOP_K 件を選んでからページ全体に広げる。
ask_haiku は、同じ並べ替えを Claude Haiku 5.5(INDEX_MODEL)に 0〜10 の点数を付けさせて行う。Jev と比べるための形。

埋め込みは 3072 次元で、pgvector の vector 型の索引(HNSW / IVFFlat)は 2000 次元までしか張れない。
チャンクが数百件なので、索引なしの全件走査で十分速い。
"""

import time
from functools import cache

import numpy as np
from pydantic import BaseModel

from rag_poc.answer import Passage, Result, generate, unique_pages
from rag_poc.docs import DOCS_BY_ID, clean_pages
from rag_poc.llm import INDEX_MODEL, RERANK_MODEL, Usage, anthropic_client, typesafe_client
from rag_poc.vector_build import connect, embed

TOP_K = 8
PER_DOC = 2
RERANK_CANDIDATES = 30
HAIKU_RERANK_EFFORT = "low"

HAIKU_RERANK_SYSTEM = """質問と、検索で取った抜粋の一覧が与えられます。各抜粋に質問の答えが書かれているかを 0〜10 の点数で判定してください。

- 10: 質問の答えがそのまま書かれている
- 5: 答えの一部、または答えを探す手がかりが書かれている
- 0: 質問と関係がない

すべての抜粋に点数を付け、id と点数の組で返してください。"""


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


class _Score(BaseModel):
    id: int
    score: int


class _Scores(BaseModel):
    scores: list[_Score]


def rerank_haiku(question: str, passages: list[Passage], usage: Usage, k: int = TOP_K) -> tuple[list[Passage], list[float]]:
    """rerank と同じ並べ替えを Haiku で行う。点数のない抜粋は 0 点とし、同じ点数なら元の順。"""
    listing = "\n\n".join(f'<passage id="{i}">\n{p.text}\n</passage>' for i, p in enumerate(passages))
    response = anthropic_client().messages.parse(
        model=INDEX_MODEL,
        max_tokens=16000,
        system=HAIKU_RERANK_SYSTEM,
        output_config={"effort": HAIKU_RERANK_EFFORT},
        messages=[{"role": "user", "content": f"<passages>\n{listing}\n</passages>\n\n質問: {question}"}],
        output_format=_Scores,
    )
    usage.add_message(INDEX_MODEL, response.usage)
    given = {s.id: s.score / 10 for s in response.parsed_output.scores} if response.parsed_output else {}
    scores = [given.get(i, 0.0) for i in range(len(passages))]
    order = sorted(range(len(passages)), key=lambda i: -scores[i])[:k]
    return [passages[i] for i in order], [scores[i] for i in order]


@cache
def _pages(doc_id: str) -> list[str]:
    return clean_pages(DOCS_BY_ID[doc_id])


def expand_to_pages(passages: list[Passage]) -> list[Passage]:
    return [Passage(doc=p["doc"], page=p["page"], text=_pages(p["doc"])[p["page"] - 1]) for p in unique_pages(passages)]


def ask(question: str, expand: bool = False, per_doc: int = 0, jev: bool = False, haiku: bool = False, method: str = "vector") -> Result:
    usage = Usage()
    detail = {}
    started = time.perf_counter()
    if jev:
        passages, scores = rerank(question, search(question, usage, k=RERANK_CANDIDATES), usage)
        detail["rerank_scores"] = scores
    elif haiku:
        passages, scores = rerank_haiku(question, search(question, usage, k=RERANK_CANDIDATES), usage)
        detail["rerank_scores"] = scores
    else:
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
        detail=detail,
    )


def ask_page(question: str) -> Result:
    return ask(question, expand=True, method="vector_page")


def ask_page_doc(question: str) -> Result:
    return ask(question, expand=True, per_doc=PER_DOC, method="vector_page_doc")


def ask_jev(question: str) -> Result:
    return ask(question, expand=True, jev=True, method="vector_jev")


def ask_haiku(question: str) -> Result:
    return ask(question, expand=True, haiku=True, method="vector_haiku")
