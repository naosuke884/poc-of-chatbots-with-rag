"""比べる方式の一覧。CLI・API・計測・比較表はここから方式を引く。"""

from collections.abc import Callable

from rag_poc.answer import Passage, Result
from rag_poc.llm import Usage

# 方式 id と表示名
METHODS = {
    "vector": "ベクトル",
    "vector_page": "ベクトル(ページ拡張)",
    "vector_page_doc": "ベクトル(ページ拡張+資料枠)",
    "vector_jev": "ベクトル(Jev で並べ替え)",
    "vector_haiku": "ベクトル(Haiku で並べ替え)",
    "pageindex": "PageIndex",
}


def pipeline(method: str) -> Callable[[str], Result]:
    """方式の ask 関数を返す。使わない方式の依存(DB・目次ツリー)を読み込まないよう、ここで import する。"""
    if method == "vector":
        from rag_poc.vector_rag import ask
    elif method == "vector_page":
        from rag_poc.vector_rag import ask_page as ask
    elif method == "vector_page_doc":
        from rag_poc.vector_rag import ask_page_doc as ask
    elif method == "vector_jev":
        from rag_poc.vector_rag import ask_jev as ask
    elif method == "vector_haiku":
        from rag_poc.vector_rag import ask_haiku as ask
    elif method == "pageindex":
        from rag_poc.pageindex_rag import ask
    else:
        raise ValueError(f"unknown method: {method}")
    return ask


def retriever(method: str) -> Callable[[str, Usage], list[Passage]]:
    """方式の検索だけを行う関数を返す。回答生成に渡す抜粋を返す。"""
    if method == "pageindex":
        from rag_poc.pageindex_rag import search

        return lambda q, usage: search(q, usage)[0]
    from rag_poc.vector_rag import METHOD_OPTIONS, retrieve

    if method not in METHOD_OPTIONS:
        raise ValueError(f"unknown method: {method}")
    return lambda q, usage: retrieve(q, usage, **METHOD_OPTIONS[method])[0]
