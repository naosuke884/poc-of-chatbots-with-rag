"""PageIndex(OSS)で資料ごとの目次ツリーを作り、data/index/pageindex/<doc>.json に保存する。

索引作成の LLM は Claude Haiku 5.5。PageIndex は LiteLLM 経由で呼ぶので、
LiteLLM のコールバックでトークン数を集計し、作成にかかった時間と料金も一緒に保存する。
ページのテキストは PDF から直接ではなく clean_pages(余白の章見出しを除いたもの)を渡す。

    uv run python -m rag_poc.pageindex_build [doc ...]
"""

import json
import sys
import time

import litellm
from pageindex.page_index_classic import page_index_main
from pageindex.utils import ConfigLoader, count_tokens

from rag_poc.docs import DATA_DIR, DOCS, DOCS_BY_ID, clean_pages
from rag_poc.llm import INDEX_MODEL, Usage

INDEX_DIR = DATA_DIR / "index" / "pageindex"


def tree_path(doc_id: str):
    return INDEX_DIR / f"{doc_id}.json"


def build(doc_id: str) -> dict:
    doc = DOCS_BY_ID[doc_id]
    usage = Usage()

    def on_success(kwargs, response, start, end):
        u = response.usage
        usage.add(INDEX_MODEL, input=u.prompt_tokens, output=u.completion_tokens)

    litellm.success_callback = [on_success]
    litellm.callbacks = []
    opt = ConfigLoader().load({"model": f"anthropic/{INDEX_MODEL}", "if_add_node_summary": "yes", "if_add_node_text": "no"})
    page_list = [(t, count_tokens(t)) for t in clean_pages(doc)]

    started = time.perf_counter()
    result = page_index_main(str(doc.path), opt, page_list=page_list)
    elapsed = time.perf_counter() - started
    time.sleep(1)  # 非同期のコールバックが残っていれば待つ

    result["doc"] = doc_id
    result["build"] = {"seconds": round(elapsed, 1), "pages": len(page_list), **usage.to_dict()}
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    tree_path(doc_id).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def load_tree(doc_id: str) -> dict:
    return json.loads(tree_path(doc_id).read_text(encoding="utf-8"))


def main(argv: list[str]) -> None:
    for doc_id in argv or [d.id for d in DOCS]:
        result = build(doc_id)
        print(doc_id, json.dumps(result["build"], ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
