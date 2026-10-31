"""資料を増やしたときの、1 問あたりの検索の時間・料金の伸び方を測る。

資料の量は次の段階で測る。
- 3 資料: 評価セットの資料(140 ページ)
- 11 資料: 同じガイドラインの付録・説明資料 8 冊(docs.EXTRA_DOCS)を足したもの
- 22 資料以上: 11 資料を複製したもの(合成)。複製は資料 id に「~2」などを付けた別の資料として扱う。
  中身が同じなので精度は測らず、時間と量だけを見る

ベクトル方式は、段階ごとに chunks_scale_<n> テーブルを作り、質問の埋め込み(資料の量によらない)と
DB の検索を分けて測る。DB の検索は、シンプルなベクトル検索の TOP_K 件と、Jev で並べ替える前の RERANK_CANDIDATES 件で測る。
並べ替えは候補の件数で決まり、資料の量によらないので、ここでは測らない。
PageIndex 方式は、段階ごとの目次ツリーで節の選択を行い、時間・トークン数・料金を測る。
ツリーはキャッシュするので、各段階の最初の 1 問(キャッシュ作成)とそれ以降を分けて集計する。

    uv run python -m rag_poc.scale
"""

import json
import statistics
import time

import numpy as np

from rag_poc import evalset, pageindex_rag, vector_rag
from rag_poc.docs import DATA_DIR, DOCS, DOCS_BY_ID, EXTRA_DOCS, clean_pages
from rag_poc.llm import ANSWER_MODEL, Usage, anthropic_client
from rag_poc.pageindex_build import load_tree
from rag_poc.vector_build import connect, embed, make_chunks

RESULT_PATH = DATA_DIR / "index" / "scale.json"
VECTOR_COPIES = [1, 2, 4, 8, 32]  # 11 資料の何倍で測るか
PAGEINDEX_COPIES = [1, 2, 4]
N_VECTOR_QUESTIONS = 20
N_PAGEINDEX_QUESTIONS = 10


def _copy_id(doc_id: str, copy: int) -> str:
    return doc_id if copy == 1 else f"{doc_id}~{copy}"


def _base_id(doc_id: str) -> str:
    return doc_id.split("~")[0]


def levels(copies: list[int]) -> list[tuple[str, list[str]]]:
    """(段階の名前, 資料 id のリスト)。"""
    base = [d.id for d in DOCS]
    all11 = base + [d.id for d in EXTRA_DOCS]
    out = [("3", base)]
    for c in copies:
        out.append((str(len(all11) * c), [_copy_id(d, k) for k in range(1, c + 1) for d in all11]))
    return out


def _count_pages(doc_ids: list[str]) -> int:
    return sum(len(clean_pages(DOCS_BY_ID[_base_id(d)])) for d in doc_ids)


# ベクトル方式


def build_vector_tables() -> dict:
    """11 資料のチャンクを chunks_scale_11 に入れ、複製の段階のテーブルを作る。3 資料の埋め込みは chunks から流用する。"""
    usage = Usage()
    extra = make_chunks(EXTRA_DOCS)
    started = time.perf_counter()
    for i in range(0, len(extra), 64):
        batch = extra[i : i + 64]
        for c, e in zip(batch, embed([c["text"] for c in batch], usage)):
            c["embedding"] = e
    embed_seconds = time.perf_counter() - started

    with connect() as conn:
        for name, docs in levels(VECTOR_COPIES):
            table = f"chunks_scale_{name}"
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(f"CREATE TABLE {table} (id serial PRIMARY KEY, doc text NOT NULL, page int NOT NULL, text text NOT NULL, embedding vector(3072) NOT NULL)")
        conn.execute("INSERT INTO chunks_scale_11 (doc, page, text, embedding) SELECT doc, page, text, embedding FROM chunks ORDER BY id")
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO chunks_scale_11 (doc, page, text, embedding) VALUES (%s, %s, %s, %s)", [(c["doc"], c["page"], c["text"], c["embedding"]) for c in extra])
        conn.execute("INSERT INTO chunks_scale_3 (doc, page, text, embedding) SELECT doc, page, text, embedding FROM chunks ORDER BY id")
        for c in VECTOR_COPIES[1:]:
            table = f"chunks_scale_{11 * c}"
            for k in range(1, c + 1):
                conn.execute(f"INSERT INTO {table} (doc, page, text, embedding) SELECT doc || %s, page, text, embedding FROM chunks_scale_11 ORDER BY id", ("" if k == 1 else f"~{k}",))
        counts = {name: conn.execute(f"SELECT count(*) FROM chunks_scale_{name}").fetchone()[0] for name, _ in levels(VECTOR_COPIES)}
    return {"extra_chunks": len(extra), "extra_pages": _count_pages([d.id for d in EXTRA_DOCS]), "embed_seconds": round(embed_seconds, 1), **usage.to_dict(), "chunks": counts}


def _search_ms(conn, table: str, v, k: int) -> float:
    started = time.perf_counter()
    conn.execute(f"SELECT id FROM {table} ORDER BY embedding <=> %s LIMIT %s", (v, k)).fetchall()
    return (time.perf_counter() - started) * 1000


def measure_vector(questions: list[dict]) -> list[dict]:
    usage = Usage()
    vectors, embed_ms = [], []
    for q in questions:
        started = time.perf_counter()
        [e] = embed([q["question"]], usage)
        embed_ms.append((time.perf_counter() - started) * 1000)
        vectors.append(np.array(e))

    out = []
    with connect() as conn:
        for name, docs in levels(VECTOR_COPIES):
            table = f"chunks_scale_{name}"
            _search_ms(conn, table, vectors[0], vector_rag.TOP_K)  # 最初の 1 回はテーブルの読み込みを含むので捨てる
            row = {"docs": len(docs), "pages": _count_pages(docs), "chunks": conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]}
            for key, k in (("top_k", vector_rag.TOP_K), ("rerank_candidates", vector_rag.RERANK_CANDIDATES)):
                row[key] = {"k": k, "db_ms": round(statistics.mean(_search_ms(conn, table, v, k) for v in vectors), 1)}
            out.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    return [{"embed_ms": round(statistics.mean(embed_ms)), "embed_usage": usage.to_dict()}, *out]


# PageIndex 方式


def _trees_text(doc_ids: list[str]) -> str:
    lines = []

    def walk(doc_id, items, depth):
        for n in items:
            summary = " ".join((n.get("summary") or "").split())
            lines.append(f"{'  ' * depth}- [{doc_id}:{n['node_id']}] {n['title']} (p{n['start_index']}-{n['end_index']}): {summary}")
            walk(doc_id, n.get("nodes", []), depth + 1)

    for doc_id in doc_ids:
        lines.append(f'<tree doc="{doc_id}" title="{DOCS_BY_ID[_base_id(doc_id)].title}">')
        walk(doc_id, load_tree(_base_id(doc_id))["structure"], 0)
        lines.append("</tree>")
    return "\n".join(lines)


def _count_nodes(doc_ids: list[str]) -> int:
    def count(items):
        return sum(1 + count(n.get("nodes", [])) for n in items)

    return sum(count(load_tree(_base_id(d))["structure"]) for d in doc_ids)


def _select(trees: str, question: str) -> tuple[float, Usage]:
    usage = Usage()
    system = pageindex_rag.SELECT_SYSTEM.replace("{n} つの資料", "資料").format(trees=trees)
    started = time.perf_counter()
    response = anthropic_client().messages.parse(
        model=ANSWER_MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": pageindex_rag.SELECT_EFFORT},
        messages=[{"role": "user", "content": f"質問: {question}"}],
        output_format=pageindex_rag._Selection,
    )
    elapsed = time.perf_counter() - started
    usage.add_message(ANSWER_MODEL, response.usage)
    return elapsed * 1000, usage


def measure_pageindex(questions: list[dict]) -> list[dict]:
    out = []
    for name, docs in levels(PAGEINDEX_COPIES):
        trees = _trees_text(docs)
        runs = [_select(trees, q["question"]) for q in questions]
        first_ms, first = runs[0]
        warm = runs[1:]
        t = first.tokens[ANSWER_MODEL]
        row = {
            "docs": len(docs),
            "pages": _count_pages(docs),
            "nodes": _count_nodes(docs),
            "tree_tokens": t["input"] + t["cache_read"],
            "first": {"ms": round(first_ms), "cost_usd": round(first.cost_usd, 4)},
            "warm": {
                "ms": round(statistics.mean(ms for ms, _ in warm)),
                "cost_usd": round(statistics.mean(u.cost_usd for _, u in warm), 4),
                "cache_hit": all(u.tokens[ANSWER_MODEL]["cache_read"] > 0 for _, u in warm),
            },
        }
        out.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return out


def main() -> None:
    base = evalset.load(DATA_DIR / "eval" / "questions_50.json")
    answerable = [q for q in base if q["type"] != "unanswerable"]
    step = len(answerable) // N_VECTOR_QUESTIONS
    vector_qs = answerable[::step][:N_VECTOR_QUESTIONS]
    pageindex_qs = vector_qs[:N_PAGEINDEX_QUESTIONS]

    result = {"vector_build": build_vector_tables()}
    print(json.dumps(result["vector_build"], ensure_ascii=False), flush=True)
    result["pageindex_build"] = {d.id: load_tree(d.id)["build"] for d in DOCS + EXTRA_DOCS}
    vector = measure_vector(vector_qs)
    result["vector_query"] = {"embed": vector[0], "levels": vector[1:]}
    result["pageindex_query"] = measure_pageindex(pageindex_qs)
    RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
