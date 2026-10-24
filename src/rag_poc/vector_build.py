"""ベクトル方式の索引作成。資料をチャンクに分割し、埋め込みを PostgreSQL(pgvector)に登録する。

チャンクはページをまたがない。1 チャンクが必ず 1 つの物理ページに対応するので、
検索結果をそのまま根拠ページとして扱える。ページ内は行の区切りで CHUNK_CHARS 文字前後に分け、
前のチャンクの末尾 OVERLAP_CHARS 文字ぶんを次のチャンクの先頭に重ねる。

    uv run python -m rag_poc.vector_build
"""

import json
import os
import re
import time

import psycopg
from pgvector.psycopg import register_vector

from rag_poc.docs import DATA_DIR, DOCS, Doc, clean_pages
from rag_poc.llm import EMBED_MODEL, Usage, openai_client

CHUNK_CHARS = 600
OVERLAP_CHARS = 150
EMBED_DIM = 3072
EMBED_BATCH = 64
# ページ番号だけのページなど、これより短いチャンクは登録しない
MIN_CHARS = 30
STATS_PATH = DATA_DIR / "index" / "vector_build.json"

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    id serial PRIMARY KEY,
    doc text NOT NULL,
    page int NOT NULL,
    seq int NOT NULL,
    text text NOT NULL,
    embedding vector({EMBED_DIM}) NOT NULL
);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def _tidy(text: str) -> str:
    # 空白だけの行を除き、行内の連続する空白を 1 つにする
    lines = [re.sub(r"[ \t　]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def split_page(text: str) -> list[str]:
    """1 ページのテキストを、行の区切りで CHUNK_CHARS 文字前後のチャンクに分ける。"""
    text = _tidy(text)
    if len(text) <= CHUNK_CHARS:
        return [text] if text else []
    chunks, current = [], ""
    for line in text.split("\n"):
        if current and len(current) + len(line) + 1 > CHUNK_CHARS:
            chunks.append(current)
            current = current[-OVERLAP_CHARS:]
        current = f"{current}\n{line}" if current else line
    if current and (not chunks or len(current) > OVERLAP_CHARS):
        chunks.append(current)
    return chunks


def make_chunks(docs: list[Doc] = DOCS) -> list[dict]:
    chunks = []
    for doc in docs:
        for page, text in enumerate(clean_pages(doc), start=1):
            for seq, chunk in enumerate(c for c in split_page(text) if len(c) >= MIN_CHARS):
                chunks.append({"doc": doc.id, "page": page, "seq": seq, "text": chunk})
    return chunks


def embed(texts: list[str], usage: Usage) -> list[list[float]]:
    res = openai_client().embeddings.create(model=EMBED_MODEL, input=texts)
    usage.add(EMBED_MODEL, input=res.usage.prompt_tokens)
    return [d.embedding for d in res.data]


def build() -> dict:
    chunks = make_chunks()
    usage = Usage()
    started = time.perf_counter()
    for i in range(0, len(chunks), EMBED_BATCH):
        batch = chunks[i : i + EMBED_BATCH]
        for c, e in zip(batch, embed([c["text"] for c in batch], usage)):
            c["embedding"] = e
    elapsed = time.perf_counter() - started

    with connect() as conn:
        conn.execute("DROP TABLE IF EXISTS chunks")
        conn.execute(SCHEMA)
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks (doc, page, seq, text, embedding) VALUES (%s, %s, %s, %s, %s)",
                [(c["doc"], c["page"], c["seq"], c["text"], c["embedding"]) for c in chunks],
            )

    lengths = [len(c["text"]) for c in chunks]
    stats = {
        "chunks": len(chunks),
        "chunk_chars": CHUNK_CHARS,
        "overlap_chars": OVERLAP_CHARS,
        "avg_chars": round(sum(lengths) / len(lengths)),
        "max_chars": max(lengths),
        "embed_seconds": round(elapsed, 1),
        **usage.to_dict(),
    }
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    return stats


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False))
