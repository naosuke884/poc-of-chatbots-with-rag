"""Allganize の日本語 RAG 評価データセット(RAG-Evaluation-Dataset-JA)を、この PoC の資料と評価セットの形にする。

data/allganize/ に documents.csv・rag_evaluation_result.csv と、documents.csv の URL から取った PDF(pdf/<業種>/)を置いてから実行する。
PDF が開けないもの・ページ数が documents.csv と違うもの(版が違うおそれがある)は除き、その資料が正解の問題も除く。

- data/allganize/docs.json: 使う資料(業種・id・題名)。docs.AG_DOCS がこれを読む
- data/eval/ag_<業種>/questions.json: 業種ごとの評価セット。type は正解の箇所の種類(paragraph・table・image)。ag_all は 5 業種をまとめたもの

    uv run python -m rag_poc.allganize
"""

import csv
import json
from collections import Counter

import pymupdf

from rag_poc.docs import DATA_DIR

SRC = DATA_DIR / "allganize"


def _usable(row: dict) -> bool:
    path = SRC / "pdf" / row["domain"] / row["file_name"]
    if not path.exists():
        return False
    try:
        with pymupdf.open(path) as pdf:
            for page in pdf:
                page.get_text()
            return len(pdf) == int(row["page"])
    except Exception:
        return False


def main() -> None:
    rows = list(csv.DictReader((SRC / "documents.csv").open(encoding="utf-8")))
    docs, ids = [], {}
    for domain in sorted({r["domain"] for r in rows}):
        for i, r in enumerate([r for r in rows if r["domain"] == domain and _usable(r)], start=1):
            doc_id = f"{domain}_{i:02d}"
            ids[(domain, r["file_name"])] = doc_id
            docs.append({"domain": domain, "id": doc_id, "title": r["title"], "label": r["title"][:20], "filename": f"allganize/pdf/{domain}/{r['file_name']}"})
    (SRC / "docs.json").write_text(json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8")

    questions: dict[str, list[dict]] = {}
    for q in csv.DictReader((SRC / "rag_evaluation_result.csv").open(encoding="utf-8")):
        doc_id = ids.get((q["domain"], q["target_file_name"]))
        if doc_id is None:
            continue
        rows_ = questions.setdefault(q["domain"], [])
        rows_.append(
            {
                "id": f"{q['domain'][:3]}{len(rows_) + 1:02d}",
                "doc": doc_id,
                "type": q["type"],
                "question": q["question"],
                "answer": q["target_answer"],
                "evidence": [{"doc": doc_id, "page": int(p)} for p in q["target_page_no"].replace(" ", "").split(",") if p],
            }
        )
    for name, qs in [*questions.items(), ("all", [q for qs in questions.values() for q in qs])]:
        path = DATA_DIR / "eval" / f"ag_{name}" / "questions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(qs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"資料 {len(docs)} / {len(rows)} 件({dict(Counter(d['domain'] for d in docs))})")
    print(f"問題 {sum(len(v) for v in questions.values())} 問({ {k: len(v) for k, v in questions.items()} }、{dict(Counter(q['type'] for v in questions.values() for q in v))})")


if __name__ == "__main__":
    main()
