"""評価セット(data/eval/questions.json)の読み込みと検証。

各質問の evidence は、正解の根拠となる資料・ページ(物理ページ、1 始まり)と、
そのページに含まれる文言(quote)の組。検証では quote がそのページの抽出テキストに
含まれることを確かめ、ページ番号の誤りを見つける。空白と改行は比較から除く。
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from rag_poc.docs import DATA_DIR, DOCS, extract_pages

EVAL_PATH = DATA_DIR / "eval" / "questions.json"
TYPES = {"fact", "multi", "table", "unanswerable"}


def load(path: Path = EVAL_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text)


def verify(questions: list[dict]) -> list[str]:
    """問題点を文字列のリストで返す。空なら問題なし。"""
    errors = []
    pages = {doc.id: [_normalize(p) for p in extract_pages(doc)] for doc in DOCS}

    ids = [q["id"] for q in questions]
    for dup in [i for i, n in Counter(ids).items() if n > 1]:
        errors.append(f"{dup}: id が重複している")

    for q in questions:
        qid = q["id"]
        if q["type"] not in TYPES:
            errors.append(f"{qid}: type が不正: {q['type']}")
        if q["type"] == "unanswerable":
            if q["evidence"]:
                errors.append(f"{qid}: 答えのない質問に evidence がある")
            continue
        if not q["evidence"]:
            errors.append(f"{qid}: evidence がない")
        for ev in q["evidence"]:
            doc_pages = pages[ev["doc"]]
            if not 1 <= ev["page"] <= len(doc_pages):
                errors.append(f"{qid}: {ev['doc']} p{ev['page']} は範囲外")
                continue
            if _normalize(ev["quote"]) not in doc_pages[ev["page"] - 1]:
                errors.append(f"{qid}: {ev['doc']} p{ev['page']} に「{ev['quote']}」がない")
    return errors


def main() -> int:
    questions = load()
    errors = verify(questions)
    for e in errors:
        print(e)
    print(f"{len(questions)} 問 / doc: {dict(Counter(q['doc'] for q in questions))} / type: {dict(Counter(q['type'] for q in questions))}")
    print("OK" if not errors else f"NG: {len(errors)} 件")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
