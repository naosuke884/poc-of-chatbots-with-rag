"""図・画像の中の文字と内容を、ページ画像から Claude(OCR_MODEL)で書き出す前処理。

PDF から取り出せる文字だけでは、図・グラフ・画像の中に書かれた答えを拾えない。図や画像のあるページを画像にして
OCR_MODEL に渡し、図中の文字と図が示す内容(グラフの数値・表・関係)を書き出させて data/ocr/<doc>.json に保存する。
環境変数 RAG_OCR=1 のとき、docs.clean_pages がこの書き出しをページの本文の後ろに足す。

対象のページは、画像を含む・線や図形が多い(DRAWINGS_MIN 個以上)・文字が少ない(TEXT_MIN 文字未満)のどれかに当てはまるもの。

    RAG_CORPUS=ag_finance uv run python -m rag_poc.ocr   # その資料の組の全資料
"""

import base64
import json
from concurrent.futures import ThreadPoolExecutor

import pymupdf

from rag_poc.docs import CORPUS_DOCS, Doc, extract_pages, ocr_path
from rag_poc.llm import INDEX_MODEL, Usage, anthropic_client

OCR_MODEL = INDEX_MODEL  # Claude Haiku 5.5
DPI = 110
DRAWINGS_MIN = 30
TEXT_MIN = 200
WORKERS = 8

PROMPT = """このページの画像から、図・グラフ・表・写真・イラストの中に書かれている文字と、それらが示す内容を書き出してください。

- グラフは、項目ごとの数値・単位・年度などを読み取れる範囲で書く。表は行と列の対応が分かるように書く
- 図は、矢印や枠の関係(何が何につながるか、どの順番か)を文章で書く
- 下の <extracted_text> は PDF から取り出せた本文です。そこにすでにある文章は繰り返さない
- 図・画像がない、または本文以外に書き出すことがなければ「なし」とだけ書く
- 前置きや説明は付けず、書き出した内容だけを日本語で書く

<extracted_text>
{text}
</extracted_text>"""


def needs_ocr(page: pymupdf.Page) -> bool:
    return bool(page.get_images()) or len(page.get_drawings()) >= DRAWINGS_MIN or len(page.get_text()) < TEXT_MIN


def _transcribe(png: bytes, text: str, usage: Usage) -> str:
    response = anthropic_client().messages.create(
        model=OCR_MODEL,
        max_tokens=4000,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.standard_b64encode(png).decode()}},
                    {"type": "text", "text": PROMPT.format(text=text)},
                ],
            }
        ],
    )
    usage.add_message(OCR_MODEL, response.usage)
    out = "".join(b.text for b in response.content if b.type == "text").strip()
    return "" if out in ("なし", "") else out


def run(doc: Doc) -> Usage:
    """資料の対象ページを書き出して保存する。保存済みのページは飛ばす。"""
    path = ocr_path(doc.id)
    done = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    texts = extract_pages(doc)
    with pymupdf.open(doc.path) as pdf:
        jobs = [(i, pdf[i].get_pixmap(dpi=DPI).tobytes("png")) for i in range(len(pdf)) if str(i + 1) not in done and needs_ocr(pdf[i])]
    usages = [Usage() for _ in jobs]
    with ThreadPoolExecutor(WORKERS) as pool:
        outs = list(pool.map(lambda j: _transcribe(j[0][1], texts[j[0][0]], j[1]), zip(jobs, usages)))
    for (i, _), out in zip(jobs, outs):
        done[str(i + 1)] = out
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(done, ensure_ascii=False, indent=2), encoding="utf-8")
    usage = Usage()
    for u in usages:
        usage.merge(u)
    print(doc.id, f"{len(jobs)} ページ", f"${usage.cost_usd:.4f}", flush=True)
    return usage


def main() -> None:
    usage = Usage()
    for doc in CORPUS_DOCS:
        usage.merge(run(doc))
    print(f"合計 ${usage.cost_usd:.3f}")


if __name__ == "__main__":
    main()
