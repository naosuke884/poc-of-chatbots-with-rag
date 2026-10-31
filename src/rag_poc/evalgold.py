"""11 冊で計測するために、評価セットの正解ページを補う。

追加の 8 冊(docs.EXTRA_DOCS)にも答えが書かれている問題があるため、JUDGE_MODEL に 8 冊の全文を渡し、
問題ごとに「そのページだけで正解の要点がすべてわかるページ」を挙げさせる。答えのない質問は、
8 冊のどこかに答えがあればそのページを挙げさせ、11 冊の計測からは除く。

結果は data/eval/alt_evidence_11.json に {問題 id: {pages, reason}} の形で保存する。

    uv run python -m rag_poc.evalgold
"""

import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from rag_poc import evalset
from rag_poc.docs import DATA_DIR, EXTRA_DOCS, clean_pages
from rag_poc.llm import JUDGE_MODEL, Usage, anthropic_client

ALT_PATH = DATA_DIR / "eval" / "alt_evidence_11.json"
WORKERS = 4

SYSTEM = """あなたは RAG チャットボットの評価セットを点検する担当者です。<documents> は、評価セットを作ったあとに検索の対象に加えた資料の全文です。

質問と正解(<expected>)が与えられます。<documents> の中で、そのページだけを読めば正解の要点がすべてわかるページを挙げてください。
- 要点の一部しか書かれていないページ、話題が近いだけのページは挙げない
- 正解と食い違う内容のページは挙げない
- 該当するページがなければ空のリストにする

<expected> が「答えなし」の質問では、<documents> のどこかに質問の答えが書かれていれば、そのページを挙げてください。

<documents>
{documents}
</documents>"""


class _Page(BaseModel):
    doc: str
    page: int


class _Alt(BaseModel):
    reason: str
    pages: list[_Page]


def _documents_text() -> str:
    return "\n\n".join(
        f'<document id="{d.id}" title="{d.title}">\n'
        + "\n".join(f'<page doc="{d.id}" number="{i}">\n{t}\n</page>' for i, t in enumerate(clean_pages(d), start=1))
        + "\n</document>"
        for d in EXTRA_DOCS
    )


def find(q: dict, documents: str) -> tuple[_Alt, Usage]:
    usage = Usage()
    expected = "答えなし" if q["type"] == "unanswerable" else q["answer"]
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": SYSTEM.format(documents=documents), "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": f"<question>{q['question']}</question>\n<expected>{expected}</expected>"}],
        output_format=_Alt,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    return response.parsed_output or _Alt(reason=f"判定に失敗した({response.stop_reason})", pages=[]), usage


def load() -> dict:
    return json.loads(ALT_PATH.read_text(encoding="utf-8"))


def main() -> None:
    questions = evalset.load()
    documents = _documents_text()
    n_pages = {d.id: len(clean_pages(d)) for d in EXTRA_DOCS}
    first = find(questions[0], documents)  # キャッシュを作ってから並列にする
    with ThreadPoolExecutor(WORKERS) as pool:
        results = [first, *pool.map(lambda q: find(q, documents), questions[1:])]

    usage = Usage()
    out = {}
    for q, (alt, u) in zip(questions, results):
        usage.merge(u)
        pages = sorted({(p.doc, p.page) for p in alt.pages if 1 <= p.page <= n_pages.get(p.doc, 0)})
        out[q["id"]] = {"pages": [{"doc": d, "page": p} for d, p in pages], "reason": alt.reason}
    ALT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    by_type = {}
    for q in questions:
        by_type.setdefault(q["type"], []).append(bool(out[q["id"]]["pages"]))
    print({t: f"{sum(v)}/{len(v)}" for t, v in by_type.items()}, f"${usage.cost_usd:.2f}")


if __name__ == "__main__":
    main()
