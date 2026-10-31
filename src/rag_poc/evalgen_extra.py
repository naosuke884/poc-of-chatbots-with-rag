"""追加の 8 冊(docs.EXTRA_DOCS)から評価用の問題を作り、data/eval/questions_extra.json に保存する。

11 冊での検索の計測(retrieval_eval)で、追加の資料にしか書かれていない答えを取れるかを見るための問題。
作り方は evalgen と同じで、2 ページずつ JUDGE_MODEL に下書きさせ、根拠の文言を evalset.verify で確かめたあと、
11 冊の全文を渡して点検する。点検では、正解が本文どおりで答えが 1 つに決まるかに加えて、
そのページだけで正解の要点がわかるほかのページ(元の 3 冊も含む)を挙げさせ、data/eval/alt_evidence_11.json に足す。
検索だけを測るので、答えのない質問は作らない。id は e01 から振る。人は確認していない。

    uv run python -m rag_poc.evalgen_extra
"""

import json
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel

from rag_poc import evalgen, evalgold, evalset
from rag_poc.docs import DATA_DIR, DOCS, EXTRA_DOCS, clean_pages
from rag_poc.llm import JUDGE_MODEL, Usage, anthropic_client

EXTRA_PATH = DATA_DIR / "eval" / "questions_extra.json"
DRAFTS_PATH = DATA_DIR / "eval" / "drafts_extra.json"
PER_WINDOW = (2, 0)  # 2 ページあたりの答えのある / ない質問の数
WORKERS = 4

CHECK_SYSTEM = """あなたは RAG チャットボットの評価セットを点検する担当者です。<documents> は質問の対象となる 11 資料の全文です。

次をすべて満たすときだけ ok を true にしてください。
- 正解が根拠ページの本文どおりで、質問の要点に過不足なく答えている
- 質問の答えが 1 つに決まる。聞き方があいまいだったり、他のページ・他の資料から別の答えも成り立ったりしない

あわせて、根拠ページ以外で、そのページだけを読めば正解の要点がすべてわかるページを other_pages に挙げてください。なければ空のリストにします。

<documents>
{documents}
</documents>"""


class _Page(BaseModel):
    doc: str
    page: int


class _Check(BaseModel):
    reason: str
    ok: bool
    other_pages: list[_Page]


def check(q: dict, documents: str) -> tuple[_Check, Usage]:
    usage = Usage()
    pages = ", ".join(f"{e['doc']} p{e['page']}" for e in q["evidence"])
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": CHECK_SYSTEM.format(documents=documents), "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": f"<question>{q['question']}</question>\n<expected>{q['answer']}</expected>\n<evidence_pages>{pages}</evidence_pages>"}],
        output_format=_Check,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    return response.parsed_output or _Check(reason=f"点検に失敗した({response.stop_reason})", ok=False, other_pages=[]), usage


def load() -> list[dict]:
    return evalset.load(EXTRA_PATH) if EXTRA_PATH.exists() else []


def main() -> None:
    usage = Usage()
    jobs = [(d.id, w, [], PER_WINDOW) for d in EXTRA_DOCS for w in evalgen.windows(d.id)]
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda j: evalgen.draft(*j), jobs))
    candidates = [q for rows, _ in results for q in rows]
    for _, u in results:
        usage.merge(u)
    print(f"下書き {len(candidates)} 問 ${usage.cost_usd:.2f}", flush=True)

    bad = {e.split(":")[0] for e in evalset.verify([{**q, "id": str(i)} for i, q in enumerate(candidates)])}
    candidates = [q for i, q in enumerate(candidates) if str(i) not in bad]
    print(f"文言の確認で {len(bad)} 問を除いた", flush=True)

    documents = evalgen._documents_text(DOCS + EXTRA_DOCS)
    first = check(candidates[0], documents)  # キャッシュを作ってから並列にする
    with ThreadPoolExecutor(WORKERS) as pool:
        checks = [first, *pool.map(lambda q: check(q, documents), candidates[1:])]
    n_pages = {d.id: len(clean_pages(d)) for d in DOCS + EXTRA_DOCS}
    for q, (c, u) in zip(candidates, checks):
        usage.merge(u)
        q["check"] = {"reason": c.reason, "ok": c.ok}
        gold = {(e["doc"], e["page"]) for e in q["evidence"]}
        q["other_pages"] = [{"doc": d, "page": p} for d, p in sorted({(x.doc, x.page) for x in c.other_pages if 1 <= x.page <= n_pages.get(x.doc, 0)} - gold)]
    DRAFTS_PATH.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = [q for q in candidates if q["check"]["ok"]]
    added = [{"id": f"e{i:02d}", **{k: q[k] for k in ("doc", "type", "question", "answer", "evidence")}} for i, q in enumerate(passed, start=1)]
    EXTRA_PATH.write_text(json.dumps(added, ensure_ascii=False, indent=2), encoding="utf-8")
    alt = evalgold.load()
    for a, q in zip(added, passed):
        alt[a["id"]] = {"pages": q["other_pages"], "reason": q["check"]["reason"]}
    evalgold.ALT_PATH.write_text(json.dumps(alt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"点検を通った {len(added)} / {len(candidates)} 問。料金 ${usage.cost_usd:.2f}")


if __name__ == "__main__":
    main()
