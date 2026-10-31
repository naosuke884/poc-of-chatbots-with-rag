"""評価セットに追加する問題を LLM で下書きし、点検して data/eval/questions.json に足す。

1. 下書き: 資料の 2 ページずつを JUDGE_MODEL に渡し、利用者がしそうな言い回しの質問・正解・根拠を作らせる。
   答えのない質問は、ページの話題に近いが資料に書かれていない細部を問うものにする
2. 点検: 3 資料の全文を渡して、正解が本文どおりか・答えが 1 つに決まるか(答えのない質問は、
   どの資料にも答えがないか)を確かめさせる。根拠の文言がページにあるかは evalset.verify で確かめる
3. 選択: 点検を通った問題から、資料・種類ごとの問題数(TARGETS)だけ選び、q51 から番号を振る

元の 50 問は questions_50.json に退避し、何度実行してもその 50 問に足す。人は確認していない。

    uv run python -m rag_poc.evalgen
"""

import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from pydantic import BaseModel

from rag_poc import evalset
from rag_poc.docs import DATA_DIR, DOCS, DOCS_BY_ID, Doc, clean_pages
from rag_poc.llm import JUDGE_MODEL, Usage, anthropic_client

BASE_PATH = DATA_DIR / "eval" / "questions_50.json"
DRAFTS_PATH = DATA_DIR / "eval" / "drafts.json"
MIN_CHARS = 300  # これより短いページ(表紙・白紙など)からは作らない
WORKERS = 4
SEED = 0

# 資料ごとの追加する問題数(種類別)と、2 ページあたりに下書きさせる問題数(答えのある / ない)
TARGETS = {
    "main": {"fact": 23, "table": 9, "multi": 7, "unanswerable": 6},
    "app5": {"fact": 20, "table": 8, "multi": 6, "unanswerable": 6},
    "app3": {"fact": 7, "table": 3, "multi": 2, "unanswerable": 3},
}
PER_WINDOW = {"main": (2, 1), "app5": (2, 1), "app3": (7, 2)}

DRAFT_SYSTEM = """あなたは RAG チャットボットの評価セットを作る担当者です。資料のページ本文から、利用者がしそうな質問と正解を作ってください。

- 質問は、資料を読んでいない利用者が実際にしそうな聞き方にする。ページの見出しや文言をそのまま使わず、言い換える(例: 「情報資産管理台帳」→「会社の情報資産を一覧にまとめる表」)
- ただし答えが 1 つに決まるようにする。同じ話題が本編・付録の他の資料にもありうるときは、「規程のサンプルでは」「自社診断で」のように対象の資料が分かる聞き方にする
- 正解は短く、要点だけにする。ページ本文だけから言えることにする
- type は fact(1 か所の事実)、table(表・数値・一覧の読み取り)、multi(2 ページ、または 1 ページ内の離れた 2 か所以上を合わせて答える)
- evidence は正解の根拠となるページ番号と、そのページ本文からそのまま写した短い文言(10〜40 文字)。multi は 2 つ以上
- 表紙・目次・奥付からは作らない
- unanswerable は、ページの話題に近いが資料に書かれていない細部(ページにない製品名・金額・頻度など)を問う、利用者が聞きそうな質問
- <existing> にある質問と同じことは聞かない"""

CHECK_SYSTEM = """あなたは RAG チャットボットの評価セットを点検する担当者です。<documents> は質問の対象となる 3 資料の全文です。

答えのある質問(<expected> がある)は、次をすべて満たすときだけ ok を true にしてください。
- 正解が根拠ページの本文どおりで、質問の要点に過不足なく答えている
- 質問の答えが 1 つに決まる。聞き方があいまいだったり、他のページ・他の資料から別の答えも成り立ったりしない

答えのない質問(<expected> がない)は、3 資料のどこにも答えが書かれていないときだけ ok を true にしてください。

<documents>
{documents}
</documents>"""


class Evidence(BaseModel):
    page: int
    quote: str


class Draft(BaseModel):
    type: Literal["fact", "table", "multi"]
    question: str
    answer: str
    evidence: list[Evidence]


class Drafts(BaseModel):
    answerable: list[Draft]
    unanswerable: list[str]


class Check(BaseModel):
    reason: str
    ok: bool


def _page_xml(doc_id: str, page: int, text: str) -> str:
    return f'<page doc="{doc_id}" number="{page}">\n{text}\n</page>'


def windows(doc_id: str) -> list[list[int]]:
    """本文のあるページを 2 ページずつに分ける(ページ番号は 1 始まり)。"""
    pages = [i for i, t in enumerate(clean_pages(DOCS_BY_ID[doc_id]), start=1) if len(t) >= MIN_CHARS]
    return [pages[i : i + 2] for i in range(0, len(pages), 2)]


def draft(doc_id: str, pages: list[int], existing: list[str], per_window: tuple[int, int] | None = None) -> tuple[list[dict], Usage]:
    usage = Usage()
    doc = DOCS_BY_ID[doc_id]
    texts = clean_pages(doc)
    n_ans, n_unans = per_window or PER_WINDOW[doc_id]
    content = (
        f"<document>{doc.title}</document>\n<existing>\n" + "\n".join(existing) + "\n</existing>\n\n"
        + "\n\n".join(_page_xml(doc_id, p, texts[p - 1]) for p in pages)
        + f"\n\n答えのある質問を {n_ans} 問まで、答えのない質問を {n_unans} 問作ってください。"
    )
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=DRAFT_SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content}],
        output_format=Drafts,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    out = response.parsed_output
    if out is None:
        return [], usage
    rows = [
        {"doc": doc_id, "type": d.type, "question": d.question, "answer": d.answer, "evidence": [{"doc": doc_id, **e.model_dump()} for e in d.evidence]}
        for d in out.answerable[:n_ans]
    ]
    rows += [{"doc": doc_id, "type": "unanswerable", "question": q, "answer": "資料に記載がない", "evidence": []} for q in out.unanswerable[:n_unans]]
    return rows, usage


def _documents_text(docs: list[Doc] = DOCS) -> str:
    return "\n\n".join(
        f'<document id="{d.id}" title="{d.title}">\n' + "\n".join(_page_xml(d.id, i, t) for i, t in enumerate(clean_pages(d), start=1)) + "\n</document>"
        for d in docs
    )


def check(q: dict, documents: str) -> tuple[Check, Usage]:
    usage = Usage()
    content = f"<question>{q['question']}</question>"
    if q["type"] != "unanswerable":
        pages = ", ".join(f"{e['doc']} p{e['page']}" for e in q["evidence"])
        content += f"\n<expected>{q['answer']}</expected>\n<evidence_pages>{pages}</evidence_pages>"
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": CHECK_SYSTEM.format(documents=documents), "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content}],
        output_format=Check,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    return response.parsed_output or Check(reason=f"点検に失敗した({response.stop_reason})", ok=False), usage


def select(candidates: list[dict]) -> list[dict]:
    """点検を通った問題から、TARGETS の数だけ選ぶ。足りない種類の分は fact で埋める。"""
    rng = random.Random(SEED)
    chosen = []
    for doc_id, targets in TARGETS.items():
        pool = {t: [q for q in candidates if q["doc"] == doc_id and q["type"] == t] for t in targets}
        for t in pool:
            rng.shuffle(pool[t])
        picked = {t: pool[t][:n] for t, n in targets.items()}
        short = sum(n - len(picked[t]) for t, n in targets.items())
        picked["fact"] += pool["fact"][targets["fact"] : targets["fact"] + short]
        chosen += [q for t in targets for q in picked[t]]
    return chosen


def main() -> None:
    usage = Usage()
    if not BASE_PATH.exists():
        shutil.copy(evalset.EVAL_PATH, BASE_PATH)
    base = evalset.load(BASE_PATH)

    jobs = [(d.id, w, [q["question"] for q in base if q["doc"] == d.id]) for d in DOCS for w in windows(d.id)]
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(lambda j: draft(*j), jobs))
    candidates = [q for rows, u in results for q in rows]
    for _, u in results:
        usage.merge(u)
    print(f"下書き {len(candidates)} 問 ${usage.cost_usd:.2f}", flush=True)

    # 根拠の文言がページにない問題を除く
    bad = {e.split(":")[0] for e in evalset.verify([{**q, "id": str(i)} for i, q in enumerate(candidates)])}
    candidates = [q for i, q in enumerate(candidates) if str(i) not in bad]
    print(f"文言の確認で {len(bad)} 問を除いた", flush=True)

    documents = _documents_text()
    check(candidates[0], documents)  # キャッシュを作ってから並列にする
    with ThreadPoolExecutor(WORKERS) as pool:
        checks = list(pool.map(lambda q: check(q, documents), candidates))
    for q, (c, u) in zip(candidates, checks):
        q["check"] = c.model_dump()
        usage.merge(u)
    DRAFTS_PATH.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")
    passed = [q for q in candidates if q["check"]["ok"]]
    print(f"点検を通った {len(passed)} / {len(candidates)} 問 ${usage.cost_usd:.2f}", flush=True)

    chosen = select(passed)
    added = [{"id": f"q{len(base) + i}", **{k: q[k] for k in ("doc", "type", "question", "answer", "evidence")}} for i, q in enumerate(chosen, start=1)]
    evalset.EVAL_PATH.write_text(json.dumps(base + added, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(added)} 問を足して {len(base) + len(added)} 問にした。料金 ${usage.cost_usd:.2f}")


if __name__ == "__main__":
    main()
