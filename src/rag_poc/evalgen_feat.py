"""資料の組 feat(docs.FEAT_DOCS)の評価セットを作り、data/eval/feat/questions.json に保存する。

方式の差が出やすい特徴ごとに問題を作る。category に種類を入れる。
- version: 本編の第4.0版で第3.1版から内容が変わった箇所を、版を指定せずに聞く。正解は第4.0版のページ
  第4.0版の各ページに似た第3.1版のページを本文の一致度で探し、一致度が VERSION_MAX_RATIO より低いページから作る
- structure: テレワークのガイドラインの、複数ページにわたる節(PDF の目次の第 2・第 3 階層)の内容をまとめて聞く(すべて挙げる・それぞれ答える)
- control: 版で内容が変わらない本編のページの問題。元の評価セット(data/eval/questions.json)から選び、
  対応する第3.1版のページも正解に数える(alt_evidence)

version と structure は JUDGE_MODEL で下書きし、根拠の文言を evalset.verify で確かめたあと、資料の全文を渡して点検する。
version の点検では、第3.1版では答えが違う(または書かれていない)ことも確かめ、同じ答えになる問題は除く。人は確認していない。

    RAG_CORPUS=feat uv run python -m rag_poc.evalgen_feat
"""

import difflib
import json
import random
import re
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from typing import Literal

import pymupdf
from pydantic import BaseModel

from rag_poc import evalgen, evalset
from rag_poc.docs import DATA_DIR, DOCS_BY_ID, clean_pages
from rag_poc.llm import JUDGE_MODEL, Usage, anthropic_client
from rag_poc.pageindex_build import load_tree

OUT_PATH = DATA_DIR / "eval" / "feat" / "questions.json"
DRAFTS_PATH = DATA_DIR / "eval" / "feat" / "drafts.json"
NEW, OLD, TELEWORK = "main", "main_v31", "telework"
VERSION_MAX_RATIO = 0.9  # これより一致度が低い第4.0版のページを「変わった」とみなす
CONTROL_MIN_RATIO = 0.85  # これ以上なら「変わらない」とみなす
MIN_CHARS = 300
STRUCTURE_MIN_PAGES, STRUCTURE_MAX_PAGES = 2, 6
N_CONTROL = 30
WORKERS = 4
SEED = 0

VERSION_DRAFT_SYSTEM = """あなたは RAG チャットボットの評価セットを作る担当者です。同じガイドラインの現行版(第4.0版)のページと、内容が近い旧版(第3.1版)のページが与えられます。

現行版で旧版から内容が変わった点(数値・項目・名称・手順・説明の追加や削除など)について、利用者がしそうな質問と正解を作ってください。
- 質問では版を指定しない(利用者は版を意識せずに聞く)。正解は現行版にもとづく
- 旧版の内容で答えると誤りになる質問にする。旧版と同じ答えになる質問は作らない
- ページの見出しや文言をそのまま使わず、利用者の言い回しにする
- 正解は短く、要点だけにする。evidence は現行版のページ本文からそのまま写した短い文言(10〜40 文字)
- old_answer には、旧版で答えるとどうなるか(旧版に記載がなければ「記載なし」)を書く
- type は fact(1 か所の事実)、table(表・数値・一覧)、multi(離れた 2 か所以上を合わせる)
- 表紙・目次・奥付・改訂履歴からは作らない"""

STRUCTURE_DRAFT_SYSTEM = """あなたは RAG チャットボットの評価セットを作る担当者です。ガイドラインの 1 つの節(複数ページ)が、見出しの階層とともに与えられます。

節全体を読まないと答えられない質問と正解を作ってください。
- 例: 節で挙げている対策・項目をすべて答える、役割や方式ごとにそれぞれ答える、節の中の複数の箇所を合わせて答える
- 質問は利用者がしそうな言い回しにし、節の見出しをそのまま使わない。どのガイドラインの話かは分かるようにする(「テレワークのガイドラインでは」など)
- 答えが 1 つに決まるようにする。正解は要点を漏れなく、短く並べる
- evidence は正解の根拠となるページ番号と、そのページ本文からそのまま写した短い文言(10〜40 文字)。なるべく 2 ページ以上から挙げる
- type は multi(複数箇所)か table(表・一覧の読み取り)"""

CHECK_SYSTEM = """あなたは RAG チャットボットの評価セットを点検する担当者です。<documents> は質問の対象となる資料の全文です。

次をすべて満たすときだけ ok を true にしてください。
- 正解が根拠ページの本文どおりで、質問の要点に過不足なく答えている
- 質問の答えが 1 つに決まる。聞き方があいまいだったり、別の答えも成り立ったりしない
{extra}
<documents>
{documents}
</documents>"""

VERSION_CHECK_EXTRA = """- 現行版(第4.0版)にもとづく正解であり、旧版(第3.1版)にもとづいて答えると正解と食い違う(旧版では内容が違う、または書かれていない)
"""


class Evidence(BaseModel):
    page: int
    quote: str


class _VersionDraft(BaseModel):
    type: Literal["fact", "table", "multi"]
    question: str
    answer: str
    old_answer: str
    evidence: list[Evidence]


class _StructureDraft(BaseModel):
    type: Literal["multi", "table"]
    question: str
    answer: str
    evidence: list[Evidence]


class _Drafts(BaseModel):
    questions: list[_VersionDraft]


class _StructureDrafts(BaseModel):
    questions: list[_StructureDraft]


class _Check(BaseModel):
    reason: str
    ok: bool


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", text)


@cache
def alignment() -> dict[int, tuple[int, float]]:
    """第4.0版の各ページ → (本文がいちばん近い第3.1版のページ, 一致度)。"""
    new = [_norm(t) for t in clean_pages(DOCS_BY_ID[NEW])]
    old = [_norm(t) for t in clean_pages(DOCS_BY_ID[OLD])]
    out = {}
    for i, a in enumerate(new, start=1):
        best = (0, 0.0)
        for j, b in enumerate(old, start=1):
            if a and b and difflib.SequenceMatcher(None, a, b, autojunk=False).quick_ratio() > best[1]:
                r = difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()
                if r > best[1]:
                    best = (j, r)
        out[i] = best
    return out


def _parse(model: type[BaseModel], system: str, content: str, usage: Usage):
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=system,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content}],
        output_format=model,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    return response.parsed_output


def draft_version(page: int) -> tuple[list[dict], Usage]:
    usage = Usage()
    new, old = clean_pages(DOCS_BY_ID[NEW]), clean_pages(DOCS_BY_ID[OLD])
    old_page = alignment()[page][0]
    content = (
        f'<current version="第4.0版" page="{page}">\n{new[page - 1]}\n</current>\n\n'
        + (f'<old version="第3.1版" page="{old_page}">\n{old[old_page - 1]}\n</old>\n\n' if old_page else "")
        + "質問を 1 問作ってください。旧版との違いがなく作れない場合は空のリストにしてください。"
    )
    out = _parse(_Drafts, VERSION_DRAFT_SYSTEM, content, usage)
    rows = [
        {"doc": NEW, "category": "version", "type": d.type, "question": d.question, "answer": d.answer, "old_answer": d.old_answer, "evidence": [{"doc": NEW, **e.model_dump()} for e in d.evidence]}
        for d in (out.questions[:1] if out else [])
    ]
    return rows, usage


def structure_sections() -> list[dict]:
    """テレワークの PDF の目次から、STRUCTURE_MIN_PAGES〜STRUCTURE_MAX_PAGES ページの節(第 2 階層)を選ぶ。
    長すぎて入らない節は、その下の第 3 階層の節を使う。"""
    toc = [(level, title, page) for level, title, page in pymupdf.open(DOCS_BY_ID[TELEWORK].path).get_toc()]
    n_pages = len(clean_pages(DOCS_BY_ID[TELEWORK]))
    chapter, sections = "", []

    def end_of(i, level):
        nxt = next((p for lv, _, p in toc[i + 1 :] if lv <= level), n_pages + 1)
        return max(toc[i][2], nxt - 1)

    for i, (level, title, start) in enumerate(toc):
        if level == 1:
            chapter = title
        if level not in (2, 3):
            continue
        end = end_of(i, level)
        span = end - start + 1
        parent_too_long = level == 3 and end_of(max(j for j in range(i) if toc[j][0] == 2), 2) - toc[max(j for j in range(i) if toc[j][0] == 2)][2] + 1 > STRUCTURE_MAX_PAGES
        if STRUCTURE_MIN_PAGES <= span <= STRUCTURE_MAX_PAGES and (level == 2 or parent_too_long):
            parent = "" if level == 2 else toc[max(j for j in range(i) if toc[j][0] == 2)][1] + " > "
            sections.append({"heading": f"{chapter} > {parent}{title}", "start": start, "end": end})
    return sections


def draft_structure(section: dict) -> tuple[list[dict], Usage]:
    usage = Usage()
    pages = clean_pages(DOCS_BY_ID[TELEWORK])
    n = 2 if section["end"] - section["start"] >= 2 else 1  # 3 ページ以上の節は 2 問
    content = (
        f"<heading>{section['heading']}</heading>\n\n"
        + "\n\n".join(evalgen._page_xml(TELEWORK, p, pages[p - 1]) for p in range(section["start"], section["end"] + 1))
        + f"\n\n質問を {n} 問作ってください。"
    )
    out = _parse(_StructureDrafts, STRUCTURE_DRAFT_SYSTEM, content, usage)
    rows = [
        {"doc": TELEWORK, "category": "structure", "type": d.type, "question": d.question, "answer": d.answer, "evidence": [{"doc": TELEWORK, **e.model_dump()} for e in d.evidence]}
        for d in (out.questions[:n] if out else [])
    ]
    return rows, usage


def check(q: dict, documents: str, extra: str) -> tuple[_Check, Usage]:
    usage = Usage()
    pages = ", ".join(f"{e['doc']} p{e['page']}" for e in q["evidence"])
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=[{"type": "text", "text": CHECK_SYSTEM.format(extra=extra, documents=documents), "cache_control": {"type": "ephemeral"}}],
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": f"<question>{q['question']}</question>\n<expected>{q['answer']}</expected>\n<evidence_pages>{pages}</evidence_pages>"}],
        output_format=_Check,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    return response.parsed_output or _Check(reason=f"点検に失敗した({response.stop_reason})", ok=False), usage


def _check_all(candidates: list[dict], doc_ids: list[str], extra: str, usage: Usage) -> None:
    if not candidates:
        return
    documents = evalgen._documents_text([DOCS_BY_ID[d] for d in doc_ids])
    first = check(candidates[0], documents, extra)  # キャッシュを作ってから並列にする
    with ThreadPoolExecutor(WORKERS) as pool:
        results = [first, *pool.map(lambda q: check(q, documents, extra), candidates[1:])]
    for q, (c, u) in zip(candidates, results):
        usage.merge(u)
        q["check"] = c.model_dump()


def control_questions() -> list[dict]:
    """元の評価セットの本編の問題のうち、根拠が 1 ページで、そのページが第3.1版とほぼ同じものを選ぶ。"""
    align = alignment()
    pool = [
        q
        for q in evalset.load(DATA_DIR / "eval" / "questions.json")
        if q["doc"] == NEW and q["type"] != "unanswerable" and len({e["page"] for e in q["evidence"]}) == 1 and align[q["evidence"][0]["page"]][1] >= CONTROL_MIN_RATIO
    ]
    random.Random(SEED).shuffle(pool)
    return [
        {**{k: q[k] for k in ("doc", "type", "question", "answer", "evidence")}, "category": "control", "source_id": q["id"], "alt_evidence": [{"doc": OLD, "page": align[q["evidence"][0]["page"]][0]}]}
        for q in pool[:N_CONTROL]
    ]


def main() -> None:
    usage = Usage()
    texts = clean_pages(DOCS_BY_ID[NEW])
    version_pages = [p for p, (_, r) in alignment().items() if r < VERSION_MAX_RATIO and len(texts[p - 1]) >= MIN_CHARS]
    sections = structure_sections()
    print(f"版で変わったページ {len(version_pages)}、構造の節 {len(sections)}", flush=True)

    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(draft_version, version_pages)) + list(pool.map(draft_structure, sections))
    candidates = [q for rows, _ in results for q in rows]
    for _, u in results:
        usage.merge(u)
    bad = {e.split(":")[0] for e in evalset.verify([{**q, "id": str(i)} for i, q in enumerate(candidates)])}
    candidates = [q for i, q in enumerate(candidates) if str(i) not in bad]
    print(f"下書き {len(candidates)} 問(文言の確認で {len(bad)} 問を除いた) ${usage.cost_usd:.2f}", flush=True)

    _check_all([q for q in candidates if q["category"] == "version"], [NEW, OLD], VERSION_CHECK_EXTRA, usage)
    _check_all([q for q in candidates if q["category"] == "structure"], [TELEWORK], "", usage)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    DRAFTS_PATH.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    passed = [q for q in candidates if q["check"]["ok"]]
    questions = passed + control_questions()
    out = []
    for prefix, cat in (("v", "version"), ("s", "structure"), ("c", "control")):
        for i, q in enumerate([q for q in questions if q["category"] == cat], start=1):
            out.append({"id": f"{prefix}{i:02d}", **{k: v for k, v in q.items() if k != "check"}})
    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {c: sum(q["category"] == c for q in out) for c in ("version", "structure", "control")}
    print(f"点検を通った {len(passed)} / {len(candidates)} 問。合計 {len(out)} 問 {counts}。料金 ${usage.cost_usd:.2f}")


if __name__ == "__main__":
    main()
