"""評価セットの全問を 1 つの方式で実行し、採点して data/eval/results/<method>.json に保存する。

方式ごとの速度を比べられるように、質問は 1 問ずつ順に実行する。
採点は 2 つ:
- 回答の正しさ: 正解と回答を JUDGE_MODEL に比べさせ、correct / partial / incorrect を付ける。
  答えのない質問は、資料に記載がないと答えていれば correct とする
- 根拠ページの一致: 正解の根拠ページのうち、検索で取得したページ・回答が挙げたページに入っていた割合

採点の誤りを見つけるため、回答の正しさは JUDGE_RUNS 回採点する。採点には正解の根拠ページの本文も渡し、
回答の補足が資料どおりかを確かめられるようにする。判定が食い違った問題は低いほうの判定を採り、
needs_review を付けて人が見る対象にする。

    uv run python -m rag_poc.evaluate pageindex [--ids q01,q02]
    uv run python -m rag_poc.evaluate pageindex --rejudge   # 保存済みの回答を採点し直す
"""

import argparse
import json
import time
from typing import Literal

from pydantic import BaseModel

from rag_poc import evalset
from rag_poc.docs import DATA_DIR, DOCS_BY_ID, clean_pages
from rag_poc.llm import JUDGE_MODEL, Usage, anthropic_client
from rag_poc.methods import METHODS, pipeline

RESULTS_DIR = DATA_DIR / "eval" / "results"
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}
JUDGE_RUNS = 2

JUDGE_SYSTEM = """あなたは RAG チャットボットの回答を採点する評価者です。質問・正解・回答を比べて判定してください。

- correct: 正解の要点をすべて含み、誤りがない。表現の違いや、正しい補足情報は減点しない
- partial: 要点の一部だけ合っている、または要点は合っているが誤った内容も含む
- incorrect: 要点が合っていない、または資料に答えがあるのに「記載がない」と答えた

正解が「資料に記載がない」質問では、回答が記載がないと答えていれば correct、資料にない内容で答えていれば incorrect です。

<evidence> には正解の根拠となる資料のページ本文があります。正解にない補足は、evidence と合っていれば誤りとせず、evidence と食い違うときだけ誤りとしてください。"""


class Judgement(BaseModel):
    reason: str
    verdict: Literal["correct", "partial", "incorrect"]


def _evidence_text(q: dict) -> str:
    pages = sorted({(e["doc"], e["page"]) for e in q["evidence"]})
    return "\n\n".join(f'<page doc="{d}" page="{p}">\n{clean_pages(DOCS_BY_ID[d])[p - 1]}\n</page>' for d, p in pages)


def judge(q: dict, answer: str, usage: Usage) -> Judgement:
    expected = "資料に記載がない(答えられない質問)" if q["type"] == "unanswerable" else q["answer"]
    content = f"<question>{q['question']}</question>\n<expected>{expected}</expected>\n<evidence>\n{_evidence_text(q)}\n</evidence>\n<answer>{answer}</answer>"
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": content}],
        output_format=Judgement,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    return response.parsed_output or Judgement(reason=f"採点に失敗した({response.stop_reason})", verdict="incorrect")


def grade(q: dict, answer: str) -> dict:
    """JUDGE_RUNS 回採点し、行に入れる採点の項目を返す。"""
    usage = Usage()
    judgements = [judge(q, answer, usage) for _ in range(JUDGE_RUNS)]
    verdict = min((j.verdict for j in judgements), key=lambda v: SCORES[v])
    return {
        "verdict": verdict,
        "score": SCORES[verdict],
        "needs_review": len({j.verdict for j in judgements}) > 1,
        "judge_reason": next(j.reason for j in judgements if j.verdict == verdict),
        "judgements": [j.model_dump() for j in judgements],
        "judge_usage": usage.to_dict(),
    }


def page_recall(gold: list[dict], pages: list[dict]) -> float | None:
    """正解の根拠ページのうち pages に入っている割合。答えのない質問は None。"""
    gold_set = {(e["doc"], e["page"]) for e in gold}
    if not gold_set:
        return None
    return len(gold_set & {(p["doc"], p["page"]) for p in pages}) / len(gold_set)


def run(method: str, ids: list[str] | None = None) -> list[dict]:
    """1 問ごとに途中経過を <method>.progress.json に書く。途中で止まったら、次の実行で続きから行う。"""
    ask = pipeline(method)
    progress = RESULTS_DIR / f"{method}.progress.json"
    rows = json.loads(progress.read_text(encoding="utf-8")) if progress.exists() and not ids else []
    done = {r["id"] for r in rows}
    if done:
        print(f"{len(done)} 問は途中経過から再開する", flush=True)

    questions = [q for q in evalset.load() if (not ids or q["id"] in ids) and q["id"] not in done]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for q in questions:
        result = ask(q["question"]).to_dict()
        g = grade(q, result["answer"])
        rows.append(
            {
                "id": q["id"],
                "doc": q["doc"],
                "type": q["type"],
                **g,
                "retrieved_recall": page_recall(q["evidence"], result["retrieved"]),
                "cited_recall": page_recall(q["evidence"], result["sources"]),
                "result": result,
            }
        )
        if not ids:
            progress.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        print(f"{q['id']} {g['verdict']:9}{' review' if g['needs_review'] else ''} retrieved={rows[-1]['retrieved_recall']} cited={rows[-1]['cited_recall']} {result['total_ms']}ms ${result['usage']['cost_usd']:.4f}", flush=True)
    return rows


def summarize(rows: list[dict]) -> dict:
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    def group(key):
        return {k: {"n": len(g), "score": mean([r["score"] for r in g])} for k in sorted({r[key] for r in rows}) for g in [[r for r in rows if r[key] == k]]}

    usage = Usage()
    for r in rows:
        for model, t in r["result"]["usage"]["tokens"].items():
            usage.add(model, **t)
    ms = sorted(r["result"]["total_ms"] for r in rows)
    return {
        "n": len(rows),
        "score": mean([r["score"] for r in rows]),
        "verdicts": {v: sum(r["verdict"] == v for r in rows) for v in SCORES},
        "needs_review": [r["id"] for r in rows if r.get("needs_review")],
        "by_type": group("type"),
        "by_doc": group("doc"),
        "retrieved_recall": mean([r["retrieved_recall"] for r in rows]),
        "cited_recall": mean([r["cited_recall"] for r in rows]),
        "latency_ms": {
            "mean": round(sum(ms) / len(ms)),
            "p50": ms[len(ms) // 2],
            "max": ms[-1],
            "retrieval_mean": round(sum(r["result"]["retrieval_ms"] for r in rows) / len(rows)),
            "generation_mean": round(sum(r["result"]["generation_ms"] for r in rows) / len(rows)),
        },
        "usage": usage.to_dict(),
        "cost_per_question_usd": round(usage.cost_usd / len(rows), 5),
        "judge_cost_usd": round(sum(r["judge_usage"]["cost_usd"] for r in rows), 4),
    }


def rejudge(method: str) -> tuple[list[dict], str]:
    """保存済みの結果の回答を採点し直す。検索・回答生成はやり直さない。"""
    data = json.loads((RESULTS_DIR / f"{method}.json").read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in evalset.load()}
    for r in data["rows"]:
        before = r["verdict"]
        r.update(grade(questions[r["id"]], r["result"]["answer"]))
        print(f"{r['id']} {before:9} -> {r['verdict']:9}{' review' if r['needs_review'] else ''}", flush=True)
    return data["rows"], data["summary"]["started_at"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=list(METHODS))
    parser.add_argument("--ids", help="カンマ区切りの質問 id(省略時は全問)")
    parser.add_argument("--rejudge", action="store_true", help="保存済みの回答を採点し直す")
    args = parser.parse_args()
    ids = args.ids.split(",") if args.ids else None

    if args.rejudge:
        rows, started_at = rejudge(args.method)
    else:
        started = time.time()
        rows = run(args.method, ids)
        started_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started))
    summary = summarize(rows)
    summary["started_at"] = started_at
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.method if not ids else f"{args.method}_partial"
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps({"method": args.method, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / f"{args.method}.progress.json").unlink(missing_ok=True)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
