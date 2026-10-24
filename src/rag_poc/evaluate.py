"""評価セットの全問を 1 つの方式で実行し、採点して data/eval/results/<method>.json に保存する。

方式ごとの速度を比べられるように、質問は 1 問ずつ順に実行する。
採点は 2 つ:
- 回答の正しさ: 正解と回答を JUDGE_MODEL に比べさせ、correct / partial / incorrect を付ける。
  答えのない質問は、資料に記載がないと答えていれば correct とする
- 根拠ページの一致: 正解の根拠ページのうち、検索で取得したページ・回答が挙げたページに入っていた割合

    uv run python -m rag_poc.evaluate pageindex [--ids q01,q02]
"""

import argparse
import json
import time
from typing import Literal

from pydantic import BaseModel

from rag_poc import evalset
from rag_poc.docs import DATA_DIR
from rag_poc.llm import JUDGE_MODEL, Usage, anthropic_client

RESULTS_DIR = DATA_DIR / "eval" / "results"
SCORES = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}

JUDGE_SYSTEM = """あなたは RAG チャットボットの回答を採点する評価者です。質問・正解・回答を比べて判定してください。

- correct: 正解の要点をすべて含み、誤りがない。表現の違いや、正しい補足情報は減点しない
- partial: 要点の一部だけ合っている、または要点は合っているが誤った内容も含む
- incorrect: 要点が合っていない、または資料に答えがあるのに「記載がない」と答えた

正解が「資料に記載がない」質問では、回答が記載がないと答えていれば correct、資料にない内容で答えていれば incorrect です。"""


class Judgement(BaseModel):
    reason: str
    verdict: Literal["correct", "partial", "incorrect"]


def judge(q: dict, answer: str, usage: Usage) -> Judgement:
    expected = "資料に記載がない(答えられない質問)" if q["type"] == "unanswerable" else q["answer"]
    response = anthropic_client().messages.parse(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": f"<question>{q['question']}</question>\n<expected>{expected}</expected>\n<answer>{answer}</answer>"}],
        output_format=Judgement,
    )
    usage.add_message(JUDGE_MODEL, response.usage)
    return response.parsed_output or Judgement(reason=f"採点に失敗した({response.stop_reason})", verdict="incorrect")


def page_recall(gold: list[dict], pages: list[dict]) -> float | None:
    """正解の根拠ページのうち pages に入っている割合。答えのない質問は None。"""
    gold_set = {(e["doc"], e["page"]) for e in gold}
    if not gold_set:
        return None
    return len(gold_set & {(p["doc"], p["page"]) for p in pages}) / len(gold_set)


def run(method: str, ids: list[str] | None = None) -> list[dict]:
    if method == "vector":
        from rag_poc.vector_rag import ask
    else:
        from rag_poc.pageindex_rag import ask

    questions = [q for q in evalset.load() if not ids or q["id"] in ids]
    rows = []
    for q in questions:
        result = ask(q["question"]).to_dict()
        judge_usage = Usage()
        j = judge(q, result["answer"], judge_usage)
        rows.append(
            {
                "id": q["id"],
                "doc": q["doc"],
                "type": q["type"],
                "verdict": j.verdict,
                "score": SCORES[j.verdict],
                "judge_reason": j.reason,
                "retrieved_recall": page_recall(q["evidence"], result["retrieved"]),
                "cited_recall": page_recall(q["evidence"], result["sources"]),
                "judge_usage": judge_usage.to_dict(),
                "result": result,
            }
        )
        print(f"{q['id']} {j.verdict:9} retrieved={rows[-1]['retrieved_recall']} cited={rows[-1]['cited_recall']} {result['total_ms']}ms ${result['usage']['cost_usd']:.4f}", flush=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=["vector", "pageindex"])
    parser.add_argument("--ids", help="カンマ区切りの質問 id(省略時は全問)")
    args = parser.parse_args()
    ids = args.ids.split(",") if args.ids else None

    started = time.time()
    rows = run(args.method, ids)
    summary = summarize(rows)
    summary["started_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    name = args.method if not ids else f"{args.method}_partial"
    (RESULTS_DIR / f"{name}.json").write_text(json.dumps({"method": args.method, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
