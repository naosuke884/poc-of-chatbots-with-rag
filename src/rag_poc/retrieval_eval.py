"""検索だけを評価セットに流し、正解ページを取得できたか・検索の時間・料金を測る。回答生成と採点はしない。

資料の範囲は環境変数 RAG_CORPUS で切り替え、結果は data/eval/retrieval_<3|11>/<method>.json に保存する。
11 冊では evalgold で補った正解ページ(追加の 8 冊にある同じ答えのページ)も正解として数える。
答えのない質問は、追加の 8 冊に答えがあれば 11 冊の計測から除く。
--extra を付けると、追加の 8 冊から作った問題(evalgen_extra)を流し、<method>_extra.json に保存する(11 冊のみ)。

    RAG_CORPUS=11 uv run python -m rag_poc.retrieval_eval vector_jev [--extra]
    uv run python -m rag_poc.retrieval_eval --table   # 3 冊と 11 冊の結果を Markdown の表で出す
"""

import argparse
import json
import statistics
import time

from rag_poc import evalgen_extra, evalgold, evalset
from rag_poc.answer import unique_pages
from rag_poc.docs import CORPUS, DATA_DIR
from rag_poc.evaluate import page_recall
from rag_poc.llm import Usage
from rag_poc.methods import METHODS, retriever


def results_dir(corpus: str):
    return DATA_DIR / "eval" / f"retrieval_{corpus}"


def questions(extra: bool = False) -> tuple[list[dict], dict]:
    """計測する問題と、補った正解ページ(3 冊では空)を返す。"""
    alt = evalgold.load() if CORPUS == "11" else {}
    if extra:
        return evalgen_extra.load(), alt
    qs = [q for q in evalset.load() if not (q["type"] == "unanswerable" and alt.get(q["id"], {}).get("pages"))]
    return qs, alt


def run(method: str, extra: bool = False) -> dict:
    if extra and CORPUS != "11":
        raise ValueError("--extra は RAG_CORPUS=11 で使う")
    retrieve = retriever(method)
    qs, alt = questions(extra)
    rows = []
    for q in qs:
        usage = Usage()
        started = time.perf_counter()
        pages = unique_pages(retrieve(q["question"], usage))
        ms = round((time.perf_counter() - started) * 1000)
        recall = page_recall(q["evidence"], pages)
        alt_pages = {(p["doc"], p["page"]) for p in alt.get(q["id"], {}).get("pages", [])}
        # 補ったページは、それだけで正解の要点がわかるページなので、1 つ取れていれば取れたとみなす
        recall_alt = None if recall is None else (1.0 if alt_pages & {(p["doc"], p["page"]) for p in pages} else recall)
        rows.append({"id": q["id"], "doc": q["doc"], "type": q["type"], "recall": recall, "recall_alt": recall_alt, "pages": pages, "ms": ms, "usage": usage.to_dict()})
        print(f"{q['id']} recall={recall} alt={recall_alt} pages={len(pages)} {ms}ms", flush=True)

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return round(sum(xs) / len(xs), 3) if xs else None

    ms = sorted(r["ms"] for r in rows)
    summary = {
        "corpus": CORPUS,
        "n": len(rows),
        "recall": mean(r["recall"] for r in rows),
        "recall_alt": mean(r["recall_alt"] for r in rows),
        "recall_by_doc": {d: mean(r["recall_alt"] for r in rows if r["doc"] == d) for d in sorted({r["doc"] for r in rows})},
        "pages_mean": round(statistics.mean(len(r["pages"]) for r in rows), 1),
        "ms": {"mean": round(statistics.mean(ms)), "p50": ms[len(ms) // 2], "max": ms[-1]},
        "cost_per_question_usd": round(sum(r["usage"]["cost_usd"] for r in rows) / len(rows), 5),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    out = results_dir(CORPUS)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{method}{'_extra' if extra else ''}.json").write_text(json.dumps({"method": method, "summary": summary, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def table() -> str:
    def load(corpus, method, suffix=""):
        path = results_dir(corpus) / f"{method}{suffix}.json"
        return json.loads(path.read_text(encoding="utf-8"))["summary"] if path.exists() else None

    lines = [
        "| 方式 | 正解ページの取得 3 冊 → 11 冊 | 付録3 の問題 3 冊 → 11 冊 | 追加資料の問題(11 冊) | 渡すページ数 3 冊 → 11 冊 | 検索の時間 3 冊 → 11 冊 | 検索の料金 3 冊 → 11 冊 |",
        "|---|---|---|---|---|---|---|",
    ]
    for m, label in METHODS.items():
        s3, s11, ex = load("3", m), load("11", m), load("11", m, "_extra")
        if not (s3 and s11):
            continue
        extra_recall = f"{ex['recall_alt'] * 100:.0f}%" if ex else "-"
        lines.append(
            f"| {label} | {s3['recall_alt'] * 100:.0f}% → {s11['recall_alt'] * 100:.0f}% "
            f"| {s3['recall_by_doc']['app3'] * 100:.0f}% → {s11['recall_by_doc']['app3'] * 100:.0f}% "
            f"| {extra_recall} "
            f"| {s3['pages_mean']} → {s11['pages_mean']} "
            f"| {s3['ms']['mean'] / 1000:.1f} → {s11['ms']['mean'] / 1000:.1f} 秒 "
            f"| ${s3['cost_per_question_usd']:.5f} → ${s11['cost_per_question_usd']:.5f} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", nargs="?", choices=list(METHODS))
    parser.add_argument("--extra", action="store_true", help="追加の 8 冊から作った問題を流す")
    parser.add_argument("--table", action="store_true", help="3 冊と 11 冊の結果を表にする")
    args = parser.parse_args()
    if args.table:
        print(table())
    else:
        print(json.dumps(run(args.method, args.extra), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
