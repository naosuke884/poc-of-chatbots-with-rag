"""data/eval/results/ の両方式の計測結果を並べ、Markdown の比較表を出力する。

公開リポジトリに置けるよう、質問文や回答は出さず集計値だけを出す。

    uv run python -m rag_poc.compare
"""

import json

from rag_poc.evaluate import RESULTS_DIR
from rag_poc.methods import METHODS

TYPE_LABELS = {"fact": "事実の参照", "multi": "複数箇所", "table": "数値・表", "unanswerable": "答えなし"}
DOC_LABELS = {"main": "本編", "app5": "付録5", "app3": "付録3"}


def load(method: str) -> dict | None:
    path = RESULTS_DIR / f"{method}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _pct(x):
    return "-" if x is None else f"{x * 100:.0f}%"


def table() -> str:
    data = {m: load(m) for m in METHODS}
    cols = [(m, label) for m, label in METHODS.items() if data[m]]
    s = {m: data[m]["summary"] for m, _ in cols}

    def row(name, f):
        return f"| {name} | " + " | ".join(f(s[m]) for m, _ in cols) + " |"

    lines = ["| 指標 | " + " | ".join(label for _, label in cols) + " |", "|---|" + "---|" * len(cols)]
    lines.append(row("正答率(部分正解は 0.5)", lambda x: _pct(x["score"])))
    lines.append(row("正解 / 部分正解 / 不正解", lambda x: "{correct} / {partial} / {incorrect}".format(**x["verdicts"])))
    for t, label in TYPE_LABELS.items():
        lines.append(row(f"　{label}", lambda x, t=t: f"{_pct(x['by_type'][t]['score'])}(n={x['by_type'][t]['n']})" if t in x["by_type"] else "-"))
    for d, label in DOC_LABELS.items():
        lines.append(row(f"　{label}の問題", lambda x, d=d: _pct(x["by_doc"][d]["score"]) if d in x["by_doc"] else "-"))
    lines.append(row("正解ページを検索で取得できた割合", lambda x: _pct(x["retrieved_recall"])))
    lines.append(row("正解ページを根拠に挙げた割合", lambda x: _pct(x["cited_recall"])))
    lines.append(row("応答時間 平均 / 中央値 / 最大", lambda x: "{:.1f} / {:.1f} / {:.1f} 秒".format(x["latency_ms"]["mean"] / 1000, x["latency_ms"]["p50"] / 1000, x["latency_ms"]["max"] / 1000)))
    lines.append(row("　うち検索 / 生成(平均)", lambda x: "{:.1f} / {:.1f} 秒".format(x["latency_ms"]["retrieval_mean"] / 1000, x["latency_ms"]["generation_mean"] / 1000)))
    lines.append(row("1 問あたりの料金", lambda x: f"${x['cost_per_question_usd']:.4f}"))

    def tokens(x):
        t = {"input": 0, "output": 0, "cache_read": 0}
        for v in x["usage"]["tokens"].values():
            for k in t:
                t[k] += v[k]
        n = x["n"]
        return f"{t['input'] // n:,} / {t['output'] // n:,} / {t['cache_read'] // n:,}"

    lines.append(row("1 問あたりのトークン 入力 / 出力 / キャッシュ", tokens))
    return "\n".join(lines)


def misses(method: str) -> list[dict]:
    """部分正解・不正解の問題(id・種類・判定・正解ページを取得できたか)。分析用で、出力先は data/ のみ。"""
    data = load(method)
    return [
        {k: r[k] for k in ("id", "doc", "type", "verdict", "retrieved_recall", "cited_recall", "judge_reason")}
        for r in (data["rows"] if data else [])
        if r["verdict"] != "correct"
    ]


if __name__ == "__main__":
    print(table())
