import { useEffect, useRef, useState } from "react";
import { ask, fetchDocs, type AskResponse, type Method } from "./api";
import ResultColumn from "./ResultColumn";

export type ColumnState =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; message: string }
  | { status: "done"; result: AskResponse };

const METHODS: { id: Method; label: string }[] = [
  { id: "vector", label: "ベクトル検索" },
  { id: "pageindex", label: "PageIndex" },
];

export default function App() {
  const [question, setQuestion] = useState("");
  const [enabled, setEnabled] = useState<Record<Method, boolean>>({ vector: true, pageindex: true });
  const [states, setStates] = useState<Record<Method, ColumnState>>({
    vector: { status: "idle" },
    pageindex: { status: "idle" },
  });
  const [docTitles, setDocTitles] = useState<Record<string, string>>({});
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    fetchDocs()
      .then((docs) => setDocTitles(Object.fromEntries(docs.map((d) => [d.id, d.label]))))
      .catch(() => {}); // 取得できなければ資料 ID をそのまま表示する
  }, []);

  const loading = Object.values(states).some((s) => s.status === "loading");
  const targets = METHODS.filter((m) => enabled[m.id]).map((m) => m.id);

  function submit(e: React.FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || targets.length === 0) return;

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    // 各手法を並列に呼び、結果は列ごとに独立して反映する
    for (const m of METHODS.map((x) => x.id)) {
      if (!enabled[m]) {
        setStates((s) => ({ ...s, [m]: { status: "idle" } }));
        continue;
      }
      setStates((s) => ({ ...s, [m]: { status: "loading" } }));
      ask(q, m, controller.signal)
        .then((result) => setStates((s) => ({ ...s, [m]: { status: "done", result } })))
        .catch((err: unknown) => {
          if (controller.signal.aborted) return;
          const message = err instanceof Error ? err.message : String(err);
          setStates((s) => ({ ...s, [m]: { status: "error", message } }));
        });
    }
  }

  return (
    <div className="app">
      <header>
        <h1>RAG 比較 PoC</h1>
        <p className="subtitle">同じ質問をベクトル検索と PageIndex の両方で回答し、結果を並べて比較します。</p>
      </header>

      <form className="ask-form" onSubmit={submit}>
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) submit(e);
          }}
          placeholder="資料についての質問を入力してください"
          rows={3}
          aria-label="質問"
        />
        <div className="form-row">
          <div className="method-toggles">
            {METHODS.map((m) => (
              <label key={m.id}>
                <input
                  type="checkbox"
                  checked={enabled[m.id]}
                  onChange={(e) => setEnabled((s) => ({ ...s, [m.id]: e.target.checked }))}
                />
                {m.label}
              </label>
            ))}
          </div>
          <span className="hint">Ctrl+Enter で送信</span>
          <button type="submit" disabled={!question.trim() || targets.length === 0}>
            {loading ? "回答中…" : "質問する"}
          </button>
        </div>
      </form>

      <main className="columns">
        {METHODS.filter((m) => enabled[m.id] || states[m.id].status !== "idle").map((m) => (
          <ResultColumn key={m.id} label={m.label} state={states[m.id]} docTitles={docTitles} />
        ))}
      </main>
    </div>
  );
}
