import type { AskResponse, PageRef } from "./api";
import type { ColumnState } from "./App";

interface Props {
  label: string;
  state: ColumnState;
  docTitles: Record<string, string>;
}

const sec = (ms: number) => `${(ms / 1000).toFixed(1)} 秒`;

function docName(docTitles: Record<string, string>, id: string) {
  return docTitles[id] ?? id;
}

function PageList({ pages, docTitles }: { pages: PageRef[]; docTitles: Record<string, string> }) {
  if (pages.length === 0) return <p className="muted">なし</p>;
  return (
    <ul className="pages">
      {pages.map((p, i) => (
        <li key={`${p.doc}-${p.page}-${i}`}>
          {docName(docTitles, p.doc)} p.{p.page}
        </li>
      ))}
    </ul>
  );
}

function Result({ result, docTitles }: { result: AskResponse; docTitles: Record<string, string> }) {
  const totals = Object.values(result.usage.tokens).reduce(
    (acc, t) => ({
      input: acc.input + t.input,
      output: acc.output + t.output,
      cache_read: acc.cache_read + t.cache_read,
    }),
    { input: 0, output: 0, cache_read: 0 },
  );
  const nodes = result.detail.nodes ?? [];
  const reasoning = result.detail.reasoning;

  return (
    <>
      {!result.answerable && <span className="badge">資料に記載なし</span>}
      <div className="answer">{result.answer}</div>

      <h3>根拠ページ</h3>
      <PageList pages={result.sources} docTitles={docTitles} />

      <details>
        <summary>検索で取得したページ({result.retrieved.length})</summary>
        <PageList pages={result.retrieved} docTitles={docTitles} />
      </details>

      {result.method === "pageindex" && (
        <details>
          <summary>選択されたノードと推論({nodes.length})</summary>
          {nodes.length > 0 ? (
            <ul className="pages">
              {nodes.map((n) => (
                <li key={`${n.doc}-${n.id}`}>
                  {n.title}
                  <span className="muted">
                    {" "}
                    ({docName(docTitles, n.doc)} p.{n.start}
                    {n.end !== n.start && `–${n.end}`})
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="muted">ノードなし</p>
          )}
          {reasoning && <div className="reasoning">{reasoning}</div>}
        </details>
      )}

      <dl className="metrics">
        <dt>時間</dt>
        <dd>
          検索 {sec(result.retrieval_ms)} / 生成 {sec(result.generation_ms)} / 合計 {sec(result.total_ms)}
        </dd>
        <dt>トークン</dt>
        <dd>
          入力 {totals.input.toLocaleString()} / 出力 {totals.output.toLocaleString()} / キャッシュ読込{" "}
          {totals.cache_read.toLocaleString()}
        </dd>
        <dt>料金</dt>
        <dd>${result.usage.cost_usd.toFixed(4)}</dd>
      </dl>
    </>
  );
}

export default function ResultColumn({ label, state, docTitles }: Props) {
  return (
    <section className="column" aria-busy={state.status === "loading"}>
      <h2>{label}</h2>
      {state.status === "idle" && <p className="muted">質問を送信すると、ここに回答が表示されます。</p>}
      {state.status === "loading" && <p className="loading">回答を生成しています…</p>}
      {state.status === "error" && (
        <div className="error" role="alert">
          エラーが発生しました: {state.message}
        </div>
      )}
      {state.status === "done" && <Result result={state.result} docTitles={docTitles} />}
    </section>
  );
}
