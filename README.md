# poc-of-chatbots-with-rag

RAG（検索拡張生成）を前提としたチャットボットの PoC。同じ資料・同じ評価セットで、次の2方式を比較する。

1. **ベクトル埋め込み方式**：資料をチャンクに分割・埋め込みし、ベクトル類似度検索で関連箇所を取得する
2. **PageIndex 方式**：資料の階層構造（目次ツリー）をもとに、LLM の推論で関連箇所を辿って取得する（[PageIndex](https://github.com/VectifyAI/PageIndex) OSS 版）

## 検証の前提

| 項目 | 内容 |
|---|---|
| 対象資料 | IPA「[中小企業の情報セキュリティ対策ガイドライン第4.0版](https://www.ipa.go.jp/security/guide/sme/about.html)」の本編・付録5（情報セキュリティ関連規程サンプル）・付録3（自社診断） |
| 回答生成 LLM | Claude Sonnet 5.5（両方式共通）。PageIndex の索引作成は Claude Haiku 5.5 |
| 埋め込み | OpenAI text-embedding-3-large |
| ベクトル DB | PostgreSQL + pgvector |
| 評価 | 50問（正解ページ番号付き）、LLM による採点。精度・根拠ページの一致率・速度・コスト・構築の手間を計測して比較する |
| 予算 | API 利用料は合計 $100 まで |

## 構成（予定）

- RAG 処理・計測：Python（CLI と HTTP API）
- UI：TypeScript の Web UI
- マルチターン会話・認証は対象外（1問1答のみ）

## 開発環境

dev container（`~/dev-containers/poc-of-chatbots-with-rag/`）で開発する。pgvector 入りの PostgreSQL は `db` サービスとして同時に起動し、接続先は環境変数 `DATABASE_URL` に入っている。

| 用途 | ホスト側の URL |
|---|---|
| 質問応答 API（FastAPI） | http://localhost:8100 （`--host 0.0.0.0 --port 8100` で起動） |
| Web UI（vite dev） | http://localhost:5180 （`--host 0.0.0.0 --port 5180` で起動） |

API キーは `.env.example` をコピーした `.env` に書く（`.env` はコミットしない）。

Python のパッケージは [uv](https://docs.astral.sh/uv/) で管理する。依存は `pyproject.toml`、固定したバージョンは `uv.lock` にある。仮想環境は `UV_PROJECT_ENVIRONMENT` が指す `/home/dev-container/.venv` を使う。

```sh
uv sync                  # uv.lock どおりに仮想環境へ入れる
uv add <package>         # 依存を追加する（uv.lock も更新される）
uv run python -m rag_poc.docs
```

## 資料の扱い

このリポジトリは公開リポジトリのため、検証に使う資料（PDF・Word）はコミットしない。`data/` に置き、`.gitignore` で除外している。

付録5 は Word で配布されているので、LibreOffice で PDF に変換して使う。

```sh
soffice --headless --convert-to pdf --outdir data data/sme_guideline_v4.0_app_5.docx
uv sync
uv run python -m rag_poc.docs   # ページごとのテキストを data/text/<doc>/p001.txt に書き出す
```

| doc | ファイル |
|---|---|
| `main` | `sme_guideline_v4.0.pdf`（本編） |
| `app5` | `sme_guideline_v4.0_app_5.pdf`（付録5、docx から変換） |
| `app3` | `sme_guideline_v4.0_app_3.pdf`（付録3） |

ページ番号は PDF の物理ページ（1 始まり）で数える。資料に印刷されたページ番号とは一致しない。

## 索引の作成

どちらも `data/index/` に結果を書き出す（コミットしない）。本編は左右の余白に縦書きの章見出しがあるので、テキストから除いてから使う（`rag_poc.docs.clean_pages`）。

```sh
uv run python -m rag_poc.pageindex_build   # PageIndex の目次ツリー（Claude Haiku 5.5）→ data/index/pageindex/<doc>.json
uv run python -m rag_poc.vector_build      # チャンク分割・埋め込み（text-embedding-3-large）→ PostgreSQL の chunks テーブル
```

## 質問応答

```sh
uv run python -m rag_poc.ask --method pageindex "質問"            # 1 問を CLI で答える(--method vector も可)
uv run uvicorn rag_poc.api:app --host 0.0.0.0 --port 8100        # 質問応答 API
cd web && npm install && npm run dev                             # Web UI(http://localhost:5180、/api は 8100 へ転送)
```

API は `POST /api/ask`（`{"question": "…", "method": "vector" | "pageindex"}`）で、回答・根拠ページ・検索で取得したページ・処理時間・トークン数・料金を返す。Web UI は同じ質問を両方式に並列で投げ、結果を左右に並べて表示する。

## 評価セット

`data/eval/questions.json`（コミットしない）。50問で、3資料を横断して検索する前提。

```json
{
  "id": "q01",
  "doc": "main",
  "type": "fact",
  "question": "…",
  "answer": "…",
  "evidence": [{"doc": "main", "page": 70, "quote": "…"}]
}
```

- `doc`: 問題を作った資料。配分は `main` 30・`app5` 10・`app3` 10
- `type`: `fact`（事実の参照）/ `multi`（複数の節・資料にまたがる）/ `table`（数値・表の読み取り）/ `unanswerable`（資料に答えがない）
- `evidence`: 正解の根拠のページと、そのページにある文言。`unanswerable` は空

`uv run python -m rag_poc.evalset` で、各 `quote` が指定したページの本文にあるかを確かめる。

## 計測

```sh
uv run python -m rag_poc.evaluate pageindex   # 全問を実行・採点し data/eval/results/pageindex.json に保存(vector も同様)
uv run python -m rag_poc.compare              # 両方式の集計値を Markdown の表で出す
```

質問は 1 問ずつ順に実行する。回答の正しさは Claude Opus 5.5 が正解と比べて correct / partial / incorrect で採点し、答えのない質問は「記載がない」と答えれば correct とする。あわせて、正解の根拠ページを検索で取得できたか・回答の根拠に挙げたか、処理時間、トークン数、料金を記録する。

比較の結果は [docs/comparison.md](docs/comparison.md)、採用方針と本実装に向けた課題は [docs/report.md](docs/report.md) にまとめている。
