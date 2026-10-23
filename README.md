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

## 資料の扱い

このリポジトリは公開リポジトリのため、検証に使う資料（PDF・Word）はコミットしない。`data/` に置き、`.gitignore` で除外している。
