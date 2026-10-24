"""質問応答 API。Web UI から両方式を試すためのもの。1 問 1 答のみで、認証はない。

    uv run uvicorn rag_poc.api:app --host 0.0.0.0 --port 8100
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

from rag_poc.docs import DOCS
from rag_poc.methods import pipeline

app = FastAPI(title="rag-poc")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    method: Literal["vector", "vector_page", "vector_page_doc", "vector_jev", "pageindex"]


@app.get("/api/docs")
def docs() -> list[dict]:
    return [{"id": d.id, "title": d.title, "label": d.label} for d in DOCS]


@app.post("/api/ask")
async def ask(req: AskRequest) -> dict:
    try:
        result = await run_in_threadpool(pipeline(req.method), req.question.strip())
    except Exception as e:  # 外部 API の失敗は、方式ごとに画面へ出せるようにする
        raise HTTPException(status_code=502, detail=f"{type(e).__name__}: {e}") from e
    return result.to_dict()
