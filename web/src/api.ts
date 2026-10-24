export type Method = "vector" | "vector_page" | "vector_page_doc" | "vector_jev" | "vector_haiku" | "pageindex";

export interface DocInfo {
  id: string;
  title: string;
  label: string;
}

export interface PageRef {
  doc: string;
  page: number;
}

export interface PageIndexNode {
  id: string;
  doc: string;
  title: string;
  start: number;
  end: number;
}

export interface AskResponse {
  method: Method;
  question: string;
  answer: string;
  answerable: boolean;
  sources: PageRef[];
  retrieved: PageRef[];
  retrieval_ms: number;
  generation_ms: number;
  total_ms: number;
  usage: {
    tokens: Record<string, { input: number; output: number; cache_read: number }>;
    cost_usd: number;
  };
  detail: { reasoning?: string; nodes?: PageIndexNode[]; rerank_scores?: number[] };
}

// FastAPI のエラー(502 は文字列、422 は配列)をメッセージにする
async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    const d = body?.detail;
    if (typeof d === "string") return `${res.status}: ${d}`;
    if (Array.isArray(d)) return `${res.status}: ${d.map((e) => e?.msg ?? JSON.stringify(e)).join(" / ")}`;
    return `${res.status}: ${JSON.stringify(body)}`;
  } catch {
    return `${res.status} ${res.statusText}`;
  }
}

export async function fetchDocs(): Promise<DocInfo[]> {
  const res = await fetch("/api/docs");
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}

export async function ask(question: string, method: Method, signal?: AbortSignal): Promise<AskResponse> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, method }),
    signal,
  });
  if (!res.ok) throw new Error(await errorMessage(res));
  return res.json();
}
