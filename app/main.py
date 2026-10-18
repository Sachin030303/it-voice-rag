import json
import time
from collections import defaultdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, ingest, llm
from .store import Store

app = FastAPI(title="IT Troubleshooting Voice Assistant", version="1.0")
store: Store | None = None
sessions: dict[str, list] = defaultdict(list)  # session_id -> [(role, text)]


@app.on_event("startup")
def _startup():
    global store
    store = Store()
    store.embed_query("warmup")  # load model weights at boot, not on the first user query


class Filters(BaseModel):
    category: str | None = None
    source: str | None = None
    topic: str | None = None
    district: str | None = None


class RetrieveReq(BaseModel):
    query: str
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)
    threshold: float = Field(default=config.SIM_THRESHOLD, ge=0, le=1)
    hybrid: bool = True
    filters: Filters = Filters()


class QueryReq(RetrieveReq):
    session_id: str = "default"
    rewrite: bool = True


def _public(chunks):
    return [{"n": i, "source": c["metadata"]["source"], "section": c["metadata"].get("section"),
             "category": c["metadata"].get("category"), "similarity": c["similarity"],
             "snippet": c["text"][:240]} for i, c in enumerate(chunks, 1)]


@app.get("/health")
def health():
    return {"status": "ok", **store.stats(), "embed_model": config.EMBED_MODEL, "llm": config.LLM_MODEL}


@app.post("/upload")
async def upload(file: UploadFile = File(...), category: str = Form(None), topic: str = Form(None),
                 district: str = Form(None)):
    data = await file.read()
    try:
        text = ingest.extract_text(file.filename, data)
    except ValueError as e:
        raise HTTPException(415, str(e))
    if not text.strip():
        raise HTTPException(422, "No extractable text (scanned PDF? run OCR first).")
    t0 = time.perf_counter()
    chunks = ingest.chunk_text(text, config.CHUNK_CHARS, config.CHUNK_OVERLAP_PARAS)
    meta = {"category": category or ingest.infer_category(text, file.filename),
            "topic": topic or file.filename.rsplit(".", 1)[0].replace("_", " "),
            "district": district or "global"}
    n = store.add(file.filename, chunks, meta)
    return {"source": file.filename, "chunks": n, "metadata": meta, "ingest_ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/retrieve")
def retrieve(req: RetrieveReq):
    t0 = time.perf_counter()
    res = store.search(req.query, req.top_k, req.threshold, req.filters.model_dump(), req.hybrid)
    return {"query": req.query, "results": res, "retrieval_ms": round((time.perf_counter() - t0) * 1000)}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query")
async def query(req: QueryReq):
    history = sessions[req.session_id]

    async def gen():
        t0 = time.perf_counter()
        q = await llm.rewrite_query(req.query, history) if req.rewrite else req.query
        t1 = time.perf_counter()
        chunks = store.search(q, req.top_k, req.threshold, req.filters.model_dump(), req.hybrid)
        t2 = time.perf_counter()
        yield _sse("sources", {"rewritten_query": q, "sources": _public(chunks)})
        answer, first = [], None
        async for tok in llm.stream_answer(req.query, chunks, history):
            if first is None:
                first = time.perf_counter()
            answer.append(tok)
            yield _sse("token", tok)
        history.extend([("user", req.query), ("assistant", "".join(answer))])
        del history[:-2 * config.MAX_HISTORY_TURNS]
        yield _sse("done", {"timings_ms": {"rewrite": round((t1 - t0) * 1000), "retrieval": round((t2 - t1) * 1000),
                                           "time_to_first_token": round(((first or t2) - t0) * 1000),
                                           "total": round((time.perf_counter() - t0) * 1000)}})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.delete("/session/{session_id}")
def clear(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
