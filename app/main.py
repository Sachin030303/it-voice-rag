
import json
import logging
import time
from collections import defaultdict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import config, ingest, llm
from .store import Store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="IT Troubleshooting Voice Assistant",
    version="1.0",
)

store: Store | None = None
sessions: dict[str, list] = defaultdict(list)


@app.on_event("startup")
def startup():
    global store
    store = Store()
    logger.info("Store initialized successfully.")


def get_store() -> Store:
    if store is None:
        raise HTTPException(
            status_code=503,
            detail="The knowledge store is not initialized.",
        )
    return store


class Filters(BaseModel):
    category: str | None = None
    source: str | None = None
    topic: str | None = None
    district: str | None = None


class RetrieveReq(BaseModel):
    query: str
    top_k: int = Field(default=config.TOP_K, ge=1, le=20)
    threshold: float = Field(
        default=config.SIM_THRESHOLD, ge=0, le=1
    )
    hybrid: bool = True
    filters: Filters = Field(default_factory=Filters)


class QueryReq(RetrieveReq):
    session_id: str = "default"
    rewrite: bool = True


def public_chunks(chunks):
    return [
        {
            "n": i,
            "source": chunk["metadata"].get("source", "Unknown"),
            "section": chunk["metadata"].get("section"),
            "category": chunk["metadata"].get("category"),
            "similarity": chunk["similarity"],
            "snippet": chunk["text"][:240],
        }
        for i, chunk in enumerate(chunks, 1)
    ]


@app.get("/health")
def health():
    current_store = get_store()
    return {
        "status": "ok",
        **current_store.stats(),
        "embed_model": config.EMBED_MODEL,
        "llm": config.LLM_MODEL,
    }


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    category: str | None = Form(default=None),
    topic: str | None = Form(default=None),
    district: str | None = Form(default=None),
):
    current_store = get_store()
    data = await file.read()

    try:
        text = ingest.extract_text(file.filename or "upload", data)
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc

    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text. Scanned PDFs may need OCR.",
        )

    started = time.perf_counter()
    chunks = ingest.chunk_text(
        text,
        config.CHUNK_CHARS,
        config.CHUNK_OVERLAP_PARAS,
    )

    metadata = {
        "category": category or ingest.infer_category(
            text, file.filename or "upload"
        ),
        "topic": topic or (
            (file.filename or "upload").rsplit(".", 1)[0]
            .replace("_", " ")
        ),
        "district": district or "global",
    }

    try:
        count = current_store.add(
            file.filename or "upload",
            chunks,
            metadata,
        )
    except Exception as exc:
        logger.exception("Document ingestion failed")
        raise HTTPException(
            status_code=500,
            detail="Document embedding or storage failed. Check server logs.",
        ) from exc

    return {
        "source": file.filename,
        "chunks": count,
        "metadata": metadata,
        "ingest_ms": round(
            (time.perf_counter() - started) * 1000
        ),
    }


@app.post("/retrieve")
def retrieve(req: RetrieveReq):
    current_store = get_store()
    started = time.perf_counter()

    try:
        results = current_store.search(
            req.query,
            req.top_k,
            req.threshold,
            req.filters.model_dump(exclude_none=True),
            req.hybrid,
        )
    except Exception as exc:
        logger.exception("Retrieval failed")
        raise HTTPException(
            status_code=500,
            detail="Retrieval failed. Check server logs.",
        ) from exc

    return {
        "query": req.query,
        "results": results,
        "retrieval_ms": round(
            (time.perf_counter() - started) * 1000
        ),
    }


def sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/query")
async def query(req: QueryReq):
    current_store = get_store()
    history = sessions[req.session_id]

    async def generate():
        started = time.perf_counter()
        first_token_time = None

        try:
            rewritten_query = (
                await llm.rewrite_query(req.query, history)
                if req.rewrite and history
                else req.query
            )
            rewrite_finished = time.perf_counter()

            chunks = current_store.search(
                rewritten_query,
                req.top_k,
                req.threshold,
                req.filters.model_dump(exclude_none=True),
                req.hybrid,
            )
            retrieval_finished = time.perf_counter()

            yield sse(
                "sources",
                {
                    "rewritten_query": rewritten_query,
                    "sources": public_chunks(chunks),
                },
            )

            answer_parts = []

            async for token in llm.stream_answer(
                req.query, chunks, history
            ):
                if first_token_time is None:
                    first_token_time = time.perf_counter()

                answer_parts.append(token)
                yield sse("token", token)

            answer = "".join(answer_parts)

            history.extend([
                ("user", req.query),
                ("assistant", answer),
            ])

            max_messages = 2 * config.MAX_HISTORY_TURNS
            if len(history) > max_messages:
                del history[:-max_messages]

            finished = time.perf_counter()

            yield sse(
                "done",
                {
                    "timings_ms": {
                        "rewrite": round(
                            (rewrite_finished - started) * 1000
                        ),
                        "retrieval": round(
                            (retrieval_finished - rewrite_finished) * 1000
                        ),
                        "time_to_first_token": round(
                            (
                                (first_token_time or finished) - started
                            ) * 1000
                        ),
                        "total": round(
                            (finished - started) * 1000
                        ),
                    }
                },
            )

        except Exception:
            logger.exception("Question processing failed")
            # The response has already started, so report the error in SSE.
            yield sse(
                "error",
                {
                    "message": (
                        "The assistant could not complete this question. "
                        "Check the backend logs for details."
                    )
                },
            )

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.delete("/session/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"cleared": session_id}


app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)


@app.get("/")
def index():
    return FileResponse("static/index.html")
