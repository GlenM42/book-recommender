"""
FastAPI recommendation server.

Endpoints
---------
GET  /health                   — liveness check + model metadata
GET  /search?q=...             — search books by title, returns work_ids
POST /get-als-recommendation   — item-to-item similar books
"""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from implicit.cpu.als import AlternatingLeastSquares
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")

# ── Shared state loaded once at startup ──────────────────────────────────────

_state: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


def _load_artifacts() -> None:
    model_path = MODELS_DIR / "als_model.npz"
    lookup_path = MODELS_DIR / "item_lookup.parquet"
    info_path = MODELS_DIR / "training_info.json"

    from als.storage import s3_configured, download_artifacts
    if s3_configured() and not model_path.exists():
        log.info("Model not found locally — downloading from S3 …")
        download_artifacts(MODELS_DIR)

    for p in (model_path, lookup_path, info_path):
        if not p.exists():
            raise RuntimeError(
                f"Model artifact not found: {p}\n"
                "Run `python console.py train` first."
            )

    log.info("Loading ALS model from %s …", model_path)
    _state["model"] = AlternatingLeastSquares.load(str(model_path))

    log.info("Loading item lookup from %s …", lookup_path)
    item_lookup = pd.read_parquet(lookup_path)

    # work_id → item_idx  (for request lookup)
    # Multiple editions share the same work_id — pick the one with the most ratings.
    best_edition = (
        item_lookup.sort_values("ratings_count", ascending=False)
        .drop_duplicates(subset="work_id", keep="first")
    )
    _state["work_id_to_idx"] = dict(
        zip(best_edition["work_id"].astype(str), best_edition["item_idx"].astype(int))
    )

    # flat DataFrame for title search (sorted by popularity)
    _state["search_df"] = (
        item_lookup[["work_id", "title", "url", "ratings_count"]]
        .sort_values("ratings_count", ascending=False)
        .reset_index(drop=True)
    )

    # item_idx → {work_id, title, url, ratings_count, author_ids}  (for response)
    _state["idx_to_meta"] = (
        item_lookup
        .set_index("item_idx")[["work_id", "title", "url", "ratings_count", "author_ids"]]
        .to_dict("index")
    )

    with open(info_path) as f:
        _state["training_info"] = json.load(f)

    log.info(
        "Model ready — %d items, %d factors, trained at %s",
        _state["training_info"]["n_items"],
        _state["training_info"]["factors"],
        _state["training_info"]["trained_at"],
    )


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Book Recommender — ALS", lifespan=lifespan)


# ── Schemas ───────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    work_id: str
    title: str | None
    url: str | None
    ratings_count: int | None


class RecommendRequest(BaseModel):
    work_id: str = Field(..., description="Goodreads work_id of the seed book")
    n: int = Field(10, ge=1, le=100, description="Number of recommendations to return")


class BookResult(BaseModel):
    work_id: str
    title: str | None
    url: str | None
    score: float


class RecommendResponse(BaseModel):
    work_id: str
    title: str | None
    similar_books: list[BookResult]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    info = _state.get("training_info", {})
    return {
        "status": "ok",
        "model_loaded": "model" in _state,
        "n_items": info.get("n_items"),
        "n_users": info.get("n_users"),
        "factors": info.get("factors"),
        "trained_at": info.get("trained_at"),
    }


@app.get("/search", response_model=list[SearchResult])
def search(
    q: str = Query(..., min_length=2, description="Title substring to search for"),
    n: int = Query(20, ge=1, le=100, description="Max results to return"),
):
    df = _state["search_df"]
    mask = df["title"].str.contains(q, case=False, na=False)
    rows = df[mask].head(n)
    return [
        SearchResult(
            work_id=str(row["work_id"]),
            title=row["title"] or None,
            url=row["url"] or None,
            ratings_count=int(row["ratings_count"]) if pd.notna(row["ratings_count"]) else None,
        )
        for _, row in rows.iterrows()
    ]


@app.post("/get-als-recommendation", response_model=RecommendResponse)
def get_recommendation(req: RecommendRequest):
    work_id = str(req.work_id)
    item_idx = _state["work_id_to_idx"].get(work_id)

    if item_idx is None:
        raise HTTPException(
            status_code=404,
            detail=f"work_id '{work_id}' not found in the trained model. "
                   "Make sure it appears in the interaction data.",
        )

    model: AlternatingLeastSquares = _state["model"]
    similar_ids, scores = model.similar_items(item_idx, N=req.n + 1)

    idx_to_meta = _state["idx_to_meta"]
    seed_meta = idx_to_meta.get(item_idx, {})

    similar_books: list[BookResult] = []
    for sid, score in zip(similar_ids, scores):
        sid = int(sid)
        if sid == item_idx:
            continue  # skip the seed itself
        meta = idx_to_meta.get(sid, {})
        similar_books.append(BookResult(
            work_id=str(meta.get("work_id", sid)),
            title=meta.get("title") or None,
            url=meta.get("url") or None,
            score=float(score),
        ))
        if len(similar_books) >= req.n:
            break

    return RecommendResponse(
        work_id=work_id,
        title=seed_meta.get("title") or None,
        similar_books=similar_books,
    )


# ── Server entry ──────────────────────────────────────────────────────────────

def run_server(args) -> None:
    uvicorn.run(
        "als.serve:app",
        host=args.host,
        port=args.port,
        reload=getattr(args, "reload", False),
        log_level="info",
    )
