"""
ALS training pipeline.

Steps
-----
1. Stream goodreads_books.json → build raw item lookup (all 2.36 M books)
2. Load interactions in 1 M-row chunks, filter is_read==1, accumulate COO triplets
3. Cold-start filtering: drop items / users below interaction thresholds
4. Build CSR matrix, train implicit ALS
5. Save model + item lookup + training info to models/
"""

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from tqdm import tqdm

log = logging.getLogger(__name__)

MODELS_DIR = Path("models")
BOOKS_PATH = Path("goodreads_books.json")
INTERACTIONS_PATH = Path("goodreads_interactions.csv")
BOOK_MAP_PATH = Path("book_id_map.csv")

# Approximate row counts used for tqdm ETA — close enough for progress display
_APPROX_BOOKS = 2_360_655
_APPROX_INTERACTION_CHUNKS = 229  # 228 648 343 rows / 1 000 000 per chunk


def _remap_indices(keep_mask: np.ndarray) -> np.ndarray:
    """Return array where kept positions get new contiguous indices, others -1."""
    remap = np.full(len(keep_mask), -1, dtype=np.int32)
    remap[keep_mask] = np.arange(keep_mask.sum(), dtype=np.int32)
    return remap


def build_item_lookup(book_map: pd.DataFrame) -> pd.DataFrame:
    """Stream goodreads_books.json and build a lookup keyed by book_id_csv."""
    log.info("Building item lookup from %s …", BOOKS_PATH)

    wanted = {"book_id", "work_id", "title_without_series", "url", "ratings_count", "authors"}
    records = []

    with open(BOOKS_PATH) as f:
        for line in tqdm(f, desc="Streaming books.json", unit=" books"):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                continue
            records.append({k: raw.get(k) for k in wanted})

    books_df = pd.DataFrame(records)
    books_df["book_id"] = pd.to_numeric(books_df["book_id"], errors="coerce")
    books_df["ratings_count"] = pd.to_numeric(books_df["ratings_count"], errors="coerce")
    books_df = books_df.rename(columns={"title_without_series": "title"})

    # Flatten authors to list of IDs
    def extract_author_ids(author_list):
        if not isinstance(author_list, list):
            return []
        primary = [a["author_id"] for a in author_list
                   if isinstance(a, dict) and a.get("role", "") in ("", "Author")]
        return primary if primary else [a["author_id"] for a in author_list if isinstance(a, dict)]

    books_df["author_ids"] = books_df["authors"].apply(extract_author_ids)
    books_df = books_df.drop(columns=["authors"])

    # Join with book_id_map to get book_id_csv (= matrix column index)
    lookup = book_map.merge(books_df, on="book_id", how="inner")
    log.info("Item lookup: %d books with metadata (out of %d total)", len(lookup), len(book_map))
    return lookup


def load_interactions(alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Read goodreads_interactions.csv in chunks.
    Returns (rows, cols, data) as int32/float32 numpy arrays for COO construction.
    """
    log.info("Loading interactions from %s (this may take a while) …", INTERACTIONS_PATH)

    row_chunks, col_chunks, data_chunks = [], [], []
    total_read = total_kept = 0

    for chunk in tqdm(
        pd.read_csv(INTERACTIONS_PATH, chunksize=1_000_000,
                    dtype={"user_id": np.int32, "book_id": np.int32,
                           "is_read": np.int8, "rating": np.int8,
                           "is_reviewed": np.int8}),
        desc="Loading interactions",
        unit=" chunks",
        total=_APPROX_INTERACTION_CHUNKS,
    ):
        total_read += len(chunk)
        chunk = chunk[chunk["is_read"] == 1]
        total_kept += len(chunk)

        if chunk.empty:
            continue

        rows = chunk["user_id"].values.astype(np.int32)
        cols = chunk["book_id"].values.astype(np.int32)
        ratings = chunk["rating"].values.astype(np.float32)

        # confidence = 1 + alpha * r  (treat unrated reads as r=1)
        confidence = 1.0 + alpha * np.where(ratings > 0, ratings, 1.0)
        data = confidence.astype(np.float32)

        row_chunks.append(rows)
        col_chunks.append(cols)
        data_chunks.append(data)

    log.info("Interactions: %d read / %d kept (is_read=1)", total_read, total_kept)
    return (
        np.concatenate(row_chunks),
        np.concatenate(col_chunks),
        np.concatenate(data_chunks),
    )


def run_training(args) -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    t0 = time.time()

    # ── 1. Item lookup ────────────────────────────────────────────────────
    log.info("Loading book_id_map …")
    book_map = pd.read_csv(BOOK_MAP_PATH)
    book_map.columns = ["book_id_csv", "book_id"]
    book_map["book_id"] = book_map["book_id"].astype(np.int64)

    item_lookup_raw = build_item_lookup(book_map)

    # ── 2. Load interactions ──────────────────────────────────────────────
    rows, cols, data = load_interactions(alpha=args.alpha)

    n_users = int(rows.max()) + 1
    n_items = int(cols.max()) + 1
    log.info("Raw matrix shape: %d users × %d items, %d interactions",
             n_users, n_items, len(rows))

    # ── 3. Cold-start filtering ───────────────────────────────────────────
    log.info("Filtering cold-start users (min %d) and items (min %d) …",
             args.min_user_interactions, args.min_item_interactions)

    item_counts = np.bincount(cols, minlength=n_items)
    user_counts = np.bincount(rows, minlength=n_users)

    keep_items = item_counts >= args.min_item_interactions
    keep_users = user_counts >= args.min_user_interactions

    item_remap = _remap_indices(keep_items)
    user_remap = _remap_indices(keep_users)

    mask = (item_remap[cols] >= 0) & (user_remap[rows] >= 0)
    new_rows = user_remap[rows[mask]]
    new_cols = item_remap[cols[mask]]
    new_data = data[mask]

    n_kept_users = int(keep_users.sum())
    n_kept_items = int(keep_items.sum())
    log.info("After filtering: %d users × %d items, %d interactions",
             n_kept_users, n_kept_items, mask.sum())

    # ── 4. Build CSR matrix ───────────────────────────────────────────────
    log.info("Building CSR matrix …")
    user_items = coo_matrix(
        (new_data, (new_rows, new_cols)),
        shape=(n_kept_users, n_kept_items),
    ).tocsr()

    # ── 5. Train ALS ──────────────────────────────────────────────────────
    log.info("Training ALS: factors=%d, iterations=%d, regularization=%g …",
             args.factors, args.iterations, args.regularization)
    model = AlternatingLeastSquares(
        factors=args.factors,
        iterations=args.iterations,
        regularization=args.regularization,
        random_state=42,
    )
    model.fit(user_items)

    # ── 6. Build final item lookup with new indices ───────────────────────
    # Map original book_id_csv → new item index
    old_to_new = item_remap  # old_book_id_csv → new_item_idx (-1 if filtered)
    item_lookup_raw["item_idx"] = item_lookup_raw["book_id_csv"].map(
        lambda x: int(old_to_new[x]) if x < len(old_to_new) else -1
    )
    item_lookup = item_lookup_raw[item_lookup_raw["item_idx"] >= 0].copy()
    item_lookup = item_lookup.reset_index(drop=True)

    # ── 7. Save artifacts ─────────────────────────────────────────────────
    model_path = MODELS_DIR / "als_model.npz"
    lookup_path = MODELS_DIR / "item_lookup.parquet"
    info_path = MODELS_DIR / "training_info.json"

    log.info("Saving model to %s …", model_path)
    model.save(str(model_path))

    log.info("Saving item lookup (%d items) to %s …", len(item_lookup), lookup_path)
    item_lookup.to_parquet(lookup_path, index=False)

    elapsed = time.time() - t0
    training_info = {
        "n_users": n_kept_users,
        "n_items": n_kept_items,
        "n_interactions": int(mask.sum()),
        "factors": args.factors,
        "iterations": args.iterations,
        "regularization": args.regularization,
        "alpha": args.alpha,
        "min_user_interactions": args.min_user_interactions,
        "min_item_interactions": args.min_item_interactions,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": round(elapsed, 1),
    }
    with open(info_path, "w") as f:
        json.dump(training_info, f, indent=2)

    log.info("Done in %.0f s. Artifacts in %s/", elapsed, MODELS_DIR)
    log.info("  %s", model_path)
    log.info("  %s", lookup_path)
    log.info("  %s", info_path)
