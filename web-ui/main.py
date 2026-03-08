"""
Web UI server — middleman between the browser and the recommender API.

Routes
------
GET  /            — main page
POST /search      — search books, returns carousel HTML fragment
POST /recommend   — get ALS recommendations, returns carousel HTML fragment
"""

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

RECOMMENDER_API_URL = os.getenv("RECOMMENDER_API_URL", "http://localhost:8001")

app = FastAPI()
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


def _error_html(message: str) -> HTMLResponse:
    return HTMLResponse(
        f'<div class="text-red-600 p-4 rounded bg-red-50 border border-red-200">'
        f"Error: {message}"
        f"</div>"
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, q: str = Form(...)):
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{RECOMMENDER_API_URL}/search", params={"q": q, "n": 20}
            )
            resp.raise_for_status()
            books = resp.json()
    except httpx.HTTPStatusError as e:
        return _error_html(f"Recommender API returned {e.response.status_code}")
    except Exception as e:
        return _error_html(f"Could not reach recommender API: {e}")

    return templates.TemplateResponse(
        "partials/book_carousel.html",
        {
            "request": request,
            "heading": "Search Results",
            "books": [
                {
                    "work_id": b["work_id"],
                    "title": b["title"] or "Unknown Title",
                    "stat_label": "ratings",
                    "stat_value": f'{b["ratings_count"]:,}' if b.get("ratings_count") else "—",
                }
                for b in books
            ],
        },
    )


@app.post("/recommend", response_class=HTMLResponse)
async def recommend(request: Request, work_id: str = Form(...)):
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{RECOMMENDER_API_URL}/get-als-recommendation",
                json={"work_id": work_id, "n": 10},
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return _error_html(f"Recommender API returned {e.response.status_code}")
    except Exception as e:
        return _error_html(f"Could not reach recommender API: {e}")

    seed_title = data.get("title") or work_id
    return templates.TemplateResponse(
        "partials/book_carousel.html",
        {
            "request": request,
            "heading": f"Matrix Factorization (ALS) — similar to: {seed_title}",
            "books": [
                {
                    "work_id": b["work_id"],
                    "title": b["title"] or "Unknown Title",
                    "stat_label": "score",
                    "stat_value": f'{b["score"]:.3f}',
                }
                for b in data.get("similar_books", [])
            ],
        },
    )
