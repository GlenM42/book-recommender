# Book Recommender

A collaborative-filtering book recommendation system built on the [Goodreads dataset](https://mengtingwan.github.io/data/goodreads.html).

## Architecture

The project is split into two independent services:

| Repo | Description |
|------|-------------|
| [book-recommender-ml](https://github.com/GlenM42/book-recommender-ml) | ALS model training pipeline and FastAPI recommendation API |
| [book-recommender-ui](https://github.com/GlenM42/book-recommender-ui) | FastAPI + Jinja2 web UI that queries the ML API |

```
Browser → book-recommender-ui
                 ↓ HTTP
          book-recommender-ml
                 ↓ reads
            model artifacts
```

## How it works

1. The ML repo trains an [ALS](https://implicit.readthedocs.io/en/latest/als.html) collaborative-filtering model on ~228M user–book interactions
2. The trained model is served as a REST API (`POST /get-als-recommendation`, `GET /search`)
3. The UI calls the API and renders results as a book carousel

## Getting started

Clone and run each service independently — see their READMEs for setup instructions.
