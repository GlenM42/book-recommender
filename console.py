"""
Book Recommender CLI

Usage
-----
  python console.py train  [options]   Train the ALS model
  python console.py serve  [options]   Start the recommendation API server
"""

import argparse
import logging
import sys


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="console",
        description="Book Recommender — ALS",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──────────────────────────────────────────────────────────────
    train_p = sub.add_parser("train", help="Train the ALS model on interaction data")
    train_p.add_argument(
        "--factors", type=int, default=64,
        help="Number of latent factors (default: 64)",
    )
    train_p.add_argument(
        "--iterations", type=int, default=20,
        help="ALS iterations (default: 20)",
    )
    train_p.add_argument(
        "--regularization", type=float, default=0.01,
        help="L2 regularisation strength (default: 0.01)",
    )
    train_p.add_argument(
        "--alpha", type=float, default=40.0,
        help="Confidence scaling: c = 1 + alpha * rating (default: 40.0)",
    )
    train_p.add_argument(
        "--min-item-interactions", type=int, default=10, dest="min_item_interactions",
        help="Drop items with fewer interactions than this (default: 10)",
    )
    train_p.add_argument(
        "--min-user-interactions", type=int, default=5, dest="min_user_interactions",
        help="Drop users with fewer interactions than this (default: 5)",
    )

    # ── serve ──────────────────────────────────────────────────────────────
    serve_p = sub.add_parser("serve", help="Start the FastAPI recommendation server")
    serve_p.add_argument(
        "--host", default="0.0.0.0",
        help="Bind host (default: 0.0.0.0)",
    )
    serve_p.add_argument(
        "--port", type=int, default=8000,
        help="Bind port (default: 8000)",
    )
    serve_p.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload on code changes (development only)",
    )

    return parser


def main() -> None:
    _configure_logging()
    args = _build_parser().parse_args()

    if args.command == "train":
        from als.train import run_training
        run_training(args)

    elif args.command == "serve":
        from als.serve import run_server
        run_server(args)


if __name__ == "__main__":
    main()
