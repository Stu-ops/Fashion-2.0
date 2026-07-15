"""Run the assignment's five evaluation queries."""

from __future__ import annotations

import argparse
from pathlib import Path

from fashion_image_search.common.config import MODEL, PATHS, SEARCH
from fashion_image_search.retriever.search import search


EVALUATION_QUERIES = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fashion retrieval pipeline.")
    parser.add_argument("--index", type=Path, default=PATHS.index_path)
    parser.add_argument("--faiss-index", type=Path, default=PATHS.faiss_index_path)
    parser.add_argument("--top-k", type=int, default=SEARCH.default_top_k)
    parser.add_argument("--backend", default=MODEL.backend, choices=["offline", "hf", "huggingface"])
    parser.add_argument("--store", default="faiss", choices=["faiss", "json"])
    parser.add_argument("--parser", default="rule", choices=["rule", "openai", "opencode", "openai-compatible"])
    args = parser.parse_args()
    for query in EVALUATION_QUERIES:
        print(f"\nQUERY: {query}")
        for rank, result in enumerate(
            search(args.index, query, args.top_k, args.backend, args.store, args.faiss_index, args.parser),
            start=1,
        ):
            print(f"{rank}. {result.image_path} score={result.score:.3f}")
            for line in result.slot_breakdown:
                print(f"   - {line}")


if __name__ == "__main__":
    main()
