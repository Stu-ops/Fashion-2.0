"""Run the assignment's five evaluation queries and optionally save results to JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fashion_image_search.common.config import MODEL, PATHS, SEARCH
from fashion_image_search.retriever.parse_query import parse_query
from fashion_image_search.retriever.search import search, _format_parsed_query


EVALUATION_QUERIES: list[str] = [
    "A person in a bright yellow raincoat.",
    "Professional business attire inside a modern office.",
    "Someone wearing a blue shirt sitting on a park bench.",
    "Casual weekend outfit for a city walk.",
    "A red tie and a white shirt in a formal setting.",
]


def run_evaluation(
    index_path: Path,
    faiss_path: Path,
    top_k: int,
    backend: str,
    store_kind: str,
    parser_backend: str,
    output_path: Path | None,
) -> list[dict]:
    """Execute all evaluation queries and return structured results."""
    all_results: list[dict] = []

    for query in EVALUATION_QUERIES:
        print(f"\n{'=' * 70}")
        print(f"QUERY: {query}")

        parsed = parse_query(query, parser_backend)
        print(f"Parsed as: {_format_parsed_query(parsed)}")

        results = search(
            index_path, query, top_k, backend, store_kind, faiss_path, parser_backend,
        )

        query_record: dict = {
            "query": query,
            "parsed_slots": [
                {"color": s.color, "garment_type": s.garment_type, "phrase": s.phrase}
                for s in parsed.garment_slots
            ],
            "scene_phrase": parsed.scene_phrase,
            "style_residual": parsed.style_residual,
            "results": [],
        }

        for rank, result in enumerate(results, start=1):
            print(f"  {rank}. {result.image_path}  score={result.score:.3f}")
            for line in result.slot_breakdown:
                print(f"       - {line}")

            query_record["results"].append({
                "rank": rank,
                "image_path": result.image_path,
                "image_id": result.image_id,
                "score": round(result.score, 6),
                "global_score": round(result.global_score, 6),
                "scene_score": round(result.scene_score, 6),
                "slot_score": round(result.slot_score, 6),
                "attribute_bonus": round(result.attribute_bonus, 6),
                "slot_breakdown": result.slot_breakdown,
            })

        all_results.append(query_record)

    return all_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fashion retrieval pipeline.")
    parser.add_argument("--index", type=Path, default=PATHS.index_path)
    parser.add_argument("--faiss-index", type=Path, default=PATHS.faiss_index_path)
    parser.add_argument("--top-k", type=int, default=SEARCH.default_top_k)
    parser.add_argument("--backend", default=MODEL.backend,
                        choices=["offline", "hf", "huggingface"])
    parser.add_argument("--store", default="faiss", choices=["faiss", "json"])
    parser.add_argument("--parser", default="rule",
                        choices=["rule", "openai", "opencode", "openai-compatible"])
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to write evaluation results JSON (e.g. artifacts/eval_results.json).",
    )
    args = parser.parse_args()

    results = run_evaluation(
        args.index,
        args.faiss_index,
        args.top_k,
        args.backend,
        args.store,
        args.parser,
        args.output,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nEvaluation results saved to: {args.output}")


if __name__ == "__main__":
    main()
