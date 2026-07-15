"""Two-stage fashion retrieval: global recall, region-aware rerank."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from fashion_image_search.common.config import MODEL, PATHS, SEARCH
from fashion_image_search.common.schemas import GarmentSlot, ImageRecord, ParsedQuery
from fashion_image_search.common.vector_db import cosine_similarity, make_vector_store
from fashion_image_search.indexer.embed import embed_text_backend
from fashion_image_search.retriever.parse_query import parse_query


@dataclass
class SearchResult:
    image_id: str
    image_path: str
    score: float
    global_score: float
    scene_score: float
    slot_score: float
    attribute_bonus: float
    slot_breakdown: list[str]


def _category_matches(query_category: str | None, indexed_category: str) -> bool:
    if query_category is None:
        return False
    if query_category == indexed_category:
        return True
    if query_category == "raincoat" and indexed_category in {"coat", "jacket"}:
        return True
    if query_category == "blazer" and indexed_category in {"jacket", "coat"}:
        return True
    return False


def _score_slot(slot: GarmentSlot, record: ImageRecord, embedding_backend: str) -> tuple[float, float, str]:
    if not record.regions:
        return 0.0, 0.0, f"{slot.phrase}: no regions"
    query_embedding = embed_text_backend(
        slot.phrase or " ".join(filter(None, [slot.color, slot.garment_type])),
        embedding_backend,
    )
    compatible_regions = [
        region for region in record.regions
        if _category_matches(slot.garment_type, region.category)
    ]
    regions = compatible_regions or record.regions
    best_score = -1.0
    best_bonus = 0.0
    best_description = ""
    for region in regions:
        sim = cosine_similarity(query_embedding, region.region_embedding)
        bonus = 0.0
        if slot.color and slot.color == region.color:
            bonus += 0.4
        if _category_matches(slot.garment_type, region.category):
            bonus += 0.6
        combined = sim + bonus
        if combined > best_score:
            best_score = combined
            best_bonus = bonus
            best_description = (
                f"{slot.phrase or slot.garment_type}: region={region.category}, "
                f"color={region.color}, sim={sim:.3f}, bonus={bonus:.2f}"
            )
    return max(best_score - best_bonus, 0.0), best_bonus, best_description


def rerank_record(
    record: ImageRecord,
    global_score: float,
    query_text: str,
    embedding_backend: str = MODEL.backend,
    parser_backend: str = "rule",
    parsed_query: ParsedQuery | None = None,
) -> SearchResult:
    parsed = parsed_query or parse_query(query_text, parser_backend)
    slot_scores: list[float] = []
    bonuses: list[float] = []
    breakdown: list[str] = []
    for slot in parsed.garment_slots:
        slot_score, bonus, description = _score_slot(slot, record, embedding_backend)
        slot_scores.append(slot_score)
        bonuses.append(bonus)
        breakdown.append(description)
    mean_slot_score = sum(slot_scores) / len(slot_scores) if slot_scores else 0.0
    attribute_bonus = sum(bonuses) / len(bonuses) if bonuses else 0.0
    scene_score = (
        cosine_similarity(embed_text_backend(parsed.scene_phrase, embedding_backend), record.scene_embedding)
        if parsed.scene_phrase
        else 0.0
    )
    score = (
        SEARCH.global_weight * global_score
        + SEARCH.slot_weight * mean_slot_score
        + SEARCH.scene_weight * scene_score
        + SEARCH.attribute_bonus_weight * attribute_bonus
    )
    if not parsed.garment_slots:
        score += 0.20 * cosine_similarity(
            embed_text_backend(parsed.style_residual or query_text, embedding_backend),
            record.global_embedding,
        )
    return SearchResult(
        image_id=record.image_id,
        image_path=record.image_path,
        score=score,
        global_score=global_score,
        scene_score=scene_score,
        slot_score=mean_slot_score,
        attribute_bonus=attribute_bonus,
        slot_breakdown=breakdown,
    )


def search(
    index_path: Path,
    query: str,
    top_k: int = SEARCH.default_top_k,
    embedding_backend: str = MODEL.backend,
    store_kind: str = "faiss",
    faiss_path: Path = PATHS.faiss_index_path,
    parser_backend: str = "rule",
) -> list[SearchResult]:
    store = make_vector_store(store_kind, index_path, faiss_path).load()
    parsed = parse_query(query, parser_backend)
    query_embedding = embed_text_backend(query, embedding_backend)
    shortlist = store.search_global(query_embedding, SEARCH.stage1_k)
    filtered = [item for item in shortlist if _record_matches_any_slot(item[0], parsed.garment_slots)]
    if filtered:
        shortlist = filtered
    results = [
        rerank_record(record, global_score, query, embedding_backend, parser_backend, parsed)
        for record, global_score in shortlist
    ]
    results.sort(key=lambda result: result.score, reverse=True)
    return results[:top_k]


def _record_matches_any_slot(record: ImageRecord, slots: list[GarmentSlot]) -> bool:
    if not slots:
        return True
    for slot in slots:
        for region in record.regions:
            category_match = _category_matches(slot.garment_type, region.category)
            color_match = bool(slot.color and slot.color == region.color)
            if category_match or color_match:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed fashion images.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=PATHS.index_path)
    parser.add_argument("--faiss-index", type=Path, default=PATHS.faiss_index_path)
    parser.add_argument("--top-k", type=int, default=SEARCH.default_top_k)
    parser.add_argument("--backend", default=MODEL.backend, choices=["offline", "hf", "huggingface"])
    parser.add_argument("--store", default="faiss", choices=["faiss", "json"])
    parser.add_argument("--parser", default="rule", choices=["rule", "openai", "opencode", "openai-compatible"])
    parser.add_argument("--explain", action="store_true")
    args = parser.parse_args()
    for rank, result in enumerate(
        search(args.index, args.query, args.top_k, args.backend, args.store, args.faiss_index, args.parser),
        start=1,
    ):
        print(f"{rank}. {result.image_path} score={result.score:.3f}")
        if args.explain:
            print(
                f"   global={result.global_score:.3f} scene={result.scene_score:.3f} "
                f"slot={result.slot_score:.3f} attr={result.attribute_bonus:.3f}"
            )
            for line in result.slot_breakdown:
                print(f"   - {line}")


if __name__ == "__main__":
    main()
