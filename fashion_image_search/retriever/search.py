"""Two-stage fashion retrieval: global ANN recall → region-aware compositional rerank."""

from __future__ import annotations

import argparse
import math
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


# ── Category matching ─────────────────────────────────────────────────────────

# Canonical expansion map: query canonical type → set of index canonical types
# that count as a match.  After Bug #1 fix, region.category is already a clean
# canonical string, so we only need the semantically related expansions here.
_CATEGORY_EXPANSIONS: dict[str, set[str]] = {
    "coat":     {"coat", "raincoat", "jacket", "blazer"},
    "raincoat": {"raincoat", "coat"},
    "blazer":   {"blazer", "jacket", "coat"},
    "jacket":   {"jacket", "blazer", "coat"},
    "shirt":    {"shirt"},
    "pants":    {"pants", "shorts"},
    "shorts":   {"shorts", "pants"},
    "dress":    {"dress"},
    "skirt":    {"skirt"},
    "hoodie":   {"hoodie", "shirt"},
    "vest":     {"vest"},
    "tie":      {"tie"},
    "scarf":    {"scarf"},
    "hat":      {"hat"},
    "shoe":     {"shoe"},
    "glasses":  {"glasses"},
    "glove":    {"glove"},
}


def _category_matches(query_category: str | None, indexed_category: str) -> bool:
    """Return True when *indexed_category* satisfies the *query_category* slot.

    Uses semantic expansion so that a "coat" query matches "raincoat" detections,
    "blazer" queries match "jacket" detections, etc.
    """
    if query_category is None:
        return False
    if query_category == indexed_category:
        return True
    expansion = _CATEGORY_EXPANSIONS.get(query_category)
    if expansion and indexed_category in expansion:
        return True
    return False


# ── Per-slot scoring ──────────────────────────────────────────────────────────

def _score_slot(
    slot: GarmentSlot,
    record: ImageRecord,
    embedding_backend: str,
) -> tuple[float, float, str]:
    """Score a single garment slot against a record's regions.

    Returns ``(embedding_similarity, attribute_bonus, description_string)``.

    Key fix (Bug #7): when the slot has a ``garment_type`` but NO compatible
    region exists for that type, we return 0.0 immediately instead of falling
    back to scoring all regions (which was producing false-positive slot scores
    for unrelated garments).
    """
    if not record.regions:
        return 0.0, 0.0, f"{slot.phrase}: no regions indexed"

    query_embedding = embed_text_backend(
        slot.phrase or " ".join(filter(None, [slot.color, slot.garment_type])),
        embedding_backend,
    )

    compatible_regions = [
        region for region in record.regions
        if _category_matches(slot.garment_type, region.category)
    ]

    # Bug #7 fix: typed slot with no matching region → return 0 (no fallback)
    if slot.garment_type is not None and not compatible_regions:
        return (
            0.0,
            0.0,
            f"{slot.phrase}: no {slot.garment_type} region found "
            f"(regions: {[r.category for r in record.regions]})",
        )

    # Untyped slot (color-only): fall back to all regions
    regions = compatible_regions if compatible_regions else record.regions

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
                f"{slot.phrase or slot.garment_type}: "
                f"region={region.category}, color={region.color}, "
                f"sim={sim:.3f}, bonus={bonus:.2f}"
            )

    return max(best_score - best_bonus, 0.0), best_bonus, best_description


# ── Multi-slot metadata pre-filter ────────────────────────────────────────────

def _record_matches_slots(record: ImageRecord, slots: list[GarmentSlot]) -> bool:
    """Pre-filter: decide whether *record* is a plausible candidate for *slots*.

    Single-slot query  → OR logic (lenient): any region matching any attribute passes.
    Multi-slot query   → AND logic (strict, Bug #4 fix): EVERY slot must be
                         represented by at least one region in the record.

    This prevents a record containing only "red pants" from passing the filter
    for a "red tie + white shirt" query just because the color "red" matches the
    tie slot.
    """
    if not slots:
        return True  # no slot constraints — include everything

    if len(slots) == 1:
        # Single-slot: OR logic for maximum recall
        slot = slots[0]
        for region in record.regions:
            category_ok = _category_matches(slot.garment_type, region.category)
            color_ok = bool(slot.color and slot.color == region.color)
            if category_ok or color_ok:
                return True
        return False

    # Multi-slot: AND logic — every slot must fire at least once
    for slot in slots:
        slot_satisfied = False
        for region in record.regions:
            category_ok = _category_matches(slot.garment_type, region.category)
            color_ok = bool(slot.color and slot.color == region.color)
            if category_ok or color_ok:
                slot_satisfied = True
                break
        if not slot_satisfied:
            return False
    return True


# ── Record reranking ──────────────────────────────────────────────────────────

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
        cosine_similarity(
            embed_text_backend(parsed.scene_phrase, embedding_backend),
            record.scene_embedding,
        )
        if parsed.scene_phrase
        else 0.0
    )

    score = (
        SEARCH.global_weight * global_score
        + SEARCH.slot_weight * mean_slot_score
        + SEARCH.scene_weight * scene_score
        + SEARCH.attribute_bonus_weight * attribute_bonus
    )

    # Style-residual fallback: when there are no garment slots (e.g. "casual
    # weekend outfit"), compare the style phrase against the global embedding.
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


# ── Public search API ─────────────────────────────────────────────────────────

def search(
    index_path: Path,
    query: str,
    top_k: int = SEARCH.default_top_k,
    embedding_backend: str = MODEL.backend,
    store_kind: str = "faiss",
    faiss_path: Path = PATHS.faiss_index_path,
    parser_backend: str = "rule",
) -> list[SearchResult]:
    """Two-stage fashion retrieval.

    Stage 1: ANN global recall (top ``SEARCH.stage1_k`` candidates via FAISS).
    Stage 2: Compositional region-aware rerank with per-slot scoring.
    """
    store = make_vector_store(store_kind, index_path, faiss_path).load()
    parsed = parse_query(query, parser_backend)
    query_embedding = embed_text_backend(query, embedding_backend)

    shortlist = store.search_global(query_embedding, SEARCH.stage1_k)

    # Apply slot-aware pre-filter (AND mode for multi-slot, OR for single-slot)
    filtered = [
        item for item in shortlist
        if _record_matches_slots(item[0], parsed.garment_slots)
    ]
    if filtered:
        shortlist = filtered

    results = [
        rerank_record(record, global_score, query, embedding_backend, parser_backend, parsed)
        for record, global_score in shortlist
    ]
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_k]


# ── CLI ───────────────────────────────────────────────────────────────────────

def _format_parsed_query(parsed: ParsedQuery) -> str:
    """Human-readable one-liner of the parsed query structure."""
    parts: list[str] = []
    for slot in parsed.garment_slots:
        tag = " ".join(filter(None, [slot.color, slot.garment_type])) or slot.phrase
        parts.append(f"[{tag}]")
    if parsed.scene_phrase:
        parts.append(f"📍 {parsed.scene_phrase}")
    if parsed.style_residual:
        parts.append(f"✨ {parsed.style_residual}")
    return "  ".join(parts) if parts else "(no structured slots)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Search indexed fashion images.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=PATHS.index_path)
    parser.add_argument("--faiss-index", type=Path, default=PATHS.faiss_index_path)
    parser.add_argument("--top-k", type=int, default=SEARCH.default_top_k)
    parser.add_argument("--backend", default=MODEL.backend,
                        choices=["offline", "hf", "huggingface"])
    parser.add_argument("--store", default="faiss", choices=["faiss", "json"])
    parser.add_argument("--parser", default="rule",
                        choices=["rule", "openai", "opencode", "openai-compatible"])
    parser.add_argument("--explain", action="store_true",
                        help="Print parsed query slots + per-score breakdown.")
    args = parser.parse_args()

    results = search(
        args.index, args.query, args.top_k,
        args.backend, args.store, args.faiss_index, args.parser,
    )

    if args.explain:
        # Show the parsed query structure so users can verify it
        parsed = parse_query(args.query, args.parser)
        print(f"\nParsed as: {_format_parsed_query(parsed)}\n")

    for rank, result in enumerate(results, start=1):
        print(f"{rank}. {result.image_path}  score={result.score:.3f}")
        if args.explain:
            print(
                f"   global={result.global_score:.3f}  scene={result.scene_score:.3f}  "
                f"slot={result.slot_score:.3f}  attr={result.attribute_bonus:.3f}"
            )
            for line in result.slot_breakdown:
                print(f"   - {line}")


if __name__ == "__main__":
    main()
