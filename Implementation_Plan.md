# Implementation Plan — Multimodal Fashion & Context Retrieval
### Glance ML Internship Assignment

This plan translates the chosen architecture (Approach D — region-aware hybrid retrieval, from
`Fashion_Retrieval_Approach_Writeup.pdf`) into a concrete build order, with the specific models selected
after research, the data structures each stage produces/consumes, and how the five evaluation queries map
to test cases. No code is written yet — this is the blueprint to execute against.

---

## 1. Models Selected (final)

| Role | Model | Why |
|---|---|---|
| **Garment detector** | `valentinafevu/yolos-fashionpedia` (YOLOS, transformer detection family) | Only fashion-trained detector found with fine-grained Fashionpedia categories (46 classes: shirt/blouse, tie, jacket, coat, dress, glove, belt, etc. — plus parts like collar/sleeve/lapel). Needed so "tie" and "shirt" don't collapse into one bucket, which would silently reintroduce the compositionality problem at the garment-type level. |
| **CLIP encoder (image + text)** | `Marqo/marqo-fashionSigLIP` | Fashion-tuned via Generalised Contrastive Learning on category + style + color + material jointly, not just captions. Benchmarked ahead of `patrickjohncyh/fashion-clip` (FashionCLIP 2.0) by a wide margin (avg recall 0.231 vs 0.163 on the public 7-dataset leaderboard) and confirmed as the strongest publicly available *paired text–image* fashion model in the independent Jan-2026 LookBench study. Drop-in replacement for OWL-ViT/CLIP in the existing `clip_utils.py` interface — still gives a text tower, so Stage 1 global recall and Stage 2 region rerank both keep working unmodified in shape. |
| Scene/context branch | `Marqo/marqo-fashionSigLIP` (whole-image embedding), optionally + a Places365-style classifier later | No change from original plan — reuses the same CLIP-class model rather than a third dependency. |
| *Not adopted* | `GR-Lite` (DINOv3-based, SOTA on LookBench for pure image similarity) | Text-free — no text tower, so it can't serve NL queries directly. Worth one sentence in the write-up as "frontier for image-only fashion embeddings," but wrong shape for this assignment's core requirement (text → image). |

**Net effect vs. the original write-up:** the architecture (regions + scene + global, retrieve-then-rerank)
is unchanged. Only the two concrete model checkpoints are upgraded from generic zero-shot (OWL-ViT +
vanilla CLIP) to fashion-specialized, benchmarked alternatives — which directly strengthens the "better
than vanilla CLIP on fashion" claim the assignment explicitly asks for.

---

## 2. Repository Layout (target — matches Section 4 of the PDF)

```
fashion_image_search/
├── indexer/
│   ├── detect.py          # YOLOS-fashionpedia region detection + cropping
│   ├── embed.py           # marqo-fashionSigLIP: region / scene / global embeddings
│   ├── attributes.py      # colour extraction per crop (histogram fallback)
│   └── build_index.py     # orchestrates detect→embed→attributes, writes to vector DB
├── retriever/
│   ├── parse_query.py     # NL query → {garment,colour} slots + scene + style residual
│   ├── search.py          # Stage 1 ANN recall + Stage 2 compositional rerank
│   └── evaluate.py        # runs the 5 assignment eval queries, reports top-k
├── common/
│   ├── config.py          # model names, thresholds, k, DB config
│   └── vector_db.py       # thin wrapper (Chroma or Qdrant client)
├── tests/
│   └── test_pipeline_offline.py   # fake-embedding harness (no GPU/network needed)
└── README.md
```

---

## 3. Data Contracts (the shape every stage must agree on)

Fixing these now avoids the rewrite churn that hit the codebase last time (v1 tagging → v2 regions).

**Per-image record, written by the indexer:**
```
image_id, image_path,
global_embedding        : float[dim]          # marqo-fashionSigLIP, whole image
scene_embedding         : float[dim]           # marqo-fashionSigLIP, whole image (separate slot, same model)
regions: [
  {
    region_idx, bbox, category (one of 46 Fashionpedia classes),
    detector_confidence,
    region_embedding    : float[dim]           # marqo-fashionSigLIP, cropped region only
    color, color_confidence
  }, ... up to N_MAX regions (pad/truncate, e.g. 6)
]
```

**Per-query parsed structure, produced by the retriever:**
```
garment_slots: [ {garment_type, colour} , ... ]   # order-preserving, in-text-order parsing (already fixed bug)
scene_phrase: str | null
style_residual: str | null
full_query_text_embedding : float[dim]            # always computed, open-vocab fallback
```

---

## 4. Build Order (phased, so each phase is independently testable)

**Phase 0 — Environment check**
- Confirm the target machine has real network/model access (HF Hub) before running anything beyond the
  offline harness — this sandbox does not, so real weight downloads and quality validation happen
  elsewhere.

**Phase 1 — Indexer, detector stage**
- `detect.py`: load `valentinafevu/yolos-fashionpedia`, run on each image, keep boxes above a confidence
  threshold (start at 0.5, tune later), cap at `N_MAX` regions per image, crop each box to a PIL image.
- Test: run on a handful of sample images, visually confirm boxes land on the right garments.

**Phase 2 — Indexer, embedding stage**
- `embed.py`: load `Marqo/marqo-fashionSigLIP` once, expose three functions —
  `embed_regions(crops)`, `embed_scene(full_image)`, `embed_global(full_image)`.
- `attributes.py`: dominant-colour extraction per crop (histogram over the masked/cropped region).
- Test: confirm embedding dimensions match config, confirm colour extraction returns sane labels on a few
  known-colour crops.

**Phase 3 — Indexer, storage**
- `build_index.py`: wire detect → embed → attributes into one pass over the dataset, write everything into
  Chroma (simplest local, no server) with the per-image record shape from Section 3.
- Test: index a small (~50 image) subset, confirm record count and shape in the DB.

**Phase 4 — Retriever, query parsing**
- `parse_query.py` — already implemented and correct (in-text-order slot extraction, verified against the
  offline harness). Just needs to be re-pointed at the new attribute vocabulary if the Fashionpedia category
  names differ from the current vocab file.

**Phase 5 — Retriever, search**
- `search.py` Stage 1: ANN search on `global_embedding` (top ~200), optional metadata pre-filter.
- `search.py` Stage 2: for each candidate in the shortlist, for each parsed garment slot, embed the slot
  phrase with the CLIP text encoder and take max cosine similarity against that candidate's region
  embeddings; match scene slot against `scene_embedding`; add an attribute-match bonus when structured
  colour/category metadata agrees exactly; combine into one weighted score.
- Test: run against the Phase-3 mini-index with known content, confirm expected images rank near the top.

**Phase 6 — Evaluation**
- `evaluate.py`: run all 5 assignment queries (Section 5 below), print/save top-k with scores for manual
  inspection.

**Phase 7 — Offline test harness update**
- Extend `test_pipeline_offline.py` to mock `detect.py`'s output (fake boxes) and `embed.py`'s output (fake
  vectors), so the full pipeline shape can be validated without GPU/model access — same pattern that
  already caught the query-parser ordering bug.

**Phase 8 — README + PDF write-up sync**
- Update README to describe the actual shipped architecture (region-aware, marqo-fashionSigLIP +
  yolos-fashionpedia) — no overclaiming, matches what Section 6 of the PDF asks for on limitations.

---

## 5. Evaluation Query → Pipeline Mapping (sanity checklist before calling it done)

| Query | What must work |
|---|---|
| "A person in a bright yellow raincoat." | Single region detected as outerwear/coat category; colour extractor returns yellow; high attribute-match bonus. |
| "Professional business attire inside a modern office." | Scene embedding matches office-like images; garment categories (blazer/shirt) cluster toward "formal"; global embedding contributes the style signal since "professional" isn't a literal attribute. |
| "Someone wearing a blue shirt sitting on a park bench." | Garment slot (blue, shirt) matched at region level; scene slot matches park; "sitting on a bench" is *not* expected to be solved precisely — flag this as a known limitation in the final write-up, not a bug. |
| "Casual weekend outfit for a city walk." | No literal garment/colour in the query — must rely on global embedding style signal + scene (urban/street), not on Stage 2 region matching. |
| "A red tie and a white shirt in a formal setting." | **The core test.** Two slots (red/tie, white/shirt) must each match their own detected region, not swap. This is exactly the case the old label-matching v1 logic and the query-parser bug both failed differently — confirm explicitly with a printed per-slot score breakdown, not just the final top-k. |

---

## 6. Explicitly Out of Scope for This Build (goes in Future Work section of the PDF, not code)

- Real location/landmark recognition (needs EXIF/metadata, not inferable from clothing).
- Weather classification branch.
- Segmentation-mask-based cropping (vs. bounding boxes) for cleaner colour extraction.
- Hard-negative fine-tuning (colour-swapped pairs) on top of `marqo-fashionSigLIP`.
- Learned reranker replacing the hand-weighted Stage 2 score.
- Stage 3 cross-attention VLM reranker (Approach E) for relational phrasing ("tucked into").

---

## 7. Immediate Next Action

Once you give the go-ahead: start Phase 1 (`detect.py` against `valentinafevu/yolos-fashionpedia`) since
every later phase depends on region outputs having the right shape. Nothing else should be coded before
Phase 1 is confirmed working, to avoid another shape-mismatch rewrite like the v1→v2 detector correction.
