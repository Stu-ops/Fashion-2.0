# Codebase Architecture Improvements Beyond the PDF

This note compares the current implementation with `docs/Fashion_Retrieval_Approach_Writeup.pdf`.
The PDF describes the intended hybrid retrieval architecture well, but the codebase now contains several concrete engineering and architecture details that are either missing from the PDF or only described at a high level.

## What Matches the PDF

- The project is split into `indexer/`, `retriever/`, and `common/` modules.
- The indexer stores global image embeddings, garment-region embeddings, scene embeddings, and structured region metadata.
- The retriever uses a two-stage flow: global ANN recall first, then region-aware reranking.
- Natural-language queries are parsed into garment slots, scene phrases, and style residual text.
- The five assignment evaluation queries are represented in `fashion_image_search/retriever/evaluate.py`.

## Implemented Improvements Not Fully Captured in the PDF

### 1. Local FAISS + JSON Vector Store

The PDF proposes Chroma or Qdrant as the simplest vector database choices. The implementation instead provides a local FAISS HNSW store plus JSON metadata sidecar in `fashion_image_search/common/vector_db.py`.

Why this improves the project:

- It keeps the submission self-contained with no external vector database service.
- FAISS provides scalable ANN recall for Stage 1.
- JSON keeps rich metadata available for Stage 2 reranking.
- A `JsonVectorStore` fallback supports exact-search tests without FAISS.

### 2. Offline Backend for Testing and Demos

The PDF focuses on the real model path. The codebase adds an `offline` backend with deterministic image/text features and fixed garment proposals.

Why this improves the project:

- Unit tests run without network access or model downloads.
- The Streamlit UI and CLI can be demonstrated on machines without GPU support.
- Pipeline behavior remains testable even when Hugging Face model loading is unavailable.

### 3. Canonical Fashionpedia Label Normalization

The implementation normalizes raw Fashionpedia detector labels into the same canonical garment vocabulary used by the query parser.

Examples:

- `"shirt, blouse"` -> `"shirt"`
- `"top, t-shirt, sweatshirt"` -> `"shirt"`
- part labels like `"collar"` and `"sleeve"` -> discarded

Why this improves the project:

- Query categories and detector categories can actually match.
- Part-level detector noise is filtered before the `max_regions` cap.
- Region matching becomes more reliable for compositional queries.

### 4. Slot-Aware Metadata Prefilter

Before reranking, `search.py` applies a slot-aware prefilter:

- Single-slot queries use lenient OR behavior for recall.
- Multi-slot queries require every slot to be represented.

Why this improves the project:

- A query like `red tie and white shirt` no longer lets an image with only red pants pass just because the color red matched.
- Stage 2 receives a cleaner shortlist before weighted reranking.

### 5. Safer Per-Slot Scoring

The reranker now returns zero for a typed slot when no compatible region exists.

Why this improves the project:

- A `tie` query cannot receive a high score from unrelated regions.
- False positives are reduced for multi-garment searches.

### 6. Scene Embedding Is Distinct From Global Embedding

The code explicitly avoids reusing the same vector for both global and scene scoring.

Why this improves the project:

- Scene/context terms such as `office`, `park`, and `street` have a separate scoring path.
- The weighted score better reflects the PDF's intended global + scene + region architecture.

### 7. Long-Running App and Windows File-Handle Handling

The Streamlit app and vector store include operational details not covered by the PDF:

- FAISS handles are closed before rebuilding or resetting indexes.
- Windows file locks are handled with cleanup and retry logic.
- Dataset upload, auto-indexing, pagination, reset, status display, and GPU/device status are available in the UI.

Why this improves the project:

- The app is easier to use as a complete demo, not just a library.
- Rebuilding indexes from the UI is more reliable on Windows.

### 8. Parser Credential Handling and Diagnostics

The parser now reloads `.env` with override behavior for long-running Streamlit sessions and reports HTTP 401/403 failures with provider/model context while redacting API keys.

Why this improves the project:

- Edited `.env` values can be picked up without stale credentials winning silently.
- Provider/model/key mismatches are easier to diagnose.
- Full secrets are not printed in logs or exceptions.

### 9. Regression Tests for Known Failure Modes

The test suite covers several practical retrieval bugs:

- Color-to-garment binding.
- Jacket alias resolution.
- Skirt and shorts vocabulary.
- Fashionpedia label normalization.
- Multi-slot filter behavior.
- LLM parser success, fallback, `.env` override, and HTTP diagnostics.
- JSON and FAISS smoke tests.

Why this improves the project:

- The architecture is backed by executable evidence.
- Future changes are less likely to break the assignment's key compositional behavior.

## Suggested PDF Update

If the PDF is revised, add a short implementation-specific section after `4. Codebase` covering:

- FAISS HNSW + JSON metadata sidecar instead of Chroma/Qdrant.
- Offline backend for tests and no-network demos.
- Canonical Fashionpedia label normalization.
- Slot-aware metadata prefilter and typed-slot no-fallback scoring.
- Streamlit operational layer for upload, rebuild, reset, pagination, and search.
- `.env`/OpenAI-compatible parser diagnostics.
- Regression-test coverage for the known assignment failure modes.
