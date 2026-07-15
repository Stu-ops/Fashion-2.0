# Fashion Image Search

Region-aware multimodal retrieval for the Glance ML internship assignment.

The implementation follows the supplied architecture write-up: index whole-image
embeddings, garment-region embeddings, and simple visual attributes, then answer
natural-language queries with global recall followed by compositional region
reranking.

## Why This Is Better Than Plain CLIP

Plain CLIP stores one pooled vector per image. That is fast, but it can confuse
queries like `red tie and white shirt` because color and garment identity are
mixed in one global representation. This code keeps garment regions separate, so
each parsed query slot is matched against the best region independently.

## Layout

```text
fashion_image_search/
  indexer/
    detect.py        # YOLOS-Fashionpedia detector interface
    embed.py         # Marqo FashionSigLIP image/text encoder interface
    attributes.py    # dominant color extraction
    build_index.py   # image folder -> FAISS ANN index + JSON metadata
  retriever/
    parse_query.py   # natural language -> garment, scene, style slots
    search.py        # global recall + region-aware rerank
    evaluate.py      # assignment evaluation queries
  common/
    config.py
    schemas.py
    vector_db.py
```

## Quick Start

Build a small FAISS index first:

```bash
python -m fashion_image_search.indexer.build_index --data-dir val_test2020/test --limit 100 --store faiss
```

Search it:

```bash
python -m fashion_image_search.retriever.search "A red tie and a white shirt in a formal setting." --explain
```

Run the five assignment queries:

```bash
python -m fashion_image_search.retriever.evaluate
```

Launch the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

Run offline tests:

```bash
python -m unittest tests.test_pipeline_offline
```

Use an OpenAI-compatible LLM parser for stronger natural-language slot extraction:

```bash
set FASHION_SEARCH_LLM_BASE_URL=https://api.openai.com/v1
set FASHION_SEARCH_LLM_API_KEY=your_key_here
set FASHION_SEARCH_LLM_MODEL=gpt-4o-mini
python -m fashion_image_search.retriever.search "A red tie and a white shirt in a formal setting." --parser openai --explain
```

For opencode or another OpenAI-compatible provider, point
`FASHION_SEARCH_LLM_BASE_URL` at that provider's `/v1` endpoint and set the
provider model name in `FASHION_SEARCH_LLM_MODEL`. If the API call fails, the
parser falls back to the deterministic rule parser so retrieval still runs.

## Backend Notes

This repository has two execution backends:

- `offline`: deterministic Pillow-based features for tests and no-network runs.
- `hf`: real Hugging Face downloads for `valentinafevu/yolos-fashionpedia` and
  `Marqo/marqo-fashionSigLIP`.

The real backend uses only those two planned models. `open_clip_torch` appears
in `requirements.txt` because Marqo's FashionSigLIP repository uses OpenCLIP as
its runtime loader; it is not a substitute encoder.

Create the local virtual environment:

```bash
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
```

Pre-download the Hugging Face models:

```bash
.\.venv\Scripts\python -m fashion_image_search.download_models
```

Build a real-model index:

```bash
.\.venv\Scripts\python -m fashion_image_search.indexer.build_index --backend hf --limit 100 --store faiss
```

Build the sampled 1600-image index:

```bash
.\.venv\Scripts\python -m fashion_image_search.indexer.build_index --data-dir val_test2020_sample_1600\test --backend hf --store faiss
```

Search that real-model index:

```bash
.\.venv\Scripts\python -m fashion_image_search.retriever.search "A red tie and a white shirt in a formal setting." --backend hf --explain
```

For production-quality retrieval, this repository now uses:

- Detector: `valentinafevu/yolos-fashionpedia`
- Encoder: `Marqo/marqo-fashionSigLIP`
- Stage-1 ANN recall: FAISS HNSW over global embeddings
- Stage-2 precision: per-garment region rerank + scene score + attribute bonus
