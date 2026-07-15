# Future Work / Issues

## P0 - Must Fix Before Final Submission

1. **Run full HF indexing on sampled dataset**
   - Build FAISS index for `val_test2020_sample_1600/test`.
   - Command:
     ```powershell
     .\.venv\Scripts\python -m fashion_image_search.indexer.build_index --data-dir val_test2020_sample_1600\test --backend hf --store faiss
     ```
   - This is required so Streamlit search uses real YOLOS + FashionSigLIP embeddings.

2. **Add category normalization for detector labels**
   - YOLOS/Fashionpedia can emit fine-grained labels like `sleeve`, `neckline`, `collar`.
   - Retrieval should map labels into query-level classes like `shirt`, `tie`, `coat`, `pants`, `dress`.
   - This directly affects compositional queries such as `red tie and white shirt`.

3. **Save evaluation outputs**
   - Add JSON/CSV output from `evaluate.py` with query, rank, score, image path, and slot breakdown.
   - Useful for the PDF/write-up and reproducible grading evidence.

## P1 - Important Quality Improvements

4. **Improve metadata filtering**
   - Current filtering is applied after FAISS recall.
   - Better approach: maintain inverted metadata maps for color/category filters before reranking.

5. **Expose parsed query in CLI and Streamlit**
   - Print/display structured slots:
     `garments=[{color,type}], scene, style`.
   - This makes the architecture easier to debug and defend.

6. **Pattern attribute extraction**
   - Current attribute branch supports category and color only.
   - Add simple pattern labels: `solid`, `striped`, `checked`, `floral`, `graphic`.

7. **Dedicated scene/context branch**
   - Current scene embedding reuses full-image FashionSigLIP.
   - A lightweight scene classifier or better scene prompt strategy would improve office/park/street queries.

8. **Resolver mismatch between CLI defaults and Streamlit paths**
   - `config.py` → `PATHS.data_dir` defaults to `val_test2020/test`.
   - `streamlit_app.py` hardcodes `Dataset/` instead.
   - Any user running CLI commands without arguments gets a different dataset than the UI.
   - Fix: make both reference the same configurable path, or make Streamlit use `PATHS.data_dir` consistently.

9. **Unused imports in Streamlit app**
   - `streamlit_app.py` imports `ImageRecord` but never uses it.
   - Remove to avoid confusion and circular-dependency risk.

10. **Metadata filter logic is too permissive for multi-slot queries**
    - `_record_matches_any_slot()` uses OR semantics: a single matching colour OR category passes the filter.
    - For `red tie + white shirt`, an image with only red pants should not pass the filter.
    - Fix: add AND mode for multi-slot queries, or make matching stricter.

11. **Deprecated Pillow API in attributes.py**
    - `Image.Image.getdata()` triggers a deprecation warning and will break in Pillow 14.
    - Replace with `get_flattened_data()` or iterate via `np.asarray(image)`.

12. **No reproducibility lockfile**
    - `requirements.txt` has version floor specifiers only (`>=`), not pinned hashes.
    - A different pip resolver on another machine can install incompatible transitive versions.
    - Fix: generate `requirements.lock` with `pip freeze` or use a lockfile tool.

## P2 - Nice To Have

13. **Index build resume/cache**
    - Store per-image intermediate records so interrupted HF indexing can resume.

14. **Batch embedding**
    - Current indexing embeds one crop/image at a time.
    - Batch region and image embeddings for faster GPU throughput.

15. **Optional UI index builder**
    - Streamlit currently searches existing indexes.
    - A controlled index-building page could be added, but CLI is safer for long HF jobs.

16. **Expose HF parser usage in Streamlit UI**
    - Currently parser dropdown is present but no docs on how to use `opencode` / OpenAI-compatible endpoints from the UI itself.
    - Add a collapsible help panel with example env vars.

17. **Detect label coverage gap**
    - `GARMENTS` dict in `parse_query.py` has 8 canonical types.
    - Fashionpedia detects 46 fine-grained classes.
    - Many detector outputs never match a query slot because they are not in `GARMENTS`.
    - Fix: extend `GARMENTS` or add a mapping table from all 46 Fashionpedia labels → canonical types.