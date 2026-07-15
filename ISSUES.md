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

## P2 - Nice To Have

8. **Index build resume/cache**
   - Store per-image intermediate records so interrupted HF indexing can resume.

9. **Batch embedding**
   - Current indexing embeds one crop/image at a time.
   - Batch region and image embeddings for faster GPU throughput.

10. **Optional UI index builder**
    - Streamlit currently searches existing indexes.
    - A controlled index-building page could be added, but CLI is safer for long HF jobs.
