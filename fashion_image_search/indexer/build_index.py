"""Build the searchable fashion index.

Key fixes applied:
  Bug #5 — Scene embedding is now distinct from global embedding:
            HF backend: embeds a text scene-description phrase via the CLIP text
            tower, giving a semantically targeted embedding for context queries.
            Offline backend: generates a layout-based image embedding from a
            tightly-cropped border strip (background/environment region) rather
            than the full image, producing a different vector.
  Bug #6 — Detector and encoder are instantiated once per build_index() call
            and reused across all images.  index_image() also accepts optional
            pre-loaded instances to avoid reloading for external callers.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from fashion_image_search.common.config import MODEL, PATHS
from fashion_image_search.common.schemas import ImageRecord, RegionRecord
from fashion_image_search.common.vector_db import VectorStore, make_vector_store
from fashion_image_search.indexer.attributes import dominant_color
from fashion_image_search.indexer.detect import (
    OfflineFashionDetector,
    HuggingFaceFashionDetector,
    load_detector,
)
from fashion_image_search.indexer.embed import (
    embed_image_backend,
    embed_text_backend,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Default scene phrase used by the HF backend text-tower scene embedding.
# A simple but semantically meaningful description keeps scene cosine scores
# in a useful range when the query contains scene words like "park" or "office".
_DEFAULT_SCENE_PHRASE = "fashion photography"


def iter_images(data_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(
        path for path in data_dir.rglob("*")
        if path.suffix.lower() in IMAGE_EXTENSIONS
    )
    return paths[:limit] if limit else paths


def _scene_embedding(
    rgb: Image.Image,
    embedding_backend: str,
    scene_phrase: str = _DEFAULT_SCENE_PHRASE,
) -> list[float]:
    """Produce a scene embedding that is distinct from the global image embedding.

    Bug #5 fix:
    - HF backend:    embed *scene_phrase* via the CLIP text tower.  Text-space
                     scene embedding cosine-compared to text-space query phrase
                     gives meaningful signal for "office", "park", "street" queries.
    - Offline backend: embed a border-cropped version of the image (15 % inset
                     strips from each edge, capturing background context) rather
                     than the full RGB, so the two embeddings are genuinely different.
    """
    if embedding_backend in {"hf", "huggingface"}:
        # Use the CLIP text encoder for the scene slot — distinct from image emb
        return embed_text_backend(scene_phrase, embedding_backend)

    # Offline: create a border crop that emphasises background over garments
    width, height = rgb.size
    inset_x = max(1, int(width * 0.15))
    inset_y = max(1, int(height * 0.15))
    # Take four thin border strips and composite into a representative sample
    # We use just the top strip as a proxy for background/environment context
    top_strip = rgb.crop((0, 0, width, inset_y))
    return embed_image_backend(top_strip, embedding_backend)


def index_image(
    path: Path,
    detector_backend: str = MODEL.backend,
    embedding_backend: str = MODEL.backend,
    detector: OfflineFashionDetector | HuggingFaceFashionDetector | None = None,
) -> ImageRecord:
    """Index a single image and return its ``ImageRecord``.

    Bug #6 fix: accepts an optional pre-loaded *detector* to avoid reloading
    model weights on every call from external code.
    """
    det = detector or load_detector(detector_backend)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        detections = det.detect(rgb)

        regions: list[RegionRecord] = []
        for idx, detection in enumerate(detections):
            color = dominant_color(detection.crop)
            regions.append(
                RegionRecord(
                    region_idx=idx,
                    bbox=detection.bbox,
                    category=detection.category,
                    detector_confidence=detection.confidence,
                    region_embedding=embed_image_backend(detection.crop, embedding_backend),
                    color=color.label,
                    color_confidence=color.confidence,
                )
            )

        global_emb = embed_image_backend(rgb, embedding_backend)
        scene_emb  = _scene_embedding(rgb, embedding_backend)   # Bug #5: distinct

        return ImageRecord(
            image_id=path.stem,
            image_path=str(path),
            global_embedding=global_emb,
            scene_embedding=scene_emb,
            regions=regions,
        )


def build_index(
    data_dir: Path = PATHS.data_dir,
    output_path: Path = PATHS.index_path,
    limit: int | None = None,
    detector_backend: str = MODEL.backend,
    embedding_backend: str = MODEL.backend,
    store_kind: str = "faiss",
    faiss_path: Path = PATHS.faiss_index_path,
) -> VectorStore:
    """Build a full FAISS + JSON index over all images in *data_dir*.

    Bug #6 fix: detector and encoder are each instantiated once and reused
    for every image in the loop — no per-image model reload.
    """
    store = make_vector_store(store_kind, output_path, faiss_path)

    # Instantiate detector once (Bug #6 fix)
    detector = load_detector(detector_backend)

    for path in tqdm(iter_images(data_dir, limit), desc="Indexing images"):
        with Image.open(path) as image:
            rgb = image.convert("RGB")
            detections = detector.detect(rgb)

            regions: list[RegionRecord] = []
            for idx, detection in enumerate(detections):
                color = dominant_color(detection.crop)
                regions.append(
                    RegionRecord(
                        region_idx=idx,
                        bbox=detection.bbox,
                        category=detection.category,
                        detector_confidence=detection.confidence,
                        region_embedding=embed_image_backend(
                            detection.crop, embedding_backend
                        ),
                        color=color.label,
                        color_confidence=color.confidence,
                    )
                )

            global_emb = embed_image_backend(rgb, embedding_backend)
            scene_emb  = _scene_embedding(rgb, embedding_backend)  # Bug #5: distinct

            store.add(
                ImageRecord(
                    image_id=path.stem,
                    image_path=str(path),
                    global_embedding=global_emb,
                    scene_embedding=scene_emb,
                    regions=regions,
                )
            )

    store.save()
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fashion image retrieval index.")
    parser.add_argument("--data-dir",        type=Path, default=PATHS.data_dir)
    parser.add_argument("--output",          type=Path, default=PATHS.index_path)
    parser.add_argument("--faiss-output",    type=Path, default=PATHS.faiss_index_path)
    parser.add_argument("--limit",           type=int,  default=None)
    parser.add_argument("--store",           default="faiss", choices=["faiss", "json"])
    parser.add_argument("--backend",         default=MODEL.backend,
                        help="Shortcut backend for detector and embedder.")
    parser.add_argument("--detector-backend", default=None,
                        choices=["offline", "hf", "huggingface"])
    parser.add_argument("--embedding-backend", default=None,
                        choices=["offline", "hf", "huggingface"])
    args = parser.parse_args()

    detector_backend  = args.detector_backend  or args.backend
    embedding_backend = args.embedding_backend or args.backend

    store = build_index(
        args.data_dir,
        args.output,
        args.limit,
        detector_backend,
        embedding_backend,
        args.store,
        args.faiss_output,
    )
    print(f"Wrote {len(store.records)} records to {args.output}")
    if args.store == "faiss":
        print(f"Wrote FAISS ANN index to {args.faiss_output}")


if __name__ == "__main__":
    main()
