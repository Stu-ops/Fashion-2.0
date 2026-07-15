"""Build the searchable fashion index."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from fashion_image_search.common.config import MODEL, PATHS
from fashion_image_search.common.schemas import ImageRecord, RegionRecord
from fashion_image_search.common.vector_db import VectorStore, make_vector_store
from fashion_image_search.indexer.attributes import dominant_color
from fashion_image_search.indexer.detect import load_detector
from fashion_image_search.indexer.embed import embed_image_backend


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(data_dir: Path, limit: int | None = None) -> list[Path]:
    paths = sorted(path for path in data_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    return paths[:limit] if limit else paths


def index_image(
    path: Path,
    detector_backend: str = MODEL.backend,
    embedding_backend: str = MODEL.backend,
) -> ImageRecord:
    detector = load_detector(detector_backend)
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
                    region_embedding=embed_image_backend(detection.crop, embedding_backend),
                    color=color.label,
                    color_confidence=color.confidence,
                )
            )
        return ImageRecord(
            image_id=path.stem,
            image_path=str(path),
            global_embedding=embed_image_backend(rgb, embedding_backend),
            scene_embedding=embed_image_backend(rgb, embedding_backend),
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
    store = make_vector_store(store_kind, output_path, faiss_path)
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
                        region_embedding=embed_image_backend(detection.crop, embedding_backend),
                        color=color.label,
                        color_confidence=color.confidence,
                    )
                )
            store.add(
                ImageRecord(
                    image_id=path.stem,
                    image_path=str(path),
                    global_embedding=embed_image_backend(rgb, embedding_backend),
                    scene_embedding=embed_image_backend(rgb, embedding_backend),
                    regions=regions,
                )
            )
    store.save()
    return store


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a fashion image retrieval index.")
    parser.add_argument("--data-dir", type=Path, default=PATHS.data_dir)
    parser.add_argument("--output", type=Path, default=PATHS.index_path)
    parser.add_argument("--faiss-output", type=Path, default=PATHS.faiss_index_path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--store", default="faiss", choices=["faiss", "json"])
    parser.add_argument("--backend", default=MODEL.backend, help="Shortcut backend for detector and embedder.")
    parser.add_argument("--detector-backend", default=None, choices=["offline", "hf", "huggingface"])
    parser.add_argument("--embedding-backend", default=None, choices=["offline", "hf", "huggingface"])
    args = parser.parse_args()
    detector_backend = args.detector_backend or args.backend
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
