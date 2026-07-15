"""Vector stores for global ANN recall plus metadata payloads.

`JsonVectorStore` is a dependency-light exact-search fallback. `FaissVectorStore`
is the assignment-facing store: FAISS handles Stage-1 vector recall, while JSON
metadata keeps image paths, regions, colors, and payloads available for Stage-2
compositional reranking.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Protocol

from fashion_image_search.common.schemas import ImageRecord, Vector


def cosine_similarity(left: Vector, right: Vector) -> float:
    numerator = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return numerator / (left_norm * right_norm)


class VectorStore(Protocol):
    records: list[ImageRecord]

    def add(self, record: ImageRecord) -> None:
        ...

    def save(self) -> None:
        ...

    def load(self) -> "VectorStore":
        ...

    def search_global(self, query_embedding: Vector, top_k: int) -> list[tuple[ImageRecord, float]]:
        ...


class JsonVectorStore:
    def __init__(self, path: Path):
        self.path = path
        self.records: list[ImageRecord] = []

    def add(self, record: ImageRecord) -> None:
        self.records.append(record)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in self.records], handle, indent=2)

    def load(self) -> "JsonVectorStore":
        with self.path.open("r", encoding="utf-8") as handle:
            self.records = [ImageRecord.from_dict(item) for item in json.load(handle)]
        return self

    def search_global(self, query_embedding: Vector, top_k: int) -> list[tuple[ImageRecord, float]]:
        scored = [
            (record, cosine_similarity(query_embedding, record.global_embedding))
            for record in self.records
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class FaissVectorStore:
    """FAISS HNSW index over whole-image global embeddings.

    The image records remain in a sidecar JSON file because FAISS stores vectors,
    not rich metadata. Embeddings are normalized and searched with inner product,
    which is cosine similarity for normalized vectors.
    """

    def __init__(self, metadata_path: Path, faiss_path: Path | None = None):
        self.metadata_path = metadata_path
        self.faiss_path = faiss_path or metadata_path.with_suffix(".faiss")
        self.records: list[ImageRecord] = []
        self.index = None

    def add(self, record: ImageRecord) -> None:
        self.records.append(record)

    def save(self) -> None:
        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "FAISS store needs faiss-cpu and numpy. Install with: "
                ".\\.venv\\Scripts\\python -m pip install faiss-cpu"
            ) from exc

        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        vectors = np.array([record.global_embedding for record in self.records], dtype="float32")
        if len(vectors) == 0:
            raise ValueError("Cannot save an empty FAISS index.")
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 80
        index.add(vectors)
        faiss.write_index(index, str(self.faiss_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump([record.to_dict() for record in self.records], handle, indent=2)
        self.index = index

    def load(self) -> "FaissVectorStore":
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError(
                "FAISS store needs faiss-cpu and numpy. Install with: "
                ".\\.venv\\Scripts\\python -m pip install faiss-cpu"
            ) from exc

        with self.metadata_path.open("r", encoding="utf-8") as handle:
            self.records = [ImageRecord.from_dict(item) for item in json.load(handle)]
        self.index = faiss.read_index(str(self.faiss_path))
        return self

    def search_global(self, query_embedding: Vector, top_k: int) -> list[tuple[ImageRecord, float]]:
        if self.index is None:
            self.load()
        import faiss
        import numpy as np

        query = np.array([query_embedding], dtype="float32")
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, min(top_k, len(self.records)))
        results: list[tuple[ImageRecord, float]] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            results.append((self.records[int(index)], float(score)))
        return results


def make_vector_store(kind: str, metadata_path: Path, faiss_path: Path | None = None) -> VectorStore:
    if kind == "json":
        return JsonVectorStore(metadata_path)
    if kind == "faiss":
        return FaissVectorStore(metadata_path, faiss_path)
    raise ValueError(f"Unknown vector store kind: {kind}")
