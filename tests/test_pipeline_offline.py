"""Offline pipeline tests that require no model downloads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fashion_image_search.common.vector_db import JsonVectorStore
from fashion_image_search.indexer.build_index import build_index
from fashion_image_search.retriever.parse_query import parse_query
from fashion_image_search.retriever.search import search


class OfflinePipelineTest(unittest.TestCase):
    def test_query_parser_keeps_slot_order_and_binding(self) -> None:
        parsed = parse_query("A red tie and a white shirt in a formal setting.")
        self.assertEqual([(slot.color, slot.garment_type) for slot in parsed.garment_slots], [
            ("red", "tie"),
            ("white", "shirt"),
        ])
        self.assertIn("formal", parsed.scene_phrase or "")

    def test_build_index_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "images"
            data_dir.mkdir()
            Image.new("RGB", (80, 120), (220, 40, 40)).save(data_dir / "red.jpg")
            Image.new("RGB", (80, 120), (40, 90, 220)).save(data_dir / "blue.jpg")
            index_path = Path(tmp) / "index.json"
            store = build_index(data_dir=data_dir, output_path=index_path, limit=None, store_kind="json")
            self.assertEqual(len(store.records), 2)
            loaded = JsonVectorStore(index_path).load()
            self.assertEqual(len(loaded.records), 2)
            results = search(index_path, "a red shirt", top_k=1, store_kind="json")
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].image_path.endswith("red.jpg"))

    def test_faiss_store_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "images"
            data_dir.mkdir()
            Image.new("RGB", (80, 120), (220, 40, 40)).save(data_dir / "red.jpg")
            Image.new("RGB", (80, 120), (40, 90, 220)).save(data_dir / "blue.jpg")
            index_path = Path(tmp) / "records.json"
            faiss_path = Path(tmp) / "global.faiss"
            build_index(
                data_dir=data_dir,
                output_path=index_path,
                limit=None,
                store_kind="faiss",
                faiss_path=faiss_path,
            )
            self.assertTrue(index_path.exists())
            self.assertTrue(faiss_path.exists())
            results = search(index_path, "a red shirt", top_k=1, store_kind="faiss", faiss_path=faiss_path)
            self.assertEqual(len(results), 1)


if __name__ == "__main__":
    unittest.main()
