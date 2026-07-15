"""Offline pipeline tests that require no model downloads."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from fashion_image_search.common.schemas import GarmentSlot, ImageRecord, RegionRecord
from fashion_image_search.common.vector_db import JsonVectorStore
from fashion_image_search.indexer.build_index import build_index
from fashion_image_search.indexer.detect import _normalize_fashionpedia_label
from fashion_image_search.retriever.parse_query import parse_query
from fashion_image_search.retriever.search import search, _record_matches_slots


class OfflinePipelineTest(unittest.TestCase):
    def test_query_parser_keeps_slot_order_and_binding(self) -> None:
        parsed = parse_query("A red tie and a white shirt in a formal setting.")
        self.assertEqual([(slot.color, slot.garment_type) for slot in parsed.garment_slots], [
            ("red", "tie"),
            ("white", "shirt"),
        ])
        self.assertIn("formal", parsed.scene_phrase or "")

    def test_jacket_alias_resolves_to_blazer_not_raincoat(self) -> None:
        # Bug #3 fix verification: "jacket" should resolve to "blazer", not "raincoat"
        parsed = parse_query("a black jacket")
        self.assertEqual(len(parsed.garment_slots), 1)
        self.assertEqual(parsed.garment_slots[0].garment_type, "blazer")
        self.assertEqual(parsed.garment_slots[0].color, "black")

    def test_skirt_slot_parsed(self) -> None:
        # Bug #10 fix verification: "skirt" should be successfully parsed
        parsed = parse_query("a blue skirt")
        self.assertEqual(len(parsed.garment_slots), 1)
        self.assertEqual(parsed.garment_slots[0].garment_type, "skirt")
        self.assertEqual(parsed.garment_slots[0].color, "blue")

    def test_category_normalization_offline(self) -> None:
        # Bug #1 & #2 normalization mapping verification
        self.assertEqual(_normalize_fashionpedia_label("shirt, blouse"), "shirt")
        self.assertEqual(_normalize_fashionpedia_label("collar"), None)
        self.assertEqual(_normalize_fashionpedia_label("top, t-shirt, sweatshirt"), "shirt")
        self.assertEqual(_normalize_fashionpedia_label("pants"), "pants")
        self.assertEqual(_normalize_fashionpedia_label("unknown-custom-label"), "unknown-custom-label")

    def test_multi_slot_filter_rejects_partial_match(self) -> None:
        # Bug #4 fix verification: multi-slot queries require all slots to fire (AND mode)
        # Create a record with ONLY a red pants region
        record = ImageRecord(
            image_id="test_img",
            image_path="test.jpg",
            global_embedding=[0.0] * 64,
            scene_embedding=[0.0] * 64,
            regions=[
                RegionRecord(
                    region_idx=0,
                    bbox=(0, 0, 10, 10),
                    category="pants",
                    detector_confidence=0.9,
                    region_embedding=[0.0] * 64,
                    color="red",
                    color_confidence=1.0,
                )
            ]
        )

        # Query: red tie and white shirt (2 slots)
        slots = [
            GarmentSlot(garment_type="tie", color="red", phrase="red tie"),
            GarmentSlot(garment_type="shirt", color="white", phrase="white shirt"),
        ]

        # In AND-mode, since there's no shirt/tie, it should return False
        self.assertFalse(_record_matches_slots(record, slots))

        # If we query just for a single slot "red tie" (OR-mode / single slot),
        # color_match ("red" == "red") makes it match
        single_slot_matching = [
            GarmentSlot(garment_type="tie", color="red", phrase="red tie"),
        ]
        self.assertTrue(_record_matches_slots(record, single_slot_matching))

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
