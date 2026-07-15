"""Central configuration for the fashion retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ModelConfig:
    detector_name: str = "valentinafevu/yolos-fashionpedia"
    encoder_name: str = "Marqo/marqo-fashionSigLIP"
    detector_threshold: float = 0.50
    max_regions: int = 6
    embedding_dim: int = 64   # offline mode only; HF mode uses 768 (FashionSigLIP)
    backend: str = os.getenv("FASHION_SEARCH_BACKEND", "offline")
    device: str = os.getenv("FASHION_SEARCH_DEVICE", "auto")


@dataclass(frozen=True)
class SearchConfig:
    stage1_k: int = 200
    default_top_k: int = 5
    slot_weight: float = 0.45
    scene_weight: float = 0.20
    global_weight: float = 0.25
    attribute_bonus_weight: float = 0.10


@dataclass(frozen=True)
class Paths:
    # Bug #9 fix: unified data_dir so CLI and Streamlit both use Dataset/ by default.
    # Override via --data-dir when running HF indexing on val_test2020_sample_1600/test.
    data_dir: Path = ROOT / "Dataset"
    index_path: Path = ROOT / "artifacts" / "fashion_index.json"
    faiss_index_path: Path = ROOT / "artifacts" / "fashion_global.faiss"


MODEL = ModelConfig()
SEARCH = SearchConfig()
PATHS = Paths()
