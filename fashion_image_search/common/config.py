"""Central configuration for the fashion retrieval pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _strip_inline_comment(value: str) -> str:
    """Strip shell-style inline comments outside quotes."""
    quote: str | None = None
    escaped = False
    for index, char in enumerate(value):
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char in {"'", '"'}:
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
            continue
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value


def load_dotenv(path: Path = ROOT / ".env", *, override: bool = False) -> None:
    """Load simple KEY=VALUE entries from .env.

    With ``override=False`` this preserves existing shell environment variables;
    with ``override=True`` it refreshes them from the file.

    Idempotent (uses ``setdefault``) — safe to call multiple times.  Called
    automatically at module import, but can also be invoked later from query
    parsers or the Streamlit UI to pick up newly-created .env files without
    restarting the process.
    """
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_inline_comment(value.strip()).strip('"').strip("'")
        if key:
            if override:
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)


load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    detector_name: str = "valentinafevu/yolos-fashionpedia"
    encoder_name: str = "Marqo/marqo-fashionSigLIP"
    detector_threshold: float = 0.50
    max_regions: int = 6
    embedding_dim: int = 64
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
    data_dir: Path = ROOT / "Dataset"
    index_path: Path = ROOT / "artifacts" / "fashion_index.json"
    faiss_index_path: Path = ROOT / "artifacts" / "fashion_global.faiss"


MODEL = ModelConfig()
SEARCH = SearchConfig()
PATHS = Paths()
