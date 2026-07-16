"""Garment detection adapters.

The offline detector produces stable, layout-based regions so the full pipeline
can be tested without model downloads.  The public interface mirrors the real
YOLOS path: each detection has bbox, category (normalized canonical string),
confidence, and crop.

Key fix (Bugs #1, #2):
    ``HuggingFaceFashionDetector`` now applies ``FASHIONPEDIA_TO_CANONICAL``
    at detection time.  Detections whose label maps to ``None`` (part/decoration
    labels like "sleeve", "collar", "lapel") are filtered out *before* the
    ``max_regions`` cap, so real garment detections are never displaced by
    part-level noise.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import logging
from pathlib import Path
from typing import Any

from PIL import Image

from fashion_image_search.common.config import MODEL
from fashion_image_search.common.schemas import BBox


logger = logging.getLogger(__name__)


# ── Fashionpedia → canonical category mapping (Bug #1 fix) ───────────────────
#
# The YOLOS-Fashionpedia model (valentinafevu/yolos-fashionpedia) outputs the
# raw Fashionpedia label strings — comma-separated compound names — as its
# id2label values.  These never match the simple canonical strings used in the
# query parser (GARMENTS dict in parse_query.py) without this mapping.
#
# Values of None = part/decoration label → filtered out entirely (Bug #2 fix).
# Values of str  = canonical garment type that matches GARMENTS keys.

FASHIONPEDIA_TO_CANONICAL: dict[str, str | None] = {
    # ── Garment-level labels ─────────────────────────────────────────────────
    "shirt, blouse":                         "shirt",
    "top, t-shirt, sweatshirt":              "shirt",
    "sweater":                               "shirt",
    "cardigan":                              "shirt",
    "jacket":                                "jacket",
    "vest":                                  "vest",
    "pants":                                 "pants",
    "shorts":                                "shorts",
    "skirt":                                 "skirt",
    "coat":                                  "coat",
    "dress":                                 "dress",
    "jumpsuit":                              "dress",
    "cape":                                  "coat",
    "glasses":                               "glasses",
    "hat":                                   "hat",
    "headband, head covering, hair accessory": "hat",
    "tie":                                   "tie",
    "glove":                                 "glove",
    "watch":                                 None,      # accessory — not queried
    "belt":                                  None,      # part/accessory — exclude
    "leg warmer":                            "pants",
    "tights, stockings":                     "pants",
    "sock":                                  None,
    "shoe":                                  "shoe",
    "bag, wallet":                           None,
    "scarf":                                 "scarf",
    "umbrella":                              None,
    # ── Part / decoration labels → None (filter out, Bug #2) ────────────────
    "collar":                                None,
    "lapel":                                 None,
    "epaulette":                             None,
    "sleeve":                                None,
    "pocket":                                None,
    "neckline":                              None,
    "buckle":                                None,
    "zipper":                                None,
    "applique":                              None,
    "bead":                                  None,
    "bow":                                   None,
    "flower":                                None,
    "fringe":                                None,
    "ribbon":                                None,
    "rivet":                                 None,
    "ruffle":                                None,
    "sequin":                                None,
    "tassel":                                None,
}


def _normalize_fashionpedia_label(raw_label: str) -> str | None:
    """Map a raw Fashionpedia label string to a canonical type, or None to discard.

    Tries the exact label first, then a lower-stripped fallback.  Unknown labels
    that are not in the mapping table are kept as-is (graceful forward-compat).
    """
    label = raw_label.strip().lower()
    if label in FASHIONPEDIA_TO_CANONICAL:
        return FASHIONPEDIA_TO_CANONICAL[label]
    # Try partial match for any future label variants not yet in the table
    for key, canonical in FASHIONPEDIA_TO_CANONICAL.items():
        if label == key:
            return canonical
    # Unknown label — keep it rather than silently dropping
    return label


@dataclass
class Detection:
    bbox: BBox
    category: str          # always a normalized canonical string post-construction
    confidence: float
    crop: Image.Image


# ── Offline detector (no model downloads) ────────────────────────────────────

class OfflineFashionDetector:
    """Dependency-free detector fallback with fashion-shaped region proposals.

    Categories match the canonical vocabulary used by parse_query.py so that
    offline slot matching works end-to-end without the HF models.
    """

    def __init__(self, max_regions: int = MODEL.max_regions):
        self.max_regions = max_regions

    def detect(self, image: Image.Image) -> list[Detection]:
        width, height = image.size
        proposals: list[tuple[BBox, str, float]] = [
            ((int(width * 0.18), int(height * 0.10),
              int(width * 0.82), int(height * 0.62)), "shirt",    0.55),
            ((int(width * 0.36), int(height * 0.10),
              int(width * 0.58), int(height * 0.58)), "tie",      0.35),
            ((int(width * 0.12), int(height * 0.04),
              int(width * 0.88), int(height * 0.78)), "coat",     0.45),
            ((int(width * 0.18), int(height * 0.52),
              int(width * 0.82), int(height * 0.96)), "pants",    0.40),
        ]
        detections: list[Detection] = []
        for bbox, category, confidence in proposals[: self.max_regions]:
            left, top, right, bottom = bbox
            if right > left and bottom > top:
                detections.append(
                    Detection(bbox, category, confidence, image.crop(bbox))
                )
        return detections


# ── Hugging Face detector (real YOLOS-Fashionpedia) ──────────────────────────

class HuggingFaceFashionDetector:
    """YOLOS-Fashionpedia detector with built-in category normalization.

    Changes vs original (Bugs #1, #2):
    - Applies ``FASHIONPEDIA_TO_CANONICAL`` to every detection at inference time.
    - Discards part/decoration labels (mapped to None) before applying max_regions.
    - Re-sorts surviving garment detections by confidence before capping to top-N.
    """

    def __init__(
        self,
        model_name: str = MODEL.detector_name,
        threshold: float = MODEL.detector_threshold,
        max_regions: int = MODEL.max_regions,
        device: str = MODEL.device,
    ):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face detector backend needs torch and transformers. "
                "Install them with: .\\.venv\\Scripts\\python -m pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForObjectDetection.from_pretrained(model_name)
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "FASHION_SEARCH_DEVICE=cuda was requested, but this PyTorch install "
                "does not see CUDA. Install a CUDA-enabled torch build in .venv."
            )
        self.device = device
        logger.info(
            "Loading YOLOS detector requested_device=%s selected_device=%s "
            "torch=%s cuda_available=%s cuda_version=%s",
            MODEL.device,
            self.device,
            torch.__version__,
            torch.cuda.is_available(),
            torch.version.cuda,
        )
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.max_regions = max_regions

    def detect(self, image: Image.Image) -> list[Detection]:
        rgb = image.convert("RGB")
        inputs = self.processor(images=rgb, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.no_grad():
            outputs = self.model(**inputs)
        target_sizes = self.torch.tensor([rgb.size[::-1]], device=self.device)
        post_process = getattr(self.processor, "post_process_object_detection")
        kwargs: dict[str, Any] = {"target_sizes": target_sizes}
        if "threshold" in inspect.signature(post_process).parameters:
            kwargs["threshold"] = self.threshold
        results = post_process(outputs, **kwargs)[0]

        width, height = rgb.size
        detections: list[Detection] = []

        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            confidence = float(score.detach().cpu())
            if confidence < self.threshold:
                continue

            raw_label = self.model.config.id2label.get(int(label), str(int(label)))
            # Bug #1 + #2 fix: normalize and filter part labels
            canonical = _normalize_fashionpedia_label(raw_label)
            if canonical is None:
                continue  # part/decoration label — discard

            left, top, right, bottom = [
                int(round(v)) for v in box.detach().cpu().tolist()
            ]
            left, top   = max(0, left),   max(0, top)
            right, bottom = min(width, right), min(height, bottom)
            if right <= left or bottom <= top:
                continue

            bbox = (left, top, right, bottom)
            detections.append(
                Detection(bbox, canonical, confidence, rgb.crop(bbox))
            )

        # Sort by confidence descending AFTER filtering part labels, then cap to max_regions
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: self.max_regions]


# ── Factory helpers ───────────────────────────────────────────────────────────

def load_detector(
    backend: str = MODEL.backend,
) -> OfflineFashionDetector | HuggingFaceFashionDetector:
    if backend == "offline":
        return OfflineFashionDetector()
    if backend in {"hf", "huggingface"}:
        return HuggingFaceFashionDetector()
    raise ValueError(f"Unknown detector backend: {backend!r}")


def detect_image(
    path: Path,
    backend: str = MODEL.backend,
    detector: OfflineFashionDetector | HuggingFaceFashionDetector | None = None,
) -> list[Detection]:
    """Detect garments in *path*.

    Accepts an optional pre-instantiated *detector* to avoid reloading model
    weights for every image (Bug #6 fix when called from external code).
    """
    det = detector or load_detector(backend)
    with Image.open(path) as image:
        return det.detect(image.convert("RGB"))
