"""Garment detection adapters.

The offline detector produces stable, layout-based regions so the full pipeline
can be tested without model downloads. The public interface mirrors the real
YOLOS path: each detection has bbox, category, confidence, and crop.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any

from PIL import Image

from fashion_image_search.common.config import MODEL
from fashion_image_search.common.schemas import BBox


@dataclass
class Detection:
    bbox: BBox
    category: str
    confidence: float
    crop: Image.Image


class OfflineFashionDetector:
    """Dependency-free detector fallback with fashion-shaped region proposals."""

    def __init__(self, max_regions: int = MODEL.max_regions):
        self.max_regions = max_regions

    def detect(self, image: Image.Image) -> list[Detection]:
        width, height = image.size
        proposals: list[tuple[BBox, str, float]] = [
            ((int(width * 0.18), int(height * 0.10), int(width * 0.82), int(height * 0.62)), "shirt", 0.55),
            ((int(width * 0.36), int(height * 0.10), int(width * 0.58), int(height * 0.58)), "tie", 0.35),
            ((int(width * 0.12), int(height * 0.04), int(width * 0.88), int(height * 0.78)), "coat", 0.45),
            ((int(width * 0.18), int(height * 0.52), int(width * 0.82), int(height * 0.96)), "pants", 0.40),
        ]
        detections = []
        for bbox, category, confidence in proposals[: self.max_regions]:
            left, top, right, bottom = bbox
            if right > left and bottom > top:
                detections.append(Detection(bbox, category, confidence, image.crop(bbox)))
        return detections


class HuggingFaceFashionDetector:
    """YOLOS/Fashionpedia detector loaded directly from Hugging Face Hub."""

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
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
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

        detections: list[Detection] = []
        width, height = rgb.size
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            confidence = float(score.detach().cpu())
            if confidence < self.threshold:
                continue
            left, top, right, bottom = [int(round(value)) for value in box.detach().cpu().tolist()]
            left, top = max(0, left), max(0, top)
            right, bottom = min(width, right), min(height, bottom)
            if right <= left or bottom <= top:
                continue
            category = self.model.config.id2label.get(int(label), str(int(label)))
            bbox = (left, top, right, bottom)
            detections.append(Detection(bbox, category.lower(), confidence, rgb.crop(bbox)))

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[: self.max_regions]


def load_detector(backend: str = MODEL.backend) -> OfflineFashionDetector | HuggingFaceFashionDetector:
    if backend == "offline":
        return OfflineFashionDetector()
    if backend in {"hf", "huggingface"}:
        return HuggingFaceFashionDetector()
    raise ValueError(f"Unknown detector backend: {backend}")


def detect_image(path: Path, backend: str = MODEL.backend) -> list[Detection]:
    with Image.open(path) as image:
        return load_detector(backend).detect(image.convert("RGB"))
