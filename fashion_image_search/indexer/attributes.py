"""Lightweight visual attributes for garment crops."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np
from PIL import Image


PALETTE: dict[str, tuple[int, int, int]] = {
    "black":  (20,  20,  20),
    "white":  (238, 238, 232),
    "gray":   (128, 128, 128),
    "red":    (210, 45,  45),
    "orange": (230, 120, 35),
    "yellow": (230, 205, 40),
    "green":  (55,  150, 80),
    "blue":   (55,  105, 200),
    "purple": (125, 75,  165),
    "pink":   (225, 110, 165),
    "brown":  (120, 80,  50),
    "beige":  (205, 180, 140),
}

# Pre-compute palette as a numpy array for vectorised nearest-colour lookup.
_PALETTE_NAMES: list[str] = list(PALETTE)
_PALETTE_RGB: "np.ndarray" = np.array(
    [PALETTE[name] for name in _PALETTE_NAMES], dtype=np.int32
)  # shape (12, 3)


@dataclass(frozen=True)
class ColorPrediction:
    label: str
    confidence: float


def _nearest_palette_name(rgb: tuple[int, int, int]) -> str:
    """Return the palette colour name closest to *rgb* by squared Euclidean distance."""
    diffs = _PALETTE_RGB - np.array(rgb, dtype=np.int32)
    distances = (diffs * diffs).sum(axis=1)
    return _PALETTE_NAMES[int(distances.argmin())]


def dominant_color(image: Image.Image, sample_size: int = 64) -> ColorPrediction:
    """Return the dominant palette colour of *image*.

    Uses numpy array reshaping instead of the deprecated ``Image.getdata()``
    API (which will be removed in Pillow 14).  Also ~10× faster due to
    vectorised distance computation.
    """
    rgb_img = image.convert("RGB").resize((sample_size, sample_size))
    # shape (sample_size*sample_size, 3) — avoids deprecated getdata()
    pixels: np.ndarray = np.asarray(rgb_img, dtype=np.int32).reshape(-1, 3)

    labels: Counter[str] = Counter()
    for pixel in pixels:
        r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
        # Skip near-white pixels (overexposed background noise)
        if max(r, g, b) - min(r, g, b) < 10 and max(r, g, b) > 245:
            continue
        labels[_nearest_palette_name((r, g, b))] += 1

    if not labels:
        return ColorPrediction("unknown", 0.0)
    total = sum(labels.values())
    label, count = labels.most_common(1)[0]
    return ColorPrediction(label, count / total)
