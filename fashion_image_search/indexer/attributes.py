"""Lightweight visual attributes for garment crops."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from PIL import Image


PALETTE: dict[str, tuple[int, int, int]] = {
    "black": (20, 20, 20),
    "white": (238, 238, 232),
    "gray": (128, 128, 128),
    "red": (210, 45, 45),
    "orange": (230, 120, 35),
    "yellow": (230, 205, 40),
    "green": (55, 150, 80),
    "blue": (55, 105, 200),
    "purple": (125, 75, 165),
    "pink": (225, 110, 165),
    "brown": (120, 80, 50),
    "beige": (205, 180, 140),
}


@dataclass(frozen=True)
class ColorPrediction:
    label: str
    confidence: float


def _nearest_palette_name(rgb: tuple[int, int, int]) -> str:
    return min(
        PALETTE,
        key=lambda name: sum((rgb[channel] - PALETTE[name][channel]) ** 2 for channel in range(3)),
    )


def dominant_color(image: Image.Image, sample_size: int = 64) -> ColorPrediction:
    rgb = image.convert("RGB").resize((sample_size, sample_size))
    labels: Counter[str] = Counter()
    for pixel in rgb.getdata():
        r, g, b = pixel
        if max(pixel) - min(pixel) < 10 and max(pixel) > 245:
            continue
        labels[_nearest_palette_name((r, g, b))] += 1
    if not labels:
        return ColorPrediction("unknown", 0.0)
    total = sum(labels.values())
    label, count = labels.most_common(1)[0]
    return ColorPrediction(label, count / total)

