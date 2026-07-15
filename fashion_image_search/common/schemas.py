"""Typed data contracts shared by indexer and retriever."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


Vector = list[float]
BBox = tuple[int, int, int, int]


@dataclass
class RegionRecord:
    region_idx: int
    bbox: BBox
    category: str
    detector_confidence: float
    region_embedding: Vector
    color: str
    color_confidence: float

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RegionRecord":
        payload = dict(payload)
        payload["bbox"] = tuple(payload["bbox"])
        return cls(**payload)


@dataclass
class ImageRecord:
    image_id: str
    image_path: str
    global_embedding: Vector
    scene_embedding: Vector
    regions: list[RegionRecord] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for region in payload["regions"]:
            region["bbox"] = list(region["bbox"])
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImageRecord":
        return cls(
            image_id=payload["image_id"],
            image_path=payload["image_path"],
            global_embedding=list(payload["global_embedding"]),
            scene_embedding=list(payload["scene_embedding"]),
            regions=[RegionRecord.from_dict(item) for item in payload.get("regions", [])],
        )


@dataclass
class GarmentSlot:
    garment_type: str | None = None
    color: str | None = None
    phrase: str = ""


@dataclass
class ParsedQuery:
    raw_text: str
    garment_slots: list[GarmentSlot]
    scene_phrase: str | None
    style_residual: str | None
    full_query_text_embedding: Vector

