"""Image and text embedding adapters."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from functools import lru_cache

from PIL import Image, ImageStat

from fashion_image_search.common.config import MODEL
from fashion_image_search.common.schemas import Vector
from fashion_image_search.indexer.attributes import PALETTE, dominant_color


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def _hash_bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % dim
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    return index, sign


def embed_text(text: str, dim: int = MODEL.embedding_dim) -> Vector:
    vector = [0.0] * dim
    for token in TOKEN_RE.findall(text.lower()):
        index, sign = _hash_bucket(token, dim)
        vector[index] += sign
        if token in PALETTE:
            color_index = list(PALETTE).index(token) % dim
            vector[color_index] += 2.5
    return _normalize(vector)


def _image_feature_tokens(image: Image.Image) -> list[str]:
    color = dominant_color(image).label
    width, height = image.size
    ratio = width / max(height, 1)
    stat = ImageStat.Stat(image.convert("L").resize((24, 24)))
    brightness = "bright" if stat.mean[0] > 150 else "dark"
    shape = "wide" if ratio > 1.2 else "tall" if ratio < 0.8 else "balanced"
    return [color, brightness, shape]


def embed_image(image: Image.Image, dim: int = MODEL.embedding_dim) -> Vector:
    return embed_text(" ".join(_image_feature_tokens(image)), dim=dim)


def embed_image_file(path: Path, dim: int = MODEL.embedding_dim) -> Vector:
    with Image.open(path) as image:
        return embed_image(image.convert("RGB"), dim=dim)


class HuggingFaceFashionEncoder:
    """Marqo FashionSigLIP encoder loaded from Hugging Face Hub.

    Marqo's repository is implemented on top of OpenCLIP, so open_clip_torch is
    the loader/runtime here. It is not a separate retrieval model.
    """

    def __init__(self, model_name: str = MODEL.encoder_name, device: str = MODEL.device):
        try:
            import torch
            import open_clip
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face encoder backend needs torch and open_clip_torch. "
                "Install them with: .\\.venv\\Scripts\\python -m pip install -r requirements.txt"
            ) from exc

        self.torch = torch
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        open_clip_name = model_name if model_name.startswith("hf-hub:") else f"hf-hub:{model_name}"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(open_clip_name)
        self.tokenizer = open_clip.get_tokenizer(open_clip_name)
        self.model.to(self.device)
        self.model.eval()

    def embed_text(self, text: str) -> Vector:
        tokens = self.tokenizer([text]).to(self.device)
        with self.torch.no_grad():
            features = self.model.encode_text(tokens)
        return self._tensor_to_vector(features[0])

    def embed_image(self, image: Image.Image) -> Vector:
        tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        with self.torch.no_grad():
            features = self.model.encode_image(tensor)
        return self._tensor_to_vector(features[0])

    def _tensor_to_vector(self, tensor: object) -> Vector:
        tensor = self.torch.nn.functional.normalize(tensor, dim=0)
        return [float(value) for value in tensor.detach().cpu().tolist()]


@lru_cache(maxsize=1)
def _hf_encoder() -> HuggingFaceFashionEncoder:
    return HuggingFaceFashionEncoder()


def embed_text_backend(text: str, backend: str = MODEL.backend) -> Vector:
    if backend == "offline":
        return embed_text(text)
    if backend in {"hf", "huggingface"}:
        return _hf_encoder().embed_text(text)
    raise ValueError(f"Unknown embedding backend: {backend}")


def embed_image_backend(image: Image.Image, backend: str = MODEL.backend) -> Vector:
    if backend == "offline":
        return embed_image(image)
    if backend in {"hf", "huggingface"}:
        return _hf_encoder().embed_image(image)
    raise ValueError(f"Unknown embedding backend: {backend}")
