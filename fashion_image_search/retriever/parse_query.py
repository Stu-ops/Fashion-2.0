"""Natural-language query parsers for compositional fashion retrieval."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request

from fashion_image_search.common.schemas import GarmentSlot, ParsedQuery
from fashion_image_search.indexer.attributes import PALETTE
from fashion_image_search.indexer.embed import embed_text


GARMENTS = {
    "raincoat": {"raincoat", "coat", "outerwear", "jacket"},
    "tie": {"tie", "neckwear"},
    "shirt": {"shirt", "button-down", "blouse", "top", "t-shirt", "tee"},
    "blazer": {"blazer", "suit", "jacket"},
    "hoodie": {"hoodie", "sweatshirt"},
    "pants": {"pants", "trousers", "jeans"},
    "dress": {"dress"},
}
SCENE_TERMS = {
    "office": {"office", "business", "professional", "workplace"},
    "park": {"park", "bench", "garden"},
    "city street": {"city", "urban", "street", "walk"},
    "home": {"home", "indoor", "room"},
    "formal setting": {"formal", "business"},
}
STYLE_TERMS = {"casual", "weekend", "professional", "formal", "business", "modern"}


SYSTEM_PROMPT = """You extract structured slots for a fashion image retrieval system.
Return only valid JSON. Do not include markdown.
Schema:
{
  "garments": [{"color": string|null, "type": string|null}],
  "scene": string|null,
  "style": string|null
}
Rules:
- Preserve multiple garments as separate objects.
- Bind each color to the garment it describes.
- Use simple garment types such as shirt, tie, blazer, coat, raincoat, pants, dress, hoodie.
- Put location/context words like office, park, city street, home, formal setting in scene.
- Put vibe words like casual, weekend, professional, business, formal in style.
"""


def _canonical_garment(token: str) -> str | None:
    for canonical, aliases in GARMENTS.items():
        if token in aliases:
            return canonical
    return None


def _find_scene(text: str) -> str | None:
    words = set(re.findall(r"[a-z-]+", text.lower()))
    matches = [
        scene for scene, aliases in SCENE_TERMS.items()
        if words.intersection(aliases)
    ]
    return ", ".join(matches) if matches else None


def _clean_optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip().lower()
    if not value or value == "null":
        return None
    return value


def _parsed_from_payload(text: str, payload: dict[str, object]) -> ParsedQuery:
    slots: list[GarmentSlot] = []
    garments = payload.get("garments", [])
    if isinstance(garments, list):
        for item in garments:
            if not isinstance(item, dict):
                continue
            garment_type = _clean_optional_string(item.get("type"))
            color = _clean_optional_string(item.get("color"))
            if not garment_type and not color:
                continue
            phrase = " ".join(part for part in [color, garment_type] if part)
            slots.append(GarmentSlot(garment_type=garment_type, color=color, phrase=phrase))

    return ParsedQuery(
        raw_text=text,
        garment_slots=slots,
        scene_phrase=_clean_optional_string(payload.get("scene")),
        style_residual=_clean_optional_string(payload.get("style")),
        full_query_text_embedding=embed_text(text),
    )


def parse_query_rule(text: str) -> ParsedQuery:
    tokens = re.findall(r"[a-z-]+", text.lower())
    slots: list[GarmentSlot] = []
    for index, token in enumerate(tokens):
        garment = _canonical_garment(token)
        if not garment:
            continue
        window = tokens[max(0, index - 4): index + 1]
        colors = [item for item in window if item in PALETTE]
        color = colors[-1] if colors else None
        phrase = " ".join(item for item in [color, garment] if item)
        slots.append(GarmentSlot(garment_type=garment, color=color, phrase=phrase or garment))

    seen: set[tuple[str | None, str | None]] = set()
    deduped: list[GarmentSlot] = []
    for slot in slots:
        key = (slot.garment_type, slot.color)
        if key not in seen:
            deduped.append(slot)
            seen.add(key)

    style_words = [token for token in tokens if token in STYLE_TERMS]
    return ParsedQuery(
        raw_text=text,
        garment_slots=deduped,
        scene_phrase=_find_scene(text),
        style_residual=" ".join(style_words) or None,
        full_query_text_embedding=embed_text(text),
    )


def parse_query_openai_compatible(text: str) -> ParsedQuery:
    base_url = os.getenv("FASHION_SEARCH_LLM_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("FASHION_SEARCH_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = os.getenv("FASHION_SEARCH_LLM_MODEL", "gpt-4o-mini")
    if not api_key:
        raise RuntimeError("Set FASHION_SEARCH_LLM_API_KEY or OPENAI_API_KEY for LLM parsing.")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    request = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"OpenAI-compatible parser failed: {exc}") from exc

    content = raw["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError("OpenAI-compatible parser returned no message content.")
    return _parsed_from_payload(text, json.loads(content))


def parse_query(text: str, parser_backend: str = "rule") -> ParsedQuery:
    if parser_backend == "rule":
        return parse_query_rule(text)
    if parser_backend in {"openai", "opencode", "openai-compatible"}:
        try:
            return parse_query_openai_compatible(text)
        except RuntimeError:
            return parse_query_rule(text)
    raise ValueError(f"Unknown parser backend: {parser_backend}")
