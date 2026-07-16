"""Natural-language query parsers for compositional fashion retrieval."""

from __future__ import annotations

import json
import logging
import os
import re
import urllib.error
import urllib.request

from fashion_image_search.common.config import load_dotenv
from fashion_image_search.common.schemas import GarmentSlot, ParsedQuery
from fashion_image_search.indexer.attributes import PALETTE
from fashion_image_search.indexer.embed import embed_text


logger = logging.getLogger(__name__)


# ── Canonical garment vocabulary ─────────────────────────────────────────────
#
# Rules:
#   • Each key is the *canonical* garment type stored in search results and
#     used in _category_matches() in search.py.
#   • Aliases are the tokens the rule parser looks for in the raw query text.
#   • "jacket" lives ONLY under "blazer" — removing it from "raincoat" fixes
#     the silent alias-collision bug (Bug #3) where "jacket" always resolved
#     to "raincoat" because that key was defined first (dict insertion order).
#   • "coat" is its own canonical type, separate from "raincoat", so that
#     Fashionpedia's "coat" label normalises correctly to a query-matchable key.

GARMENTS: dict[str, set[str]] = {
    "coat":     {"coat", "overcoat", "trench", "parka", "outerwear"},
    "raincoat": {"raincoat", "rain-coat"},            # "jacket" removed here
    "tie":      {"tie", "neckwear", "necktie"},
    "shirt":    {"shirt", "button-down", "blouse", "top", "t-shirt", "tee", "sweater",
                 "cardigan", "sweatshirt"},
    "blazer":   {"blazer", "suit", "jacket"},         # "jacket" lives here only
    "hoodie":   {"hoodie"},
    "pants":    {"pants", "trousers", "jeans", "trouser", "leggings", "tights"},
    "shorts":   {"shorts", "short"},                  # Bug #10 — was missing
    "skirt":    {"skirt", "mini-skirt", "pleated", "mini"},  # Bug #10 — was missing
    "dress":    {"dress", "jumpsuit", "gown"},
    "vest":     {"vest", "waistcoat"},
    "scarf":    {"scarf"},
    "hat":      {"hat", "cap", "beanie", "headband"},
    "shoe":     {"shoe", "shoes", "sneaker", "sneakers", "boot", "boots"},
    "glasses":  {"glasses", "sunglasses", "shades"},
    "glove":    {"glove", "gloves"},
}

SCENE_TERMS: dict[str, set[str]] = {
    "office":          {"office", "business", "professional", "workplace", "desk"},
    "park":            {"park", "bench", "garden", "outdoor", "nature"},
    "city street":     {"city", "urban", "street", "walk", "sidewalk"},
    "home":            {"home", "indoor", "room", "living"},
    "formal setting":  {"formal", "ceremony", "gala", "event"},
    "beach":           {"beach", "sand", "ocean", "sea", "summer"},
    "gym":             {"gym", "workout", "sport", "athletic", "fitness"},
}

STYLE_TERMS: set[str] = {
    "casual", "weekend", "professional", "formal", "business",
    "modern", "vintage", "classic", "sporty", "elegant", "chic",
}


# ── LLM system prompt ─────────────────────────────────────────────────────────

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
- Use simple garment types such as shirt, tie, blazer, coat, raincoat, pants,
  dress, hoodie, shorts, skirt, vest, hat, shoe, glasses, glove, scarf.
- Put location/context words like office, park, city street, home, formal setting in scene.
- Put vibe words like casual, weekend, professional, business, formal, vintage in style.
"""


# ── Internal helpers ──────────────────────────────────────────────────────────

def _redact_secret(value: str) -> str:
    if len(value) <= 12:
        return "<set>"
    return f"{value[:6]}...{value[-4:]}"


def _http_error_message(
    exc: urllib.error.HTTPError,
    *,
    base_url: str,
    model: str,
    api_key: str,
) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace").strip()
    except Exception:
        body = ""
    if len(body) > 500:
        body = f"{body[:500]}..."

    hint = ""
    if exc.code in {401, 403}:
        hint = (
            " Check that FASHION_SEARCH_LLM_BASE_URL, FASHION_SEARCH_LLM_MODEL, "
            "and the API key all belong to the same provider/account."
        )

    message = (
        f"HTTP {exc.code} {exc.reason} from {base_url}/chat/completions "
        f"using model={model!r} and key={_redact_secret(api_key)}."
    )
    if body:
        message = f"{message} Response body: {body}"
    return f"{message}{hint}"


def _canonical_garment(token: str) -> str | None:
    """Map a raw query token to its canonical garment type, or None if not a garment."""
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
            # Canonicalise the LLM-returned garment type
            if garment_type:
                canonical = _canonical_garment(garment_type)
                if canonical:
                    garment_type = canonical
            phrase = " ".join(part for part in [color, garment_type] if part)
            slots.append(GarmentSlot(garment_type=garment_type, color=color, phrase=phrase))

    return ParsedQuery(
        raw_text=text,
        garment_slots=slots,
        scene_phrase=_clean_optional_string(payload.get("scene")),
        style_residual=_clean_optional_string(payload.get("style")),
        full_query_text_embedding=embed_text(text),
    )


# ── Rule-based parser ─────────────────────────────────────────────────────────

def parse_query_rule(text: str) -> ParsedQuery:
    """Parse a fashion query using rule-based NER with sliding-window colour binding.

    Colour binding: for each detected garment token, look back up to 4 tokens to
    find a colour word. This correctly associates "red" with "tie" and "white" with
    "shirt" in "a red tie and a white shirt".
    """
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

    # Deduplicate by (type, color) while preserving in-text order
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


# ── LLM-based parser ──────────────────────────────────────────────────────────

def parse_query_openai_compatible(text: str) -> ParsedQuery:
    """Parse a fashion query via any OpenAI-compatible chat completion endpoint.

    Required env vars:
        FASHION_SEARCH_LLM_API_KEY  (or OPENAI_API_KEY)
    Optional:
        FASHION_SEARCH_LLM_BASE_URL  (default: https://api.openai.com/v1)
        FASHION_SEARCH_LLM_MODEL     (default: gpt-4o-mini)
    """
    # Re-load .env so edited parser credentials are picked up by long-running
    # Streamlit sessions without requiring a process restart.
    load_dotenv(override=True)
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
    except urllib.error.HTTPError as exc:
        raise RuntimeError(
            "OpenAI-compatible parser failed: "
            + _http_error_message(exc, base_url=base_url, model=model, api_key=api_key)
        ) from exc
    except (OSError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"OpenAI-compatible parser failed for {base_url}/chat/completions "
            f"using model={model!r}: {exc}"
        ) from exc

    content = raw["choices"][0]["message"]["content"]
    if not isinstance(content, str):
        raise RuntimeError("OpenAI-compatible parser returned no message content.")
    return _parsed_from_payload(text, json.loads(content))


# ── Public dispatcher ─────────────────────────────────────────────────────────

def parse_query(
    text: str,
    parser_backend: str = "rule",
    *,
    fallback_on_error: bool = True,
) -> ParsedQuery:
    """Parse *text* using the specified backend.

    Backends:
        ``"rule"``                – fast, offline, no setup required (default)
        ``"openai"`` / ``"opencode"`` / ``"openai-compatible"``
                                  – LLM parsing via any OpenAI-compatible API;
                                    falls back to rule parser on errors by default
    """
    if parser_backend == "rule":
        return parse_query_rule(text)
    if parser_backend in {"openai", "opencode", "openai-compatible"}:
        try:
            parsed = parse_query_openai_compatible(text)
            logger.info(
                "LLM parser succeeded for backend=%s slots=%s scene=%r style=%r",
                parser_backend,
                [(slot.color, slot.garment_type) for slot in parsed.garment_slots],
                parsed.scene_phrase,
                parsed.style_residual,
            )
            return parsed
        except Exception:
            if not fallback_on_error:
                raise
            logger.exception("LLM parser failed for backend=%s; falling back to rule parser", parser_backend)
            return parse_query_rule(text)
    raise ValueError(f"Unknown parser backend: {parser_backend!r}")
