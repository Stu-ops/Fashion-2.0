# Query Parsing & NER Extraction for Fashion Retrieval

## Overview

The fashion retrieval system supports **three query parser backends** for extracting structured search slots from natural language queries. Each has different accuracy, setup requirements, and use cases.

---

## 1. Rule-Based Parser (Default: `rule`)

**File:** `fashion_image_search/retriever/parse_query.py` → `parse_query_rule()`

### How it works
- Tokenizes the query into words using regex `[a-z-]+`
- Matches tokens against a hard-coded **garment vocabulary** (`GARMENTS` dict):
  - `tie`, `shirt`, `blazer`, `coat`, `raincoat`, `hoodie`, `pants`, `dress`
- For each detected garment, looks **4 tokens backwards** in a sliding window to find a colour word from the 12-colour palette
- Detects **scene terms** (office, park, city street, home, formal setting) using a keyword set
- Extracts **style words** (casual, weekend, professional, formal, business, modern)
- Deduplicates garment slots (by `garment_type + color`)

### Example:
```
Input:  "A red tie and a white shirt in a formal setting."
Output: garment_slots=[{color:"red", type:"tie"}, {color:"white", type:"shirt"}]
        scene_phrase="formal setting"
        style_residual="formal"
```

### Pros/Cons
- ✅ No setup — works offline, no API key, no network
- ✅ Fast (~0.1ms per query)
- ❌ Limited vocabulary — misses unknown garment types
- ❌ Colour binding fails if colour word is far from garment word
- ❌ No understanding of novel phrases like "vintage leather jacket"

---

## 2. OpenAI-Compatible LLM Parser (`openai` / `opencode` / `openai-compatible`)

**File:** `fashion_image_search/retriever/parse_query.py` → `parse_query_openai_compatible()`

### How it works
- Sends the raw query to an LLM with a structured **system prompt** asking for JSON output
- The LLM extracts: `garments` (array of `{color, type}`), `scene`, `style`
- Supports any **OpenAI-compatible API** (OpenAI, OpenCode, local LLMs via vLLM/TGI)
- Falls back to rule-based parser if the API call fails

### System Prompt (hard-coded):
```
You extract structured slots for a fashion image retrieval system.
Return only valid JSON. Do not include markdown.
Schema: {"garments": [{"color": string|null, "type": string|null}],
         "scene": string|null, "style": string|null}
Rules:
- Preserve multiple garments as separate objects.
- Bind each color to the garment it describes.
- Use simple garment types such as shirt, tie, blazer, coat, raincoat, pants, dress, hoodie.
- Put location/context words like office, park, city street, home, formal setting in scene.
- Put vibe words like casual, weekend, professional, business, formal in style.
```

### Setup — Environment Variables
```bash
# Required for LLM parsing:
export FASHION_SEARCH_LLM_API_KEY="sk-your-api-key"
# or fallback:
export OPENAI_API_KEY="sk-your-api-key"

# Optional overrides:
export FASHION_SEARCH_LLM_BASE_URL="https://api.openai.com/v1"   # default
export FASHION_SEARCH_LLM_MODEL="gpt-4o-mini"                    # default
```

### Example with OpenAI:
```bash
# Set key
set FASHION_SEARCH_LLM_API_KEY=sk-xxxxx

# Run search with LLM parser
streamlit run streamlit_app.py   # then select parser="openai" in sidebar

# Or via CLI
.venv\Scripts\python -m fashion_image_search.retriever.search \
  "A vintage leather jacket and ripped jeans" \
  --parser openai \
  --backend hf
```

### Pros/Cons
- ✅ Handles novel garments ("vintage leather jacket" → jacket, "ripped jeans" → pants)
- ✅ Better colour binding — understands "the red one" refers to tie
- ✅ Extracts scene/style from complex sentences
- ❌ Requires API key and internet
- ❌ Slower (~1-3 seconds per query)
- ❌ Costs money (or requires self-hosted LLM)

---

## 3. NER Extraction — The Core Problem

### Why NER Matters
The key challenge in fashion retrieval is **binding colours to garments** across complex queries:

```
"A red tie and a white shirt in a formal setting."
           ↑              ↑
        red→tie        white→shirt
```

A naive approach might detect "red" and "tie" independently but miss the binding. The **rule parser** solves this with a sliding window (4 tokens backwards from the garment word). The **LLM parser** solves this naturally.

### Current NER Architecture

```
Raw Query
    │
    ▼
┌─────────────────────┐
│  Tokenization        │
│  (regex split)       │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Colour Detection   │  ← 12-colour palette + cosine distance
│  (nearest palette)  │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Garment Detection  │  ← keyword matching against 8 garment types
│  (canonical map)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Colour Binding     │  ← sliding window: look 4 tokens before garment
│  (in-text-order)    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Scene Extraction   │  ← keyword set matching
│  (scene terms)      │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Style Extraction   │  ← keyword set matching
│  (style terms)      │
└─────────┬───────────┘
          │
          ▼
      ParsedQuery
```

### Where NER lives in the code

```python
fashion_image_search/retriever/parse_query.py
├── GARMENTS          → dict[str, set[str]]    – garment synonyms
├── SCENE_TERMS       → dict[str, set[str]]    – scene keywords
├── STYLE_TERMS       → set[str]               – style keywords
├── _canonical_garment()   – alias → canonical name
├── _find_scene()          – word intersection
├── parse_query_rule()     – full rule pipeline
└── parse_query_openai_compatible() – LLM pipeline
```

---

## 4. Setting Up the LLM Parser

### Option A: OpenAI (cloud, paid)
```bash
set FASHION_SEARCH_LLM_API_KEY=sk-xxxxxxxxxxxxxxxx
set FASHION_SEARCH_LLM_MODEL=gpt-4o-mini     # cheap, fast
# Then select "openai" in the Streamlit sidebar
```

### Option B: OpenCode CLI (local, free)
```bash
# Install opencode
npm install -g opencode

# Start local server
opencode start

# Point the app to it
set FASHION_SEARCH_LLM_BASE_URL=http://localhost:11434/v1
set FASHION_SEARCH_LLM_API_KEY=not-needed
set FASHION_SEARCH_LLM_MODEL=codestral
# Select "opencode" in sidebar
```

### Option C: vLLM / TGI (self-hosted, free)
```bash
# Run a local model like Llama 3 or Mistral
docker run -p 8000:8000 vllm/vllm-openai \
  --model meta-llama/Meta-Llama-3-8B-Instruct

set FASHION_SEARCH_LLM_BASE_URL=http://localhost:8000/v1
set FASHION_SEARCH_LLM_API_KEY=not-needed
set FASHION_SEARCH_LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
# Select "openai-compatible" in sidebar
```

---

## 5. How to add a new garment type

Edit `parse_query.py` → `GARMENTS` dict:

```python
GARMENTS = {
    "raincoat": {"raincoat", "coat", "outerwear", "jacket"},
    "tie": {"tie", "neckwear"},
    "shirt": {"shirt", "button-down", "blouse", "top", "t-shirt", "tee"},
    "blazer": {"blazer", "suit", "jacket"},
    "hoodie": {"hoodie", "sweatshirt"},
    "pants": {"pants", "trousers", "jeans"},
    "dress": {"dress"},
    "skirt": {"skirt", "mini-skirt", "pleated"},          # ← added
    "sunglasses": {"sunglasses", "shades", "sun-glasses"},  # ← added
}
```

And add corresponding aliases in `detect.py` if using HF detector, which uses Fashionpedia's own 46 categories.

---

## 6. Testing the parser standalone

```python
from fashion_image_search.retriever.parse_query import parse_query

# Rule-based
parsed = parse_query("A red tie and a white shirt in a formal setting.", "rule")
print(parsed.garment_slots)  # [GarmentSlot('tie','red'), GarmentSlot('shirt','white')]

# LLM-based (requires API key)
parsed = parse_query("A vintage leather jacket and ripped jeans.", "openai")
print(parsed.garment_slots)  # [GarmentSlot('jacket','leather'), GarmentSlot('pants','ripped')]