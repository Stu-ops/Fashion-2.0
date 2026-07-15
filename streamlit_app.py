"""Streamlit interface for the fashion retrieval pipeline.

Features:
- Upload images into the dataset folder via UI
- Auto-detect dataset changes and rebuild index
- Reset everything (images + DB + JSON) with one click
- Paginated dataset grid (handles large collections)
- Search via natural language queries
- Show parsed query slots before results for transparency
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import streamlit as st

from fashion_image_search.common.config import MODEL, PATHS, SEARCH
from fashion_image_search.common.vector_db import close_all_open_stores
from fashion_image_search.indexer.build_index import IMAGE_EXTENSIONS, build_index, iter_images
from fashion_image_search.retriever.parse_query import parse_query
from fashion_image_search.retriever.search import search, _format_parsed_query


logger = logging.getLogger("fashion_streamlit")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(handler)


st.set_page_config(page_title="Fashion Image Search", layout="wide")

# ── Fixed paths (no user-editable inputs) ───────────────────────────────────
ROOT = Path(__file__).resolve().parent
DATASET_DIR = PATHS.data_dir          # now consistent with CLI default (Bug #9)
JSON_PATH   = ROOT / "artifacts" / "fashion_index.json"
FAISS_PATH  = ROOT / "artifacts" / "fashion_global.faiss"
PAGE_SIZE   = 12                      # images per page in dataset grid

# Colour chips for parsed-query display
_SLOT_COLOURS: dict[str, str] = {
    "red": "#e82e2e", "blue": "#2e6ec8", "green": "#37966f",
    "yellow": "#c8a000", "orange": "#e67820", "purple": "#7c4ba0",
    "pink": "#c85090", "black": "#222222", "white": "#888888",
    "gray": "#666666", "brown": "#7a5030", "beige": "#a08060",
}


# ── Session state initialisation ─────────────────────────────────────────────
if "index_built_once" not in st.session_state:
    st.session_state.index_built_once = False
if "dataset_page" not in st.session_state:
    st.session_state.dataset_page = 0


# ── Helper functions ──────────────────────────────────────────────────────────

@st.cache_resource
def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


def _fmt_size(size_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _find_in_dataset(stem: str) -> Path | None:
    """Try to locate an image by its stem (filename without extension)."""
    if not DATASET_DIR.is_dir():
        return None
    for ext in {".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG", ".WEBP"}:
        candidate = DATASET_DIR / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def _count_images() -> int:
    if not DATASET_DIR.is_dir():
        return 0
    return len(iter_images(DATASET_DIR, limit=None))


def _count_indexed() -> int:
    if not JSON_PATH.exists():
        return 0
    try:
        with JSON_PATH.open("r", encoding="utf-8") as handle:
            return len(json.load(handle))
    except (json.JSONDecodeError, OSError):
        return 0


def _index_is_usable() -> bool:
    if not JSON_PATH.exists() or JSON_PATH.stat().st_size < 10:
        return False
    return True


def _needs_rebuild() -> bool:
    img_count = _count_images()
    idx_count = _count_indexed()
    if img_count == 0:
        return False
    return img_count != idx_count or not _index_is_usable()


def _run_indexing(backend: str) -> bool:
    try:
        # Release any open FAISS handles before writing new index
        close_all_open_stores()

        msg_placeholder = st.info("Running detection → colour extraction → embedding …")
        progress_bar = st.progress(0.0, text="Indexing images …")

        import tqdm as _real_tqdm

        class _StreamlitTqdm(_real_tqdm.tqdm):
            def update(self, n=1):
                super().update(n)
                if self.total:
                    progress_bar.progress(self.n / self.total, text=f"Indexing {self.n}/{self.total} …")

        import fashion_image_search.indexer.build_index as _bi
        _bi.tqdm = _StreamlitTqdm

        store_obj = build_index(
            data_dir=DATASET_DIR,
            output_path=JSON_PATH,
            limit=None,
            detector_backend=backend,
            embedding_backend=backend,
            store_kind="faiss",
            faiss_path=FAISS_PATH,
        )
        progress_bar.empty()
        msg_placeholder.success(f"Indexed {len(store_obj.records)} images")
        st.session_state.index_built_once = True
        return True
    except Exception as exc:
        logger.exception("Indexing failed")
        st.error(f"Indexing failed: {exc}")
        return False


def _reset_all() -> None:
    """Delete all Dataset images and index files with proper cleanup.

    On Windows, FAISS memory-mapped files block deletion until the handle is
    released. This ensures all stores are closed before attempting unlink.
    Each file deletion is wrapped in a short retry loop to handle transient
    permission errors from antivirus or shell extensions.
    """
    # Release any memory-mapped FAISS handles (Windows requirement)
    close_all_open_stores()

    def _safe_unlink(path: Path, description: str) -> bool:
        """Try to delete a file, retrying briefly on permission errors."""
        if not path.exists():
            return False
        for attempt in range(3):
            try:
                path.unlink()
                return True
            except OSError as exc:
                if attempt < 2:
                    time.sleep(0.2)  # Brief pause before retry
                    continue
                st.warning(f"Could not delete {description}: {exc}")
                return False
        return False

    # Delete Dataset images
    if DATASET_DIR.is_dir():
        for item in list(DATASET_DIR.iterdir()):
            if item.is_file():
                _safe_unlink(item, f"image {item.name}")
        st.info("🗑️ Dataset images deleted.")

    # Delete index files with retries
    if _safe_unlink(JSON_PATH, "JSON index"):
        st.info("🗑️ JSON index deleted.")
    if _safe_unlink(FAISS_PATH, "FAISS index"):
        st.info("🗑️ FAISS index deleted.")

    st.session_state.index_built_once = False
    st.session_state.dataset_page = 0
    st.cache_resource.clear()
    st.rerun()


def _render_parsed_query_card(parsed_query) -> None:
    """Render a 'Parsed as:' chip row showing garment slots, scene, and style."""
    chips: list[str] = []

    for slot in parsed_query.garment_slots:
        color_hex = _SLOT_COLOURS.get(slot.color or "", "#555555")
        color_dot = f'<span style="color:{color_hex}; font-size:1.1em;">●</span>'
        label = " ".join(filter(None, [slot.color, slot.garment_type])) or slot.phrase
        chips.append(
            f'<span style="background:#1e293b; border:1px solid #334155; '
            f'border-radius:6px; padding:3px 10px; margin:2px; display:inline-block;">'
            f'{color_dot} <strong>{label}</strong></span>'
        )

    if parsed_query.scene_phrase:
        chips.append(
            f'<span style="background:#0f3460; border:1px solid #1e5799; '
            f'border-radius:6px; padding:3px 10px; margin:2px; display:inline-block;">'
            f'📍 {parsed_query.scene_phrase}</span>'
        )
    if parsed_query.style_residual:
        chips.append(
            f'<span style="background:#2d1b3d; border:1px solid #6b21a8; '
            f'border-radius:6px; padding:3px 10px; margin:2px; display:inline-block;">'
            f'✨ {parsed_query.style_residual}</span>'
        )

    if chips:
        st.markdown(
            f'<div style="margin-bottom:12px;"><span style="color:#94a3b8; '
            f'font-size:0.85em; margin-right:8px;">Parsed as:</span>'
            + "".join(chips) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("Parsed as: (no structured slots — using global style embedding)")


# ── Sidebar — Settings ────────────────────────────────────────────────────────

st.title("🧥 Fashion Image Retrieval")

with st.sidebar:
    st.header("⚙️ Settings")

    backend = st.selectbox(
        "Embedding backend",
        ["offline", "hf"],
        index=1,
        help="See 'Workflow Comparison' below for details.",
    )
    parser_backend = st.selectbox(
        "Query parser",
        ["rule", "openai", "opencode", "openai-compatible"],
        index=0,
    )
    top_k = st.slider("Top K", min_value=1, max_value=30, value=SEARCH.default_top_k)

    with st.expander("🔁 Offline vs HF — what's the difference?"):
        st.markdown("""
| | **Offline** | **HF (HuggingFace)** |
|---|---|---|
| **Detector** | Hard-coded region proposals (4 fixed boxes) | Real YOLOS-fashionpedia — detects actual garments |
| **Encoder** | Hash-based bag-of-words (no semantics) | Marqo FashionSigLIP — SOTA fashion embeddings |
| **GPU** | No | No (CPU ok, GPU 10× faster) |
| **Internet** | No | Yes — downloads model weights once |
| **Accuracy** | Low | High |

**Recommendation:** Use **HF** for real retrieval. Use **Offline** only for testing the UI.
        """)

    with st.expander("🤖 LLM Parser Setup (openai / opencode)"):
        st.markdown("""
Set these environment variables **before** launching Streamlit:

```powershell
# OpenAI (cloud)
$env:FASHION_SEARCH_LLM_API_KEY = "sk-xxxxxx"
$env:FASHION_SEARCH_LLM_MODEL   = "gpt-4o-mini"

# Local LLM via Ollama
$env:FASHION_SEARCH_LLM_BASE_URL = "http://localhost:11434/v1"
$env:FASHION_SEARCH_LLM_API_KEY  = "not-needed"
$env:FASHION_SEARCH_LLM_MODEL    = "llama3"
```

Select **openai** or **openai-compatible** in the parser dropdown above.
        """)

    st.divider()

    # ── Status panel ─────────────────────────────────────────────────────────
    st.subheader("📊 Status")

    img_count   = _count_images()
    idx_count   = _count_indexed()
    needs_build = _needs_rebuild()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Images in Dataset", img_count)
        st.metric("Indexed records", idx_count if idx_count > 0 else "—")
    with col2:
        if needs_build and img_count > 0:
            st.warning("⚠️ Index stale / missing")
        elif img_count == 0:
            st.info("📭 No images yet")
        else:
            st.success("✓ Index up-to-date")

    build_disabled = img_count == 0
    if st.button("⚡ Build / Rebuild Index", disabled=build_disabled, type="primary",
                 help="Run detection → embedding → store for ALL images in Dataset."):
        if not _faiss_available():
            st.error("FAISS not installed. Run: .venv\\Scripts\\python -m pip install faiss-cpu")
        else:
            _run_indexing(backend)

    if st.session_state.index_built_once and idx_count > 0:
        st.caption(f"Last index: {_fmt_size(JSON_PATH.stat().st_size)}")

    st.divider()

    st.subheader("🗑️ Reset Everything")
    st.caption("Delete all uploaded images + both index files (JSON + FAISS).")
    if st.button("Reset Dataset & Indexes", type="secondary",
                 help="⚠️ Permanently deletes ALL images in Dataset/ and both index files."):
        _reset_all()


# ── Tabbed interface ──────────────────────────────────────────────────────────

tab_upload, tab_search = st.tabs(["📤 Upload Images", "🔍 Search"])


# ── Tab 1: Upload ─────────────────────────────────────────────────────────────

with tab_upload:
    st.subheader("Upload fashion images to the Dataset folder")

    uploaded_files = st.file_uploader(
        "Select images (jpg, jpeg, png, webp)",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="file_uploader",
    )

    if uploaded_files:
        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        saved = 0
        for f in uploaded_files:
            dest = DATASET_DIR / f.name
            if not dest.exists():
                dest.write_bytes(f.read())
                saved += 1

        if saved > 0:
            st.success(f"✅ Saved {saved} image(s) to Dataset/")
            st.session_state.dataset_page = 0
            if _needs_rebuild():
                st.info("🔄 Auto-indexing new images …")
                if _faiss_available():
                    _run_indexing(backend)
                else:
                    st.error("FAISS not installed. Install it and rebuild manually.")
            else:
                st.toast("Index is already up-to-date.")
        else:
            st.warning("No new files saved — all filenames already exist in Dataset/.")

    # ── Paginated dataset gallery ─────────────────────────────────────────────
    if DATASET_DIR.is_dir():
        all_images = iter_images(DATASET_DIR, limit=None)
        total = len(all_images)

        if total > 0:
            st.subheader(f"📁 Dataset — {total} image(s)")
            total_pages = (total - 1) // PAGE_SIZE + 1
            page = st.session_state.dataset_page

            start = page * PAGE_SIZE
            end   = min(start + PAGE_SIZE, total)
            batch = all_images[start:end]

            cols = st.columns(4)
            for i, path in enumerate(batch):
                with cols[i % 4]:
                    st.image(str(path), width="stretch")
                    st.caption(path.name)

            nav_cols = st.columns([2, 1, 1, 2])
            with nav_cols[1]:
                if page > 0 and st.button("⬅ Previous", key="prev_page"):
                    st.session_state.dataset_page = page - 1
                    st.rerun()
            with nav_cols[2]:
                if page < total_pages - 1 and st.button("Next ➡", key="next_page"):
                    st.session_state.dataset_page = page + 1
                    st.rerun()
            with nav_cols[0]:
                st.markdown(f"Page {page + 1} / {total_pages}")
        else:
            st.info("📭 Dataset folder is empty — upload images above.")


# ── Tab 2: Search ─────────────────────────────────────────────────────────────

with tab_search:
    st.subheader("Natural language fashion search")

    st.markdown("**Try an example query:**")
    example_cols = st.columns(3)
    examples = [
        "A red tie and a white shirt in a formal setting.",
        "A person in a bright yellow raincoat.",
        "Casual weekend outfit for a city walk.",
        "Professional business attire inside a modern office.",
        "Someone wearing a blue shirt sitting on a park bench.",
    ]
    for col, example in zip(example_cols, examples[:3]):
        if col.button(example, width="stretch"):
            st.session_state.query = example

    query = st.text_input(
        "Describe what you're looking for …",
        value=st.session_state.get("query", ""),
        placeholder="e.g. A red tie and a white shirt in a formal setting.",
    )
    st.session_state.query = query

    if st.button("🔍 Search", type="primary"):
        if not query.strip():
            st.warning("Please enter a query.")
            st.stop()

        if not _index_is_usable():
            st.warning("No index found. Upload images and build the index first.")
            st.stop()

        if not FAISS_PATH.exists():
            st.error(f"FAISS index not found at {FAISS_PATH}. Rebuild the index.")
            st.stop()

        with st.spinner("Running Stage 1 ANN recall + Stage 2 region rerank …"):
            try:
                results = search(
                    index_path=JSON_PATH,
                    query=query,
                    top_k=top_k,
                    embedding_backend=backend,
                    store_kind="faiss",
                    faiss_path=FAISS_PATH,
                    parser_backend=parser_backend,
                )
                # Parse separately to render the query card (Bug #13 fix)
                parsed = parse_query(query, parser_backend)
            except Exception as exc:
                logger.exception("Search failed for query=%r backend=%s parser=%s", query, backend, parser_backend)
                st.exception(exc)
                st.stop()

        # ── Parsed query transparency card (Bug #13 fix) ──────────────────────
        _render_parsed_query_card(parsed)

        if not results:
            st.warning("No results returned. Try a different query or add more images.")
            st.stop()

        st.success(f"Found {len(results)} result(s)")

        cols = st.columns(min(3, top_k))
        for idx, result in enumerate(results):
            image_path = Path(result.image_path)
            with cols[idx % len(cols)]:
                if image_path.exists():
                    st.image(str(image_path), width="stretch")
                else:
                    st.warning(f"Missing: {image_path.name}")
                    alt = _find_in_dataset(image_path.stem)
                    if alt:
                        st.image(str(alt), width="stretch")

                st.markdown(f"**Rank {idx + 1}**")
                st.caption(image_path.name)
                st.write(f"Score: `{result.score:.3f}`")
                with st.expander("Score breakdown"):
                    st.write(f"global: `{result.global_score:.3f}`")
                    st.write(f"scene:  `{result.scene_score:.3f}`")
                    st.write(f"slot:   `{result.slot_score:.3f}`")
                    st.write(f"attr bonus: `{result.attribute_bonus:.3f}`")
                    for line in result.slot_breakdown:
                        st.write(f"- {line}")