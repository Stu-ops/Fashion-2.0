"""Streamlit interface for the fashion retrieval pipeline."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from fashion_image_search.common.config import MODEL, PATHS, SEARCH
from fashion_image_search.retriever.search import search


st.set_page_config(page_title="Fashion Image Search", layout="wide")


def _existing_path(label: str, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        st.sidebar.warning(f"{label} does not exist: {path}")
    return path


st.title("Fashion Image Search")

with st.sidebar:
    st.header("Retrieval Settings")
    index_path = _existing_path("Metadata index", st.text_input("Metadata JSON", str(PATHS.index_path)))
    faiss_path = _existing_path("FAISS index", st.text_input("FAISS index", str(PATHS.faiss_index_path)))
    backend = st.selectbox("Embedding backend", ["hf", "offline"], index=0)
    store = st.selectbox("Vector store", ["faiss", "json"], index=0)
    parser_backend = st.selectbox("Query parser", ["rule", "openai", "opencode", "openai-compatible"], index=0)
    top_k = st.slider("Top K", min_value=1, max_value=30, value=SEARCH.default_top_k)

    st.caption("Build the index before searching:")
    st.code(
        ".\\.venv\\Scripts\\python -m fashion_image_search.indexer.build_index "
        "--data-dir val_test2020_sample_1600\\test --backend hf --store faiss",
        language="powershell",
    )


query = st.text_input(
    "Natural language query",
    value="A red tie and a white shirt in a formal setting.",
)

if st.button("Search", type="primary"):
    if not index_path.exists():
        st.error(f"Metadata index not found: {index_path}")
        st.stop()
    if store == "faiss" and not faiss_path.exists():
        st.error(f"FAISS index not found: {faiss_path}")
        st.stop()

    with st.spinner("Running Stage 1 ANN recall and Stage 2 region rerank..."):
        try:
            results = search(
                index_path=index_path,
                query=query,
                top_k=top_k,
                embedding_backend=backend,
                store_kind=store,
                faiss_path=faiss_path,
                parser_backend=parser_backend,
            )
        except Exception as exc:  # Streamlit should show actionable failure text.
            st.exception(exc)
            st.stop()

    if not results:
        st.warning("No results returned.")
        st.stop()

    cols = st.columns(3)
    for idx, result in enumerate(results):
        image_path = Path(result.image_path)
        with cols[idx % len(cols)]:
            if image_path.exists():
                st.image(str(image_path), use_container_width=True)
            else:
                st.warning(f"Missing image: {image_path}")
            st.markdown(f"**Rank {idx + 1}**")
            st.caption(image_path.name)
            st.write(f"score: `{result.score:.3f}`")
            with st.expander("Score breakdown"):
                st.write(f"global: `{result.global_score:.3f}`")
                st.write(f"scene: `{result.scene_score:.3f}`")
                st.write(f"slot: `{result.slot_score:.3f}`")
                st.write(f"attribute bonus: `{result.attribute_bonus:.3f}`")
                for line in result.slot_breakdown:
                    st.write(f"- {line}")
