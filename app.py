from __future__ import annotations

from pathlib import Path
import streamlit as st

from engine.pipeline import build_and_save_index, load_index
from engine.query_engine import answer_query
from engine.providers.hf_transformers import hf_is_available


st.set_page_config(page_title="Rhythmx Local LLM Query Engine", layout="wide")

c1, c2 = st.columns([1, 5], vertical_alignment="center")
with c1:
    st.image("assets/rhythmx_logo.png", width=200)
with c2:
    st.title("FHIR Query Engine")
    st.caption("Developed by Sujyoth Karkera for GW RhythmX.")

with st.sidebar:
    st.header("Settings")
    data_dir = st.text_input("FHIR data directory", value="data")
    index_path = st.text_input("Index path", value="artifacts/index.pkl")
    top_k = st.slider("Top-k retrieval", min_value=3, max_value=20, value=8)
    show_context = st.checkbox("Show retrieved context", value=True)

    st.divider()
    use_llm = st.checkbox("Use local LLM for final answer", value=True)
    llm_provider = st.selectbox("LLM provider", options=["hf"], index=0)
    llm_model_path = st.text_input("Local model path", value="models/tinyllama")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    max_tokens = st.slider("Max output tokens", min_value=128, max_value=2048, value=700, step=64)

    ok, msg = hf_is_available()
    if ok:
        st.success("Transformers backend available.")
    else:
        st.warning(f"Transformers backend issue: {msg}")

    if st.button("Build/Rebuild index"):
        Path("artifacts").mkdir(exist_ok=True, parents=True)
        build_and_save_index(data_dir=data_dir, index_path=index_path)
        st.success("Index built.")

st.subheader("Ask a question")
q = st.text_input("Question", value="Does the patient have asthma?")
run = st.button("Run")

if run:
    if not Path(index_path).exists():
        st.warning("Index not found. Building now...")
        build_and_save_index(data_dir=data_dir, index_path=index_path)

    idx = load_index(index_path)
    result = answer_query(
        idx,
        q,
        top_k=top_k,
        use_llm=use_llm,
        llm_provider="hf",
        llm_model_path=llm_model_path,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    st.markdown("### Answer")
    st.write(result["answer"])
    st.caption(f"Intent: {result.get('intent')} | Used LLM: {result.get('used_llm')}")

    if show_context:
        st.markdown("### Retrieved context (for transparency)")
        for hit in result["retrieved"]:
            with st.expander(f'{hit["source"]} (score={hit["score"]:.4f})', expanded=False):
                st.text(hit["text"])
