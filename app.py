"""Streamlit interface for the Advanced RAG PDF Assistant."""

import os

import streamlit as st
from dotenv import load_dotenv

from rag.pipeline import RAGPipeline

load_dotenv()

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
REQUIRED_KEYS = ["GROQ_API_KEY", "COHERE_API_KEY"]

st.set_page_config(page_title="Advanced RAG PDF Assistant", page_icon="📄")

st.title("📄 Advanced RAG PDF Assistant")
st.write(
    "Ask questions about your PDFs. Unlike basic RAG, this app first **expands your question** "
    "into several search queries, then **reranks** the results with Cohere so only the most "
    "relevant passages reach the LLM. Every answer is grounded in your documents and cites its sources."
)


def missing_api_keys() -> list:
    """Keys that are unset or still hold the .env.example placeholder."""
    return [k for k in REQUIRED_KEYS if not os.getenv(k, "").strip() or os.getenv(k).startswith("your_")]


missing = missing_api_keys()
if missing:
    st.error(
        f"Missing API key(s): {', '.join(missing)}.\n\n"
        "Copy `.env.example` to `.env`, add your keys, then restart the app."
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def get_pipeline() -> RAGPipeline:
    # Created once and reused across reruns, so API clients aren't rebuilt on every click.
    return RAGPipeline(PERSIST_DIR, os.environ["GROQ_API_KEY"], os.environ["COHERE_API_KEY"])


pipeline = get_pipeline()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Upload at least one PDF first.")
        else:
            with st.spinner("Extracting, chunking and embedding..."):
                for file in uploaded_files:
                    try:
                        added = pipeline.add_pdf(file.name, file.getvalue())
                        if added:
                            st.success(f"{file.name}: {added} chunks indexed")
                        else:
                            st.info(f"{file.name}: already indexed or no text found")
                    except Exception as e:
                        st.error(f"{file.name}: {e}")

    if st.button("Reset Database"):
        pipeline.reset()
        st.session_state.pop("result", None)
        st.success("Database cleared.")

    st.divider()
    st.metric("Indexed chunks", pipeline.vector_store.count())
    top_k = st.slider("Top-k chunks sent to the LLM", min_value=1, max_value=10, value=5)

# ---------- Main area ----------
question = st.text_input("Your question", placeholder="e.g. What are the main conclusions of the report?")

if st.button("Ask", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
    elif pipeline.vector_store.count() == 0:
        st.warning("Upload and process at least one PDF first.")
    else:
        with st.spinner("Expanding query, searching, reranking and writing the answer..."):
            try:
                st.session_state["result"] = pipeline.ask(question.strip(), top_k)
            except Exception as e:
                st.error(f"Something went wrong: {e}")

# Results are kept in session state so they stay visible when the page reruns.
result = st.session_state.get("result")
if result:
    st.subheader("Answer")
    st.markdown(result["answer"].replace("$", "\\$"))  # stop "$" from being read as LaTeX

    with st.expander("Expanded Queries"):
        for i, query in enumerate(result["queries"], start=1):
            label = " *(original)*" if i == 1 else ""
            st.markdown(f"{i}. {query}{label}")
        st.caption(f"{result['num_candidates']} unique chunks found before reranking.")

    with st.expander(f"Sources ({len(result['sources'])})"):
        for i, src in enumerate(result["sources"], start=1):
            st.markdown(
                f"**[{i}] {src['source']}** · page {src['page']} · "
                f"rerank score `{src['score']:.3f}`"
            )
            st.text(src["text"])
            if i < len(result["sources"]):
                st.divider()
