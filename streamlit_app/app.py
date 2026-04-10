"""Streamlit frontend for the Policy RAG Assistant."""

import sys
from pathlib import Path

# Allow imports from the project root when running via `streamlit run streamlit_app/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from rag_project.config import load_config  # noqa: E402
from rag_project.ingestion.embeddings import build_embeddings, build_reranker  # noqa: E402
from rag_project.ingestion.vector_store import (  # noqa: E402
    WeaviateRAGPipeline,
    connect_local_weaviate,
)
from rag_project.retrieval.assistant import RAGAssistant  # noqa: E402

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Policy Assistant",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached resource loaders — expensive objects created once per session
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding models...")
def _load_models():
    embeddings = build_embeddings(device="cpu")
    reranker = build_reranker(device="cpu")
    return embeddings, reranker


@st.cache_resource(show_spinner="Connecting to Weaviate...")
def _connect_weaviate(_cfg):
    return connect_local_weaviate(port=_cfg.weaviate_port, grpc_port=_cfg.weaviate_grpc_port)


@st.cache_resource(show_spinner="Initializing LLM assistant...")
def _load_assistant(_pipeline, _cfg):
    return RAGAssistant(
        rag_pipeline=_pipeline,
        model_name="Qwen/Qwen3-30B-A3B",
        hf_token=_cfg.hf_token,
        temperature=_cfg.llm_temperature,
        max_tokens=_cfg.llm_max_tokens,
        top_p=_cfg.llm_top_p,
        default_num_results=_cfg.num_results,
        max_retries=_cfg.llm_max_retries,
        retry_backoff_seconds=_cfg.llm_retry_backoff,
    )


def _build_pipeline(client, embeddings, reranker, cfg):
    return WeaviateRAGPipeline(
        client=client,
        embedding_model=embeddings,
        reranker_model=reranker,
        collection_name=cfg.collection_name,
        hybrid_search_limit=cfg.hybrid_search_limit,
        hybrid_alpha=cfg.hybrid_alpha,
    )


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
cfg = load_config()
embeddings, reranker = _load_models()
client = _connect_weaviate(cfg)
pipeline = _build_pipeline(client, embeddings, reranker, cfg)
assistant = _load_assistant(pipeline, cfg)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Sidebar — collection management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Collection Manager")

    has_data = pipeline.collection_has_data()
    total_chunks = pipeline.get_object_count()
    documents = pipeline.list_documents()

    col1, col2 = st.columns(2)
    col1.metric("Total chunks", total_chunks)
    col2.metric("Documents", len(documents))

    if documents:
        with st.expander("Stored documents", expanded=False):
            for doc in documents:
                st.text(doc)

    st.divider()

    st.subheader("Clear Entire Database")
    if st.button("Delete all data", type="primary", use_container_width=True):
        pipeline.delete_collection()
        st.success("Collection deleted. Re-run ingestion via CLI to repopulate.")
        st.rerun()

    st.divider()

    st.subheader("Delete a Document")
    if documents:
        selected_doc = st.selectbox("Select document", documents)
        if st.button("Delete selected document", use_container_width=True):
            deleted = pipeline.delete_document(selected_doc)
            st.success(f"Removed {deleted} chunks for **{selected_doc}**")
            st.rerun()
    else:
        st.info("No documents in the collection.")

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main area — chat interface
# ---------------------------------------------------------------------------
st.title("📋 Policy Assistant")

if not has_data:
    st.warning(
        "The vector store is empty. Run the ingestion pipeline first:\n\n"
        "```bash\npython main.py\n```"
    )

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for src in msg["sources"]:
                    st.markdown(f"**{src['title']}** — {src['heading']}  \n`{src['file_name']}`")

if prompt := st.chat_input("Ask a question about company policies..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not pipeline.collection_has_data():
            response_text = "The vector store is empty. Please ingest documents first."
            sources = []
        else:
            with st.spinner("Searching policies and generating answer..."):
                retrieval_results = pipeline.query(prompt, limit=cfg.retrieval_limit)
                sources = [
                    {
                        "title": r["metadata"]["title"],
                        "heading": r["metadata"]["heading"],
                        "file_name": r["metadata"]["file_name"],
                    }
                    for r in retrieval_results
                ]
                response_text = assistant.answer(
                    user_query=prompt, num_results=cfg.num_results
                )

        st.markdown(response_text)
        if sources:
            with st.expander("Sources"):
                for src in sources:
                    st.markdown(f"**{src['title']}** — {src['heading']}  \n`{src['file_name']}`")

    st.session_state.messages.append(
        {"role": "assistant", "content": response_text, "sources": sources}
    )
