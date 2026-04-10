"""Central configuration for the RAG pipeline."""

import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """Runtime settings loaded from environment variables."""

    zip_path: str = "devsinc-data"
    hf_token: str = ""
    query_text: str = "What is the monthly limit for mobile allowance?"
    collection_name: str = "PolicyDocuments"
    force_reindex: bool = False

    hybrid_search_limit: int = 10
    hybrid_alpha: float = 0.65
    retrieval_limit: int = 3
    num_results: int = 4

    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024
    llm_top_p: float = 0.3
    llm_max_retries: int = 3
    llm_retry_backoff: float = 1.5

    chunk_max_chars: int = 2000
    split_chunk_size: int = 1000
    split_chunk_overlap: int = 150

    weaviate_port: int = 8081
    weaviate_grpc_port: int = 50051


def load_config() -> RAGConfig:
    """Load RAG config from environment variables with sensible defaults."""
    return RAGConfig(
        zip_path=os.getenv("RAG_ZIP_PATH", RAGConfig.zip_path),
        hf_token=os.getenv("HF_TOKEN", RAGConfig.hf_token),
        query_text=os.getenv("RAG_QUERY", RAGConfig.query_text),
        collection_name=os.getenv(
            "RAG_COLLECTION_NAME", RAGConfig.collection_name),
        force_reindex=os.getenv("RAG_FORCE_REINDEX",
                                "false").lower() in ("true", "1", "yes"),
        hybrid_search_limit=int(
            os.getenv("RAG_HYBRID_SEARCH_LIMIT", str(RAGConfig.hybrid_search_limit))),
        hybrid_alpha=float(os.getenv("RAG_HYBRID_ALPHA",
                           str(RAGConfig.hybrid_alpha))),
        retrieval_limit=int(os.getenv("RAG_RETRIEVAL_LIMIT",
                            str(RAGConfig.retrieval_limit))),
        num_results=int(os.getenv("RAG_NUM_RESULTS",
                        str(RAGConfig.num_results))),
        llm_temperature=float(
            os.getenv("RAG_LLM_TEMPERATURE", str(RAGConfig.llm_temperature))),
        llm_max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS",
                           str(RAGConfig.llm_max_tokens))),
        llm_top_p=float(os.getenv("RAG_LLM_TOP_P", str(RAGConfig.llm_top_p))),
        llm_max_retries=int(os.getenv("RAG_LLM_MAX_RETRIES",
                            str(RAGConfig.llm_max_retries))),
        llm_retry_backoff=float(
            os.getenv("RAG_LLM_RETRY_BACKOFF_SECONDS", str(RAGConfig.llm_retry_backoff))),
        chunk_max_chars=int(
            os.getenv("RAG_CHUNK_MAX_SECTION_CHARS", str(RAGConfig.chunk_max_chars))),
        split_chunk_size=int(
            os.getenv("RAG_SPLIT_CHUNK_SIZE", str(RAGConfig.split_chunk_size))),
        split_chunk_overlap=int(
            os.getenv("RAG_SPLIT_CHUNK_OVERLAP", str(RAGConfig.split_chunk_overlap))),
        weaviate_port=int(os.getenv("RAG_WEAVIATE_PORT",
                          str(RAGConfig.weaviate_port))),
        weaviate_grpc_port=int(
            os.getenv("RAG_WEAVIATE_GRPC_PORT", str(RAGConfig.weaviate_grpc_port))),
    )
