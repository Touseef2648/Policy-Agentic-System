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

    hybrid_search_limit: int = 10
    hybrid_alpha: float = 0.65
    retrieval_limit: int = 3
    num_results: int = 3

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
    """Load RAG config from environment variables."""
    return RAGConfig(
        zip_path=os.getenv("RAG_ZIP_PATH", "devsinc-data"),
        hf_token=os.getenv("HF_TOKEN", ""),
        query_text=os.getenv(
            "RAG_QUERY", "What is the monthly limit for mobile allowance?"
        ),
        collection_name=os.getenv("RAG_COLLECTION_NAME", "PolicyDocuments"),
        hybrid_search_limit=int(os.getenv("RAG_HYBRID_SEARCH_LIMIT", "10")),
        hybrid_alpha=float(os.getenv("RAG_HYBRID_ALPHA", "0.65")),
        retrieval_limit=int(os.getenv("RAG_RETRIEVAL_LIMIT", "3")),
        num_results=int(os.getenv("RAG_NUM_RESULTS", "4")),
        llm_temperature=float(os.getenv("RAG_LLM_TEMPERATURE", "0.3")),
        llm_max_tokens=int(os.getenv("RAG_LLM_MAX_TOKENS", "1024")),
        llm_top_p=float(os.getenv("RAG_LLM_TOP_P", "0.9")),
        llm_max_retries=int(os.getenv("RAG_LLM_MAX_RETRIES", "3")),
        llm_retry_backoff=float(os.getenv("RAG_LLM_RETRY_BACKOFF_SECONDS", "1.5")),
        chunk_max_chars=int(os.getenv("RAG_CHUNK_MAX_SECTION_CHARS", "2000")),
        split_chunk_size=int(os.getenv("RAG_SPLIT_CHUNK_SIZE", "1000")),
        split_chunk_overlap=int(os.getenv("RAG_SPLIT_CHUNK_OVERLAP", "150")),
        weaviate_port=int(os.getenv("RAG_WEAVIATE_PORT", "8081")),
        weaviate_grpc_port=int(os.getenv("RAG_WEAVIATE_GRPC_PORT", "50051")),
    )

