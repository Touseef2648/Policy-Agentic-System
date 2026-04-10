"""Embeddings and reranker initialization module."""

from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer


def build_embeddings(device: str = "cpu") -> HuggingFaceEmbeddings:
    """Create embedding model used for vector generation."""
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )


def build_reranker(device: str = "cpu") -> SentenceTransformer:
    """Create cross-encoder reranker model."""
    return SentenceTransformer("BAAI/bge-reranker-base", device=device)
