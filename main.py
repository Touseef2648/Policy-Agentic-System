"""Project entrypoint for full RAG pipeline."""

import json

from rag_project.config import load_config
from rag_project.ingestion.chunking import DocumentChunker
from rag_project.ingestion.embeddings import build_embeddings, build_reranker
from rag_project.ingestion.preprocessing import Preprocessor
from rag_project.ingestion.vector_store import WeaviateRAGPipeline, connect_local_weaviate
from rag_project.retrieval.assistant import RAGAssistant


def _ingest_documents(cfg, rag_pipeline):
    """Run the full document ingestion pipeline: preprocess, chunk, embed, and store."""
    print("[STEP 1] Preprocessing documents")
    processor = Preprocessor(input_path=cfg.zip_path)
    processed_docs = processor.run().get_documents()
    print(f"[STEP 1] Completed: {len(processed_docs)} documents processed")
    if not processed_docs:
        print("[STOP] No documents were processed. Check input path and document formats.")
        return False
    processor.print_summary(index=0)

    print("[STEP 2] Chunking parsed sections")
    chunker = DocumentChunker(
        max_section_chars=cfg.chunk_max_chars,
        split_chunk_size=cfg.split_chunk_size,
        split_chunk_overlap=cfg.split_chunk_overlap,
    )
    final_chunks = chunker.chunk_by_sections(processed_docs)
    print(f"[STEP 2] Completed: {len(final_chunks)} chunks created")
    if not final_chunks:
        print("[STOP] No chunks were created from processed documents.")
        return False
    chunker.preview_json(index=0)

    print("[STEP 3] Creating collection and storing chunks")
    rag_pipeline.create_collection()
    rag_pipeline.store_chunks(final_chunks)
    rag_pipeline.preview_stored_json(limit=5)
    print("[STEP 3] Completed")
    return True


def run_pipeline() -> None:
    """Run end-to-end RAG pipeline, reusing stored embeddings when available."""
    cfg = load_config()

    print("[START] Full Policy RAG Pipeline")

    print("[INIT] Initializing embeddings and reranker")
    embeddings = build_embeddings(device="cpu")
    reranker = build_reranker(device="cpu")
    print("[INIT] Models loaded")

    print("[INIT] Connecting to Weaviate")
    client = connect_local_weaviate(
        port=cfg.weaviate_port, grpc_port=cfg.weaviate_grpc_port
    )

    rag_pipeline = WeaviateRAGPipeline(
        client=client,
        embedding_model=embeddings,
        reranker_model=reranker,
        collection_name=cfg.collection_name,
        hybrid_search_limit=cfg.hybrid_search_limit,
        hybrid_alpha=cfg.hybrid_alpha,
    )

    if cfg.force_reindex:
        print("[INFO] Force reindex requested — deleting existing collection")
        rag_pipeline.delete_collection()

    needs_ingestion = not rag_pipeline.collection_has_data()

    if needs_ingestion:
        print("[INFO] No existing data found — running full ingestion pipeline")
        if not _ingest_documents(cfg, rag_pipeline):
            client.close()
            return
    else:
        print(
            f"[INFO] Reusing existing embeddings from collection "
            f"'{cfg.collection_name}' — skipping preprocessing, chunking, and embedding"
        )

    print("[QUERY] Running retrieval query")
    print(f"[QUERY] User query: {cfg.query_text}")
    results = rag_pipeline.query(cfg.query_text, limit=cfg.retrieval_limit)
    print("[QUERY] Retrieval response:")
    print(json.dumps(results, indent=2))

    print("[LLM] Generating final assistant answer")
    assistant = RAGAssistant(
        rag_pipeline=rag_pipeline,
        model_name="Qwen/Qwen3-30B-A3B",
        hf_token=cfg.hf_token,
        temperature=cfg.llm_temperature,
        max_tokens=cfg.llm_max_tokens,
        top_p=cfg.llm_top_p,
        default_num_results=cfg.num_results,
        max_retries=cfg.llm_max_retries,
        retry_backoff_seconds=cfg.llm_retry_backoff,
    )
    final_answer = assistant.answer(user_query=cfg.query_text, num_results=cfg.num_results)
    print("[ANSWER] Final response:")
    print(final_answer)

    client.close()
    print("[END] Pipeline finished")


if __name__ == "__main__":
    run_pipeline()
