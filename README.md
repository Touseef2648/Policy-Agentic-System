# RAG Policy Assistant

A retrieval-augmented generation (RAG) system for querying company policy documents. Documents are parsed with **Docling**, chunked, embedded with **BGE**, stored in **Weaviate**, and answered by an LLM via the **Hugging Face Inference API**. Includes a **Streamlit** chat frontend.

## Project Structure

```text
.
├── main.py                          # CLI entrypoint — full pipeline
├── docker-compose.yml               # Weaviate vector database
├── requirements.txt
├── README.md
├── rag_project/
│   ├── __init__.py
│   ├── config.py                    # RAGConfig dataclass + env-var loader
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── preprocessing.py         # Preprocessor — unzip, parse, normalise
│   │   ├── chunking.py              # DocumentChunker — section-based splitting
│   │   ├── embeddings.py            # build_embeddings / build_reranker
│   │   └── vector_store.py          # WeaviateRAGPipeline — store & query
│   └── retrieval/
│       ├── __init__.py
│       └── assistant.py             # RAGAssistant — LLM answer generation
└── streamlit_app/
    └── app.py                       # Streamlit chat UI (decoupled frontend)
```

## Prerequisites

- **Python 3.10+**
- **Docker** (for Weaviate)
- A **Hugging Face** API token with Inference API access

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Weaviate

```bash
docker compose up -d
```

This exposes Weaviate on HTTP port **8081** and gRPC port **50051** (configurable via env vars).

### 3. Provide source documents

Place your documents (`.pdf`, `.docx`, `.pptx`) or a ZIP archive at the path configured by `RAG_ZIP_PATH` (defaults to `devsinc-data` in the project root).

### 4. Set your Hugging Face token

```bash
export HF_TOKEN="hf_your_token_here"
```

## Running the CLI Pipeline

```bash
python main.py
```

**What happens:**

1. Connects to Weaviate and checks if the collection already has data.
2. If data exists → skips ingestion, reuses stored embeddings.
3. If empty → runs full ingestion: preprocess → chunk → embed → store.
4. Performs a hybrid retrieval query with reranking.
5. Sends context + question to the LLM and prints the answer.

**Force a full re-ingestion** (deletes existing collection first):

```bash
RAG_FORCE_REINDEX=true python main.py
```

## Running the Streamlit UI

```bash
streamlit run streamlit_app/app.py
```

The web app provides:

- **Chat interface** — ask questions and get LLM-generated answers with source citations.
- **Collection Manager** (sidebar) — view stored documents, delete individual documents, or wipe the entire database.

> **Note:** The Streamlit app expects documents to already be ingested. Run `python main.py` first if the vector store is empty.

## Environment Variables

All settings have sensible defaults. Override any of them with environment variables:

| Variable | Default | Description |
|---|---|---|
| `HF_TOKEN` | `""` | Hugging Face API token |
| `RAG_ZIP_PATH` | `devsinc-data` | Path to source documents (directory or `.zip`) |
| `RAG_QUERY` | `What is the monthly limit for mobile allowance?` | Default query for CLI mode |
| `RAG_COLLECTION_NAME` | `PolicyDocuments` | Weaviate collection name |
| `RAG_FORCE_REINDEX` | `false` | Set `true` to delete and re-ingest |
| `RAG_HYBRID_SEARCH_LIMIT` | `10` | Candidates retrieved before reranking |
| `RAG_HYBRID_ALPHA` | `0.65` | Hybrid search alpha (0 = keyword, 1 = vector) |
| `RAG_RETRIEVAL_LIMIT` | `3` | Final results returned after reranking |
| `RAG_NUM_RESULTS` | `4` | Chunks included in LLM context |
| `RAG_LLM_TEMPERATURE` | `0.3` | LLM sampling temperature |
| `RAG_LLM_MAX_TOKENS` | `1024` | Max tokens in LLM response |
| `RAG_LLM_TOP_P` | `0.9` | Nucleus sampling threshold |
| `RAG_LLM_MAX_RETRIES` | `3` | Retry count for failed LLM calls |
| `RAG_LLM_RETRY_BACKOFF_SECONDS` | `1.5` | Base backoff between retries |
| `RAG_CHUNK_MAX_SECTION_CHARS` | `2000` | Section size before splitting |
| `RAG_SPLIT_CHUNK_SIZE` | `1000` | Sub-chunk size for large sections |
| `RAG_SPLIT_CHUNK_OVERLAP` | `150` | Overlap between sub-chunks |
| `RAG_WEAVIATE_PORT` | `8081` | Weaviate HTTP port |
| `RAG_WEAVIATE_GRPC_PORT` | `50051` | Weaviate gRPC port |

## Module Reference

### `rag_project.config`

- **`RAGConfig`** — Dataclass holding all pipeline settings with defaults.
- **`load_config()`** — Creates a `RAGConfig` from environment variables.

### `rag_project.ingestion.preprocessing`

- **`Preprocessor(input_path)`** — Unzips archives, discovers supported files, parses them with Docling into structured sections, and normalises text.
  - `.run()` — Executes the full preprocessing pipeline.
  - `.get_documents()` — Returns the list of parsed document dicts.
  - `.print_summary(index=0)` — Prints a debug summary for one document.

### `rag_project.ingestion.chunking`

- **`DocumentChunker(max_section_chars, split_chunk_size, split_chunk_overlap)`** — Converts preprocessed sections into chunk dicts with metadata.
  - `.chunk_by_sections(processed_documents)` — Returns a flat list of chunks.
  - `.preview_json(index=0)` — Prints chunk JSON for one document.

### `rag_project.ingestion.embeddings`

- **`build_embeddings(device="cpu")`** — Returns a `HuggingFaceEmbeddings` instance (`BAAI/bge-small-en-v1.5`).
- **`build_reranker(device="cpu")`** — Returns a `SentenceTransformer` reranker (`BAAI/bge-reranker-base`).

### `rag_project.ingestion.vector_store`

- **`connect_local_weaviate(port, grpc_port)`** — Connects to the local Weaviate instance.
- **`WeaviateRAGPipeline(client, embedding_model, ...)`** — Full vector-store interface:
  - `.collection_exists()` / `.collection_has_data()` — Introspection helpers.
  - `.create_collection()` / `.delete_collection()` — Schema management.
  - `.store_chunks(chunks)` — Embeds and stores chunks.
  - `.list_documents()` — Returns unique document file names.
  - `.delete_document(file_name)` — Removes all chunks for one document.
  - `.get_object_count()` — Returns total chunk count.
  - `.query(user_query, limit)` — Hybrid search with optional reranking.

### `rag_project.retrieval.assistant`

- **`RAGAssistant(rag_pipeline, model_name, hf_token, ...)`** — Retrieves context and generates answers via the HF Inference API.
  - `.answer(user_query, ...)` — End-to-end retrieval + generation. Automatically strips Qwen3 `<think>` blocks from the output.

### `streamlit_app.app`

Standalone Streamlit application. Run with `streamlit run streamlit_app/app.py`. Uses `rag_project` as a library — no logic is duplicated.
