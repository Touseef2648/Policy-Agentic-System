# RAG Policy Assistant

## Project Structure

```text
.
├── data/
│   ├── raw/
│   │   └── devsinc-data.zip
│   └── processed/
├── rag_project/
│   ├── __init__.py
│   ├── config.py
│   ├── ingestion/
│   │   ├── preprocessing.py
│   │   ├── chunking.py
│   │   ├── embeddings.py
│   │   └── vector_store.py
│   └── retrieval/
│       └── assistant.py
├── main.py
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Start Weaviate with Docker:

```bash
docker compose up -d
```

3. Put your source ZIP at:

```text
data/raw/devsinc-data.zip
```

## Optional Environment Variables

Set these in your shell before running:

- `HF_TOKEN` (recommended for Hugging Face API)
- `RAG_QUERY` (example: `What is the monthly limit for mobile allowance?`)
- `RAG_ZIP_PATH` (defaults to `data/raw/devsinc-data.zip`)

Example:

```bash
export HF_TOKEN="your_token_here"
export RAG_QUERY="What is the monthly limit for mobile allowance?"
```

## Run

```bash
python main.py
```

The script prints:

- preprocessing output
- chunking output
- Weaviate stored JSON preview
- retrieval JSON for your query
- final assistant response

