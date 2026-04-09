"""Chunking module for structured policy documents."""

import json
from typing import Dict, List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunker:
    """
    Convert preprocessed document sections into chunk objects for Weaviate.
    """

    def __init__(
        self,
        max_section_chars: int = 2000,
        split_chunk_size: int = 1000,
        split_chunk_overlap: int = 150,
    ):
        """
        Initialize in-memory chunk storage.

        Args:
            max_section_chars: Threshold above which a section is split.
            split_chunk_size: Character size of each fallback sub-chunk.
            split_chunk_overlap: Overlap size between consecutive sub-chunks.
        """
        self.all_chunks: List[Dict] = []
        self.max_section_chars = max_section_chars
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=split_chunk_size,
            chunk_overlap=split_chunk_overlap,
        )

    def _split_if_oversized(self, text: str) -> List[str]:
        """
        Split oversized section text into overlapping sub-chunks.
        """
        if len(text) <= self.max_section_chars:
            return [text]
        return self.splitter.split_text(text)

    def chunk_by_sections(self, processed_documents: List[Dict]) -> List[Dict]:
        """
        Build one chunk per heading-text section with metadata.
        """
        self.all_chunks = []

        for doc in processed_documents:
            doc_chunks: List[Dict] = []
            for index, section in enumerate(doc["sections"]):
                sub_chunks = self._split_if_oversized(section["clean_text"])
                for part_index, sub_text in enumerate(sub_chunks):
                    chunk = {
                        "text": sub_text,
                        "metadata": {
                            "title": doc["title"] or doc["file_name"],
                            "heading": section["heading"],
                            "file_name": doc["file_name"],
                            "chunk_index": 0,  # filled after per-doc chunk creation
                            "total_chunks": 0,  # filled after per-doc chunk creation
                            "section_index": index,
                            "section_part_index": part_index,
                            "section_parts_total": len(sub_chunks),
                        },
                    }
                    doc_chunks.append(chunk)

            for chunk_index, chunk in enumerate(doc_chunks):
                chunk["metadata"]["chunk_index"] = chunk_index
                chunk["metadata"]["total_chunks"] = len(doc_chunks)

            self.all_chunks.extend(doc_chunks)

        print(
            "[STEP] Chunking complete: "
            f"{len(self.all_chunks)} chunks from {len(processed_documents)} documents"
        )
        return self.all_chunks

    def preview_json(self, file_name: Optional[str] = None, index: int = 0) -> None:
        """
        Print chunk JSON for one selected document to validate metadata flow.
        """
        if not self.all_chunks:
            print("[WARN] No chunks available. Run chunk_by_sections first")
            return

        if file_name:
            selected_chunks = [
                chunk
                for chunk in self.all_chunks
                if file_name.lower() in chunk["metadata"]["file_name"].lower()
            ]
        else:
            unique_files = []
            for chunk in self.all_chunks:
                chunk_file = chunk["metadata"]["file_name"]
                if chunk_file not in unique_files:
                    unique_files.append(chunk_file)

            if index < len(unique_files):
                target_file = unique_files[index]
                selected_chunks = [
                    chunk
                    for chunk in self.all_chunks
                    if chunk["metadata"]["file_name"] == target_file
                ]
            else:
                print(f"[WARN] Index {index} out of range. Only {len(unique_files)} documents found")
                return

        if not selected_chunks:
            print(f"[WARN] No chunks found for: {file_name if file_name else index}")
            return

        print(
            "[DEBUG] Chunk JSON preview for document: "
            f"{selected_chunks[0]['metadata']['file_name']}"
        )
        print(json.dumps(selected_chunks, indent=2))
        print(f"[DEBUG] Total chunks shown: {len(selected_chunks)}")

