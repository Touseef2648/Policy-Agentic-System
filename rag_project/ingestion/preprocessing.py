"""Preprocessing module for policy documents."""

import os
import re
import shutil
import zipfile
from typing import Dict, List, Optional

from docling.document_converter import DocumentConverter
from docling_core.types.doc.labels import DocItemLabel


class Preprocessor:
    """
    Preprocess policy documents using Docling.

    Handles ingestion pipeline: unzip, file listing, junk removal,
    structural parsing, and text normalization.
    """

    def __init__(self, input_path: str):
        """
        Initialize preprocessor with source path and Docling converter.

        Args:
            input_path: Path to input ZIP file or directory of documents.
        """
        self.input_path: str = input_path
        self.extracted_dir: Optional[str] = None
        self.files: List[str] = []
        self.documents: List[Dict] = []
        self.converter = DocumentConverter()

    def run(self) -> "Preprocessor":
        """
        Run full preprocessing pipeline in sequence.
        """
        print("[STEP] Preprocessing: start")
        result = (
            self.unzip()
            .list_files()
            .remove_junk()
            .parse_documents()
            .normalize_text()
        )
        print("[STEP] Preprocessing: complete")
        return result

    def unzip(self) -> "Preprocessor":
        """
        Extract ZIP input into folder when input is a ZIP file.
        """
        if self.input_path.endswith(".zip"):
            extract_dir = os.path.splitext(self.input_path)[0]
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)

            with zipfile.ZipFile(self.input_path, "r") as zip_ref:
                zip_ref.extractall(extract_dir)

            self.extracted_dir = extract_dir
            print(f"[INFO] Extracted zip file to: {extract_dir}")
        else:
            self.extracted_dir = self.input_path
            print(f"[INFO] Using source directory: {self.extracted_dir}")
        return self

    def list_files(self) -> "Preprocessor":
        """
        List all supported files recursively from extracted directory.
        """
        self.files = []
        for root, _, filenames in os.walk(self.extracted_dir):
            for filename in filenames:
                if filename.lower().endswith((".pdf", ".docx", ".pptx")):
                    self.files.append(os.path.join(root, filename))

        print(f"[INFO] Found {len(self.files)} supported documents")
        return self

    def remove_junk(self) -> "Preprocessor":
        """
        Remove junk/system files from discovered file list.
        """
        junk_patterns = [".DS_Store", "Thumbs.db", "._"]
        self.files = [
            f
            for f in self.files
            if not any(
                os.path.basename(f).startswith(pattern) or os.path.basename(f) == pattern
                for pattern in junk_patterns
            )
        ]
        print(f"[INFO] Files after junk cleanup: {len(self.files)}")
        return self

    def _clean_title(self, text: str) -> str:
        """
        Clean title by removing extension/version/copy markers.
        """
        if not text:
            return ""

        text = re.sub(r"\.(pdf|docx|pptx|doc|txt)$", "", text, flags=re.IGNORECASE)
        text = re.sub(r"(?i)\s*(v|ver|version)\s*\d+$", "", text)
        text = re.sub(r"\s*\(\d+\)$", "", text)
        text = text.strip().strip("-_")
        return text

    def parse_documents(self) -> "Preprocessor":
        """
        Parse document structure into title/headings/sections using Docling.
        """
        self.documents = []
        boilerplate_header = "Policy & Standard Operating Procedure"

        for file_path in self.files:
            filename = os.path.basename(file_path)
            clean_filename_title = self._clean_title(filename)

            try:
                result = self.converter.convert(file_path)
                doc = result.document

                extracted_title = None
                headings_list = []
                sections = []
                current_heading = "General Information"
                current_content = []

                for item, _level in doc.iterate_items():
                    if item.label == DocItemLabel.TITLE:
                        if item.text.strip() == boilerplate_header:
                            continue
                        extracted_title = self._clean_title(item.text)
                        current_heading = extracted_title
                    elif item.label == DocItemLabel.SECTION_HEADER:
                        if item.text.strip() == boilerplate_header:
                            continue
                        if extracted_title and item.text.strip() == extracted_title:
                            continue
                        if current_content:
                            sections.append(
                                {
                                    "heading": current_heading,
                                    "raw_content": "\n".join(current_content),
                                }
                            )
                            current_content = []
                        current_heading = item.text
                        headings_list.append(item.text)
                    elif item.label == DocItemLabel.TABLE:
                        current_content.append(item.export_to_markdown(doc=doc))
                    elif item.label in [
                        DocItemLabel.PARAGRAPH,
                        DocItemLabel.LIST_ITEM,
                        DocItemLabel.TEXT,
                    ]:
                        current_content.append(item.text)

                if current_content:
                    sections.append(
                        {
                            "heading": current_heading,
                            "raw_content": "\n".join(current_content),
                        }
                    )

                final_title = extracted_title if extracted_title else clean_filename_title
                self.documents.append(
                    {
                        "file_name": filename,
                        "file_path": file_path,
                        "title": final_title,
                        "all_headings": headings_list,
                        "sections": sections,
                    }
                )
            except Exception as error:
                print(f"[ERROR] Failed to parse {filename}: {error}")

        print(
            "[INFO] Parsed "
            f"{len(self.documents)} docs. Boilerplate headers were suppressed."
        )
        return self

    def normalize_text(self) -> "Preprocessor":
        """
        Normalize section text by removing boilerplate/junk and collapsing spaces.
        """
        keep_chars = r""".?!,:;–—()[]{}'"\/-"""
        junk_punct_pattern = f"[^{re.escape(keep_chars)}\\w\\s]"

        for doc in self.documents:
            for section in doc["sections"]:
                text = section["raw_content"]
                text = re.sub(
                    r"(?i)(confidential|internal use only|page \d+ of \d+|draft|version \d+)",
                    "",
                    text,
                )
                text = re.sub(junk_punct_pattern, "", text)
                text = re.sub(r"\s+", " ", text).strip()
                section["clean_text"] = text

        print("[INFO] Smart text normalization completed")
        return self

    def get_documents(self) -> List[Dict]:
        """
        Return final structured and normalized documents.
        """
        return self.documents

    def print_summary(self, index: int = 0, file_name: Optional[str] = None) -> None:
        """
        Print detailed inspection summary for one parsed document.
        """
        if not self.documents:
            print("[WARN] No documents have been parsed yet")
            return

        target_doc = None
        if file_name:
            target_doc = next(
                (d for d in self.documents if file_name.lower() in d["file_name"].lower()),
                None,
            )
            if not target_doc:
                print(f"[WARN] Could not find document containing: '{file_name}'")
                return
        else:
            if index < len(self.documents):
                target_doc = self.documents[index]
            else:
                print(f"[WARN] Index {index} is out of range. Total docs: {len(self.documents)}")
                return

        print(f"[DOC] Document: {target_doc['file_name']}")
        print(f"[DOC] Cleaned title: {target_doc['title']}")
        print(f"[DOC] Total sections: {len(target_doc['sections'])}")
        headings_preview = ", ".join(target_doc["all_headings"][:10])
        if len(target_doc["all_headings"]) > 10:
            headings_preview += "..."
        print(f"[DOC] Headings preview: {headings_preview}")
        print("[DOC] First 3 sections preview:")
        for idx, section in enumerate(target_doc["sections"][:3], start=1):
            snippet = section["clean_text"][:300].replace("\n", " ")
            print(f"  - Section {idx}: {section['heading']}")
            print(f"    Text: {snippet}...")