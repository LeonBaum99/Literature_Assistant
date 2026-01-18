import re
from typing import Dict, Any, Tuple

import torch
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice
)
from docling.document_converter import DocumentConverter, PdfFormatOption


class DoclingPDFProcessor:
    """
    Handles PDF conversion and metadata extraction using Docling.
    """

    def __init__(self):
        """
        Initializes the Docling converter with GPU acceleration if available.
        """
        print("Initializing Docling Converter...")
        pipeline_options = PdfPipelineOptions()

        if torch.cuda.is_available():
            print("CUDA detected. Using GPU for PDF Processing.")
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.CUDA
            )
        else:
            print("CUDA not found. Using CPU for PDF Processing.")
            pipeline_options.accelerator_options = AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.CPU
            )

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

    def process_pdf(self, file_path: str, zotero_metadata: Dict[str, Any] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Orchestrates the conversion of a PDF file into structured metadata and sections.

        Args:
            file_path (str): The path to the PDF file.
            zotero_metadata (Dict, optional): Pre-extracted metadata from Zotero.
                If provided, uses this instead of heuristic extraction.

        Returns:
            Tuple[Dict, Dict]: A tuple containing (metadata, sections).
        """
        # 1. Convert PDF
        result = self.converter.convert(file_path)

        # 2. Extract Sections
        sections = self._extract_sections_from_doc(result.document)

        # 3. Extract Metadata
        metadata = self._extract_metadata(sections, zotero_metadata=zotero_metadata)

        return metadata, sections

    def _extract_sections_from_doc(self, doc) -> Dict[str, str]:
        """
        Parses Docling output into a section dictionary.
        """
        sections = {}
        current_header = "Preamble"
        sections[current_header] = []

        for item in doc.texts:
            text = item.text.strip()
            if not text:
                continue

            if item.label in ["section_header", "title"]:
                current_header = text
                if current_header not in sections:
                    sections[current_header] = []
            else:
                sections[current_header].append(text)

        # Join lists into single strings
        return {k: "\n".join(v) for k, v in sections.items() if v}

    def _extract_metadata(self, sections: Dict[str, str], zotero_metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extract metadata from PDF sections, with optional Zotero override.
        
        If zotero_metadata is provided, uses it as the primary source (reliable).
        Otherwise, falls back to heuristic extraction from PDF content.
        
        Args:
            sections: Extracted PDF sections
            zotero_metadata: Optional pre-extracted metadata from Zotero
        
        Returns:
            Metadata dict with title, authors, arxiv_id
        """
        # If Zotero metadata is available, use it (most reliable)
        if zotero_metadata:
            return {
                "title": zotero_metadata.get("title", "Unknown"),
                "authors": zotero_metadata.get("authors", []),
                "arxiv_id": zotero_metadata.get("arxiv_id", ""),
            }
        
        # Fallback: Heuristic extraction from PDF content
        ARXIV_PATTERN = r"arXiv:\d{4}\.\d{4,5}(v\d+)?"
        metadata = {
            "title": None,
            "authors": [],
            "arxiv_id": None
        }

        keys = list(sections.keys())

        # --- A. Extract arXiv ID from Preamble ---
        if "Preamble" in sections:
            preamble_text = sections["Preamble"]
            match = re.search(ARXIV_PATTERN, preamble_text, re.IGNORECASE)
            if match:
                metadata["arxiv_id"] = match.group(0)

        # --- B. Identify Title and Authors ---
        # Heuristic: The Title is usually the first key that is NOT 'Preamble'.
        # The Authors are usually the CONTENT of that Title key.
        title_key = None
        for key in keys:
            if key != "Preamble":
                title_key = key
                break

        if title_key:
            metadata["title"] = title_key
            author_block = sections.get(title_key, "")

            if author_block:
                raw_authors = author_block.split('\n')
                clean_authors = [a.strip() for a in raw_authors if a.strip()]
                metadata["authors"] = clean_authors

        return metadata


# --- Legacy Helper for Backward Compatibility (Optional) ---
def setup_docling_converter():
    """Returns a standalone converter instance (deprecated)."""
    processor = DoclingPDFProcessor()
    return processor.converter
