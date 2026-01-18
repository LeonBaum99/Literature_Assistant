"""
Zotero Metadata Loader

Loads Zotero export JSON and provides metadata lookup by PDF filename.
This ensures reliable title/author extraction instead of heuristic PDF parsing.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from difflib import SequenceMatcher


class ZoteroMetadataLoader:
    """
    Loads and indexes Zotero metadata for PDF files.
    
    Provides lookup by filename with fuzzy matching fallback.
    """
    
    def __init__(self, export_json_path: Optional[str] = None):
        """
        Initialize the metadata loader.
        
        Args:
            export_json_path: Path to Zotero export JSON. If None, uses latest export.
        """
        if export_json_path is None:
            export_json_path = self._get_latest_export()
        
        self.export_path = Path(export_json_path)
        self.metadata_by_title = {}
        self.metadata_by_key = {}
        self._load_metadata()
    
    def _get_latest_export(self) -> str:
        """Find the most recent Zotero export file."""
        export_dir = Path(__file__).parent / "exports"
        exports = sorted(export_dir.glob("zotero_export_*.json"), 
                        key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not exports:
            raise FileNotFoundError(
                f"No Zotero export files found in {export_dir}. "
                f"Run 'python zotero_integration/zotero.py' first."
            )
        
        return str(exports[0])
    
    def _load_metadata(self):
        """Load and index the Zotero export."""
        with open(self.export_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        for item in data.get("items", []):
            title = item.get("title", "")
            key = item.get("key", "")
            
            if title:
                # Normalize title for matching
                normalized_title = self._normalize_title(title)
                self.metadata_by_title[normalized_title] = item
            
            if key:
                self.metadata_by_key[key] = item
        
        print(f"Loaded {len(self.metadata_by_title)} items from {self.export_path.name}")
    
    @staticmethod
    def _normalize_title(title: str) -> str:
        """Normalize title for fuzzy matching."""
        return title.lower().replace("‐", "-").replace("–", "-").replace("  ", " ").strip()
    
    @staticmethod
    def _extract_title_from_filename(filename: str) -> str:
        """
        Extract title from Zotero-style filename.
        
        Expected format: "Author et al. - Year - Title.pdf"
        """
        # Remove .pdf extension
        name = filename.replace(".pdf", "").replace(".PDF", "")
        
        # Split by " - " (Zotero's separator)
        parts = name.split(" - ", 2)
        
        if len(parts) == 3:
            # Format: "Author et al. - Year - Title"
            return parts[2]
        else:
            # Fallback: use entire filename
            return name
    
    @staticmethod
    def _similarity(a: str, b: str) -> float:
        """Compute similarity ratio between two strings."""
        return SequenceMatcher(None, a, b).ratio()
    
    def get_metadata_by_filename(self, filename: str) -> Optional[Dict]:
        """
        Lookup Zotero metadata by PDF filename.
        
        Args:
            filename: PDF filename (e.g., "Author et al. - 2023 - Title.pdf")
        
        Returns:
            Dictionary with Zotero metadata fields, or None if not found.
        """
        # Extract title from filename
        pdf_title = self._extract_title_from_filename(filename)
        pdf_title_norm = self._normalize_title(pdf_title)
        
        # Try exact match first
        if pdf_title_norm in self.metadata_by_title:
            item = self.metadata_by_title[pdf_title_norm]
            return self._format_metadata(item)
        
        # Try fuzzy match (>90% similarity)
        best_match = None
        best_score = 0.0
        
        for zotero_title_norm, item in self.metadata_by_title.items():
            score = self._similarity(pdf_title_norm, zotero_title_norm)
            if score > best_score:
                best_score = score
                best_match = item
        
        if best_score > 0.9:
            return self._format_metadata(best_match)
        
        # No match found
        return None
    
    def _format_metadata(self, item: Dict) -> Dict:
        """
        Format Zotero item into metadata dict compatible with DoclingPDFProcessor.
        
        Returns:
            Dict with keys: title, authors (list), arxiv_id, DOI, abstract
        """
        # Extract authors
        authors = []
        for creator in item.get("creators", []):
            if "name" in creator:
                # Corporate/institutional author
                authors.append(creator["name"])
            else:
                # Personal author
                first = creator.get("firstName", "")
                last = creator.get("lastName", "")
                if first and last:
                    authors.append(f"{first} {last}")
                elif last:
                    authors.append(last)
        
        return {
            "title": item.get("title", "Unknown"),
            "authors": authors,
            "arxiv_id": item.get("arxiv_id", ""),  # May be empty
            "DOI": item.get("DOI", ""),
            "year": item.get("year", ""),
            "abstract": item.get("abstractNote", ""),
            "publication": item.get("publicationTitle", ""),
            "zotero_key": item.get("key", ""),
        }


def get_metadata_for_pdf(pdf_filename: str, export_path: Optional[str] = None) -> Optional[Dict]:
    """
    Convenience function to get metadata for a single PDF.
    
    Args:
        pdf_filename: Name of the PDF file
        export_path: Optional path to Zotero export JSON
    
    Returns:
        Metadata dict or None if not found
    """
    loader = ZoteroMetadataLoader(export_path)
    return loader.get_metadata_by_filename(pdf_filename)
