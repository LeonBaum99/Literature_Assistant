"""
Zotero Client & Export Manager (Class-based)
Includes Metadata Lookup functionality via Zotero API.
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Set
from difflib import SequenceMatcher

from pyzotero import zotero

class ZoteroClient:
    """
    Handles interactions with the Zotero API, including item fetching,
    filtering, normalization, export management, and metadata lookup by filename.
    """

    def __init__(
        self,
        library_id: int = 0,
        library_type: str = "user",
        api_key: str = "",
        collection_name: str = "GenAI",
        collection_key: Optional[str] = None,
        export_dir: str = "./exports",
        update_existing: bool = False
    ):
        self.library_id = library_id
        self.library_type = library_type
        self.api_key = api_key
        self.collection_name = collection_name
        self._collection_key = collection_key
        self.export_dir = Path(export_dir)
        self.update_existing = update_existing

        self.snapshot_glob = "zotero_export_*.json"
        self.exclude_item_types: Set[str] = {"attachment", "note", "annotation"}

        # Lookup Indices
        self.metadata_by_title: Dict[str, Dict] = {}
        self.metadata_by_key: Dict[str, Dict] = {}
        self._cache_populated = False

        # Initialize Pyzotero client
        self.zot = zotero.Zotero(self.library_id, self.library_type, self.api_key)

    @property
    def collection_key(self) -> Optional[str]:
        if self._collection_key:
            return self._collection_key
        if self.collection_name:
            self._collection_key = self.resolve_collection_key_by_name(self.collection_name)
        return self._collection_key

    def resolve_collection_key_by_name(self, name: str) -> Optional[str]:
        try:
            cols = self.zot.everything(self.zot.collections())
            for c in cols:
                if (c.get("data", {}) or {}).get("name") == name:
                    return c.get("key")
        except Exception as e:
            print(f"Error resolving collection key: {e}")
        return None

    def fetch_collection_items(self, collection_key: str) -> List[Dict[str, Any]]:
        """Fetch ALL items belonging to a specific collection key."""
        return self.zot.everything(self.zot.collection_items(collection_key))

    def is_literature_item(self, item: Dict[str, Any]) -> bool:
        item_type = (item.get("data", {}) or {}).get("itemType")
        return item_type not in self.exclude_item_types

    def item_to_record(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a Zotero item to a compact JSON record."""
        data = item.get("data", {}) or {}
        creators = data.get("creators", []) or []
        norm_creators = []
        for c in creators:
            if c.get("name"):
                norm_creators.append({"name": c.get("name"), "creatorType": c.get("creatorType")})
            else:
                norm_creators.append({
                    "firstName": c.get("firstName"),
                    "lastName": c.get("lastName"),
                    "creatorType": c.get("creatorType"),
                })

        date_str = data.get("date", "")
        year = "n.d."
        if date_str:
            m = re.search(r"\b(19|20)\d{2}\b", date_str)
            if m: year = m.group(0)

        # Ensure keys required by metadata loader are present at top level
        return {
            "key": item.get("key"),
            "version": item.get("version"),
            "itemType": data.get("itemType"),
            "title": data.get("title", ""),
            "creators": norm_creators,
            "date": data.get("date"),
            "year": year,
            "abstractNote": data.get("abstractNote", ""),
            "publicationTitle": data.get("publicationTitle", ""),
            "journalAbbreviation": data.get("journalAbbreviation"),
            "DOI": data.get("DOI", ""),
            "url": data.get("url"),
            "tags": [t.get("tag") for t in (data.get("tags", []) or []) if t.get("tag")],
            "collections": data.get("collections", []) or [],
            "zoteroLink": (item.get("links", {}).get("alternate", {}) or {}).get("href"),
        }

    # --- Metadata Loader Integration ---

    def _ensure_cache(self):
        """Populate the internal metadata cache from the API if not already done."""
        if self._cache_populated:
            return

        if not self.collection_key:
            print("Warning: No collection key found. Cannot build metadata cache.")
            return

        raw_items = self.fetch_collection_items(self.collection_key)

        for raw_item in raw_items:
            # We filter for literature items (skipping attachments/notes)
            if self.is_literature_item(raw_item):
                # Normalize to record format (flat structure)
                record = self.item_to_record(raw_item)

                title = record.get("title", "")
                key = record.get("key", "")

                if title:
                    normalized_title = self._normalize_title(title)
                    self.metadata_by_title[normalized_title] = record

                if key:
                    self.metadata_by_key[key] = record

        self._cache_populated = True
        print(f"Cached {len(self.metadata_by_title)} items from Zotero API.")

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
        Lookup Zotero metadata by PDF filename using API data.

        Args:
            filename: PDF filename (e.g., "Author et al. - 2023 - Title.pdf")

        Returns:
            Dictionary with Zotero metadata fields, or None if not found.
        """
        # Ensure data is loaded from API
        self._ensure_cache()

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

        Args:
            item: The normalized record dictionary (output of item_to_record)

        Returns:
            Dict with keys: title, authors (list), arxiv_id, DOI, abstract
        """
        # Extract authors from the normalized "creators" list
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

        # Arxiv ID is not standard in item_to_record, usually found in "extra" or specific fields
        # but the original loader expected 'arxiv_id' key if present.
        # item_to_record doesn't parse 'extra', but we map what we have.

        return {
            "title": item.get("title", "Unknown"),
            "authors": authors,
            "arxiv_id": item.get("arxiv_id", ""),
            "DOI": item.get("DOI", ""),
            "year": item.get("year", ""),
            "abstract": item.get("abstractNote", ""),
            "publication": item.get("publicationTitle", ""),
            "zotero_key": item.get("key", ""),
        }


def get_metadata_for_pdf(pdf_filename: str) -> Optional[Dict]:
    """
    Convenience function to get metadata for a single PDF using the API client.

    Args:
        pdf_filename: Name of the PDF file

    Returns:
        Metadata dict or None if not found
    """
    client = ZoteroClient()
    return client.get_metadata_by_filename(pdf_filename)


if __name__ == "__main__":
    # Test
    client = ZoteroClient()
    if client.collection_key:
        print(f"Connected to collection: {client.collection_key}")
        # Test lookup if you have a known filename
        print(client.get_metadata_by_filename("../data/testPDFs/Kandel et al. - 2023 - Demonstration of an AI-driven workflow for autonomous high-resolution scanning microscopy.pdf"))