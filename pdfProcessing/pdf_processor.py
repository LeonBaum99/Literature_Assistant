import re
import warnings
from difflib import SequenceMatcher
from typing import Dict, Any, Tuple, List, Optional

# Import the processors
import docling_PDF_processor
from zotero_integration.zotero_client import ZoteroClient
"""
########################################################
 DEPRECATED
########################################################
"""

class ZoteroEnrichedPDFProcessor:
    """
    A processor that combines Docling's PDF extraction with Zotero's bibliographic metadata.

    Features:
    - Robust Search: Uses API first, falls back to local collection cache for exact DOI/Title matching.
    - DOI Extraction: Scans document content if title search fails.
    - Enrichment: Overwrites Authors/Abstract and adds extra fields.
    """

    DOI_PATTERN = r'\b(10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+)\b'

    def __init__(self):
        """
        Initializes the Docling processor and the Zotero Client.
        """
        warnings.warn("The ZoteroEnrichedPDFProcessor is  DEPRECATED!!")
        self.docling_processor = docling_PDF_processor.DoclingPDFProcessor()
        self.zotero_client = ZoteroClient()

        # Local cache of collection items to bypass API search limitations
        self._collection_cache: Optional[List[Dict[str, Any]]] = None

        if self.zotero_client.collection_key:
            print(f"Zotero Client initialized for collection: {self.zotero_client.collection_name}")

    def _get_collection_cache(self) -> List[Dict[str, Any]]:
        """
        Lazy-loads all items from the target collection to ensure robust matching.
        """
        if self._collection_cache is None:
            if self.zotero_client.collection_key:
                print("[Zotero] Fetching full collection for local cache (robust lookup)...")
                # Using the client's fetch method
                self._collection_cache = self.zotero_client.fetch_collection_items(
                    self.zotero_client.collection_key
                )
                print(f"[Zotero] Cached {len(self._collection_cache)} items.")
            else:
                self._collection_cache = []
        return self._collection_cache

    def process_pdf(
            self,
            file_path: str,
            extra_fields: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, str]]:
        """
        Process a PDF and enrich it with Zotero metadata.
        """
        if extra_fields is None:
            extra_fields = []

        # 1. Docling Extraction
        print(f"\n[Docling] Processing: {file_path}")
        metadata, sections = self.docling_processor.process_pdf(file_path)

        zotero_record = None
        docling_title = metadata.get("title")

        # 2. Strategy A: Exact Title Match (API + Cache)
        if docling_title:
            print(f"[Zotero] Attempt 1: Searching by Title '{docling_title}'...")
            zotero_record = self.find_record_hybrid(query=docling_title, strategy="title")

        # 3. Strategy B: DOI Match (Extract -> API + Cache)
        if not zotero_record:
            print("[Zotero] Title search failed. Attempt 2: Scanning content for DOI...")
            extracted_doi = self.scan_for_doi(metadata, sections)

            if extracted_doi:
                print(f"[Zotero] Found DOI in document: {extracted_doi}")
                zotero_record = self.find_record_hybrid(query=extracted_doi, strategy="doi")
            else:
                print("[Zotero] No DOI found in document content.")

        # 4. Enrich
        if zotero_record:
            print(f"[Zotero] Match found: {zotero_record.get('key')} | Title: {zotero_record.get('title')}")
            self.enrich_content(metadata, sections, zotero_record, extra_fields)
        else:
            print("[Zotero] No matching item found after all attempts.")

        return metadata, sections

    def find_record_hybrid(self, query: str, strategy: str) -> Optional[Dict[str, Any]]:
        """
        Tries to find a record using a hybrid approach:
        1. API Search (fast, but sometimes unreliable for DOIs/Special chars).
        2. Local Cache Lookup (slow first time, but 100% accurate).
        """
        clean_query = query.strip()

        # --- Method 1: API Search ---
        # Note: We do NOT use 'qmode' here as it causes 400 errors on collection endpoints.
        try:
            results = []
            # Only use API search if cache isn't already loaded (optimization)
            if self._collection_cache is None:
                if self.zotero_client.collection_key:
                    results = self.zotero_client.zot.collection_items(
                        self.zotero_client.collection_key,
                        q=clean_query,
                        limit=3
                    )
                else:
                    results = self.zotero_client.zot.items(q=clean_query, limit=3)

            for item in results:
                rec = self.zotero_client.item_to_record(item)
                # Verify match to be safe
                if strategy == "doi":
                    if rec.get("DOI", "").lower() == clean_query.lower():
                        return rec
                elif strategy == "title":
                    # Simple fuzzy check
                    if SequenceMatcher(None, rec.get("title", "").lower(), clean_query.lower()).ratio() > 0.8:
                        return rec
        except Exception as e:
            print(f"[Zotero] API search warning: {e}")

        # --- Method 2: Local Cache Lookup (Fallback) ---
        # If API failed or returned irrelevant results, check the cache manually.
        print(f"[Zotero] API yielded no exact match. Checking local cache for {strategy}...")
        cache = self._get_collection_cache()

        for item in cache:
            # Skip non-literature
            if not self.zotero_client.is_literature_item(item):
                continue

            data = item.get("data", {})

            if strategy == "doi":
                # Exact DOI match (case-insensitive)
                item_doi = data.get("DOI", "")
                if item_doi and item_doi.strip().lower() == clean_query.lower():
                    return self.zotero_client.item_to_record(item)

            elif strategy == "title":
                # Exact Title match (case-insensitive)
                item_title = data.get("title", "")
                if item_title and item_title.strip().lower() == clean_query.lower():
                    return self.zotero_client.item_to_record(item)

                # High-confidence fuzzy match (optional)
                if item_title and SequenceMatcher(None, item_title.lower(), clean_query.lower()).ratio() > 0.9:
                    return self.zotero_client.item_to_record(item)

        return None

    def scan_for_doi(self, metadata: Dict[str, Any], sections: Dict[str, str]) -> Optional[str]:
        """Scans metadata and text for a DOI."""
        # Check Metadata
        for value in metadata.values():
            if isinstance(value, str):
                match = re.search(self.DOI_PATTERN, value)
                if match: return self.clean_doi(match.group(1))

        # Check Sections
        for text in sections.values():
            match = re.search(self.DOI_PATTERN, text)
            if match: return self.clean_doi(match.group(1))

        return None

    def clean_doi(self, doi: str) -> str:
        """Removes trailing punctuation often caught by regex."""
        if doi.endswith('.') or doi.endswith(','):
            return doi[:-1]
        return doi

    def enrich_content(self, metadata, sections, record, extra_fields):
        # 1. Replace Authors
        z_creators = record.get("creators", [])
        if z_creators:
            clean_authors = []
            for c in z_creators:
                if "name" in c:
                    clean_authors.append(c["name"])
                elif "firstName" in c and "lastName" in c:
                    clean_authors.append(f"{c['firstName']} {c['lastName']}")
            if clean_authors:
                metadata["authors"] = clean_authors
                print(f" -> Replaced Authors: {clean_authors}")

        # 2. Replace Abstract
        z_abstract = record.get("abstractNote")
        if z_abstract:
            metadata["abstract"] = z_abstract
            # Overwrite or create 'Abstract' section
            abstract_key = next((k for k in sections.keys() if "abstract" in k.lower()), "Abstract")
            sections[abstract_key] = z_abstract
            print(" -> Replaced Abstract")

        # 3. Extras
        enriched_count = 0
        for field in extra_fields:
            if field in record and record[field]:
                metadata[field] = record[field]
                enriched_count += 1
        if enriched_count: print(f" -> Added {enriched_count} extra fields")


if __name__ == "__main__":
    processor = ZoteroEnrichedPDFProcessor()
    # Mock usage:
    metadata, sections = processor.process_pdf(
        "../data/testPDFs/Volk and Abolhasani - 2024 - Performance metrics to unleash the power of self-driving labs in chemistry and materials science.pdf",
        ["DOI", "url", "title"])
    print("--" * 20)
    print(f"metadata: {metadata}")
    print("--" * 20)
    print(f"sections: {sections}")
