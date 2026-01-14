"""
Zotero Client & Export Manager (Class-based)
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Set

from pyzotero import zotero

class ZoteroClient:
    """
    Handles interactions with the Zotero API, including item fetching,
    filtering, normalization, and export management.
    """

    def __init__(
        self,
        library_id: int = 19245007,
        library_type: str = "user",
        api_key: str = "6IwBVbB20r1Yz9TTXbJ9hCOq",
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

        return {
            "key": item.get("key"),
            "version": item.get("version"),
            "itemType": data.get("itemType"),
            "title": data.get("title"),
            "creators": norm_creators,
            "date": data.get("date"),
            "year": year,
            "abstractNote": data.get("abstractNote"),
            "publicationTitle": data.get("publicationTitle"),
            "journalAbbreviation": data.get("journalAbbreviation"),
            "DOI": data.get("DOI"),
            "url": data.get("url"),
            "tags": [t.get("tag") for t in (data.get("tags", []) or []) if t.get("tag")],
            "collections": data.get("collections", []) or [],
            "zoteroLink": (item.get("links", {}).get("alternate", {}) or {}).get("href"),
        }

if __name__ == "__main__":
    # Test
    client = ZoteroClient()
    if client.collection_key:
        print(f"Connected to collection: {client.collection_key}")