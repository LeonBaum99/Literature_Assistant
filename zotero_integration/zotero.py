"""
Zotero incremental JSON export (no separate cache file) — COLLECTION-ONLY + LITERATURE-ONLY

What it does:
- Exports ONLY items that are inside the collection "GenAI"
- Filters out non-literature items (attachments, notes, annotations)
- Looks for the latest existing snapshot JSON in ./exports (zotero_export_*.json)
- Uses that latest snapshot as the baseline (so it only adds new items by Zotero item 'key')
- Writes a NEW dated snapshot JSON: ./exports/zotero_export_YYYYMMDD_HHMMSS.json
"""

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional

from pyzotero import zotero

# ---------------------------
# CONFIGURATION
# ---------------------------
LIBRARY_ID = 19245007
LIBRARY_TYPE = "user"  # "user" or "group"

# API key hardcoded as requested
API_KEY = "6IwBVbB20r1Yz9TTXbJ9hCOq"

# Export scope: ONLY this collection
COLLECTION_NAME: Optional[str] = "GenAI"   # <-- change if your collection has a different name
COLLECTION_KEY: Optional[str] = None       # optional override if you know the key (takes precedence)

EXPORT_DIR = Path("./exports")
SNAPSHOT_GLOB = "zotero_export_*.json"

# If True, update existing records when Zotero's 'version' changed.
# You asked for "only add new", so default is False.
UPDATE_EXISTING_ITEMS = False

# Filter: exclude non-literature item types
EXCLUDE_ITEM_TYPES = {"attachment", "note", "annotation"}


# ---------------------------
# ZOTERO CONNECTION
# ---------------------------
zot = zotero.Zotero(LIBRARY_ID, LIBRARY_TYPE, API_KEY)


# ---------------------------
# HELPERS
# ---------------------------
def extract_year(date_str: str) -> str:
    """Extract a 4-digit year from Zotero's 'date' field if possible."""
    if not date_str:
        return "n.d."
    m = re.search(r"\b(19|20)\d{2}\b", date_str)
    return m.group(0) if m else "n.d."


def safe_get(d: Dict[str, Any], *keys: str, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def is_literature_item(item: Dict[str, Any]) -> bool:
    """Return True if item is a 'main' bibliographic record (not attachment/note/annotation)."""
    item_type = (item.get("data", {}) or {}).get("itemType")
    return item_type not in EXCLUDE_ITEM_TYPES


def item_to_record(item: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a Zotero API item into a compact JSON record."""
    data = item.get("data", {}) or {}

    creators = data.get("creators", []) or []
    norm_creators = []
    for c in creators:
        if c.get("name"):
            norm_creators.append({"name": c.get("name"), "creatorType": c.get("creatorType")})
        else:
            norm_creators.append(
                {
                    "firstName": c.get("firstName"),
                    "lastName": c.get("lastName"),
                    "creatorType": c.get("creatorType"),
                }
            )

    return {
        "key": item.get("key"),  # unique per item, used for dedup
        "version": item.get("version"),
        "itemType": data.get("itemType"),
        "title": data.get("title"),
        "creators": norm_creators,
        "date": data.get("date"),
        "year": extract_year(data.get("date", "") or ""),
        "abstractNote": data.get("abstractNote"),
        "publicationTitle": data.get("publicationTitle"),
        "journalAbbreviation": data.get("journalAbbreviation"),
        "DOI": data.get("DOI"),
        "url": data.get("url"),
        "tags": [t.get("tag") for t in (data.get("tags", []) or []) if t.get("tag")],
        "collections": data.get("collections", []) or [],
        "dateAdded": data.get("dateAdded"),
        "dateModified": data.get("dateModified"),
        "zoteroLink": safe_get(item, "links", "alternate", "href", default=None),
    }


def latest_snapshot_path(export_dir: Path) -> Optional[Path]:
    files = sorted(export_dir.glob(SNAPSHOT_GLOB), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def load_snapshot(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)


def resolve_collection_key_by_name(name: str) -> Optional[str]:
    """
    Try to find a collection key by its name.
    If multiple collections share the same name, returns the first match.
    """
    cols = zot.everything(zot.collections())
    for c in cols:
        if (c.get("data", {}) or {}).get("name") == name:
            return c.get("key")
    return None


def fetch_collection_items(collection_key: str) -> List[Dict[str, Any]]:
    """Fetch items ONLY from the given collection key."""
    return zot.everything(zot.collection_items(collection_key))


# ---------------------------
# MAIN WORKFLOW
# ---------------------------
def run_export() -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Determine collection key (required in this version)
    collection_key = COLLECTION_KEY
    if not collection_key:
        if not COLLECTION_NAME:
            raise RuntimeError("This script is collection-only. Set COLLECTION_NAME or COLLECTION_KEY.")
        collection_key = resolve_collection_key_by_name(COLLECTION_NAME)
        if not collection_key:
            raise RuntimeError(
                f"Could not resolve collection name '{COLLECTION_NAME}'. "
                f"Set COLLECTION_KEY explicitly or check the collection name."
            )
        print(f"Resolved collection name '{COLLECTION_NAME}' -> key '{collection_key}'")

    # Load baseline from newest snapshot (if any)
    baseline_path = latest_snapshot_path(EXPORT_DIR)
    items_by_key: Dict[str, Dict[str, Any]] = {}

    if baseline_path:
        baseline = load_snapshot(baseline_path)
        baseline_items = baseline.get("items", []) or []
        for rec in baseline_items:
            k = rec.get("key")
            if k:
                items_by_key[k] = rec
        print(f"Loaded baseline snapshot: {baseline_path.name} ({len(items_by_key)} items)")
    else:
        print("No baseline snapshot found. Starting from empty baseline.")

    # Fetch current items from the collection and filter out attachments/notes
    print(f"Fetching items from Zotero collection '{COLLECTION_NAME}'...")
    raw_items = fetch_collection_items(collection_key)
    current_items = [it for it in raw_items if is_literature_item(it)]

    print(f"Fetched {len(raw_items)} items from collection (raw).")
    print(f"Keeping {len(current_items)} literature items (excluding attachments/notes/annotations).")

    new_count = 0
    updated_count = 0
    skipped_count = 0

    for item in current_items:
        k = item.get("key")
        if not k:
            continue

        record = item_to_record(item)

        if k not in items_by_key:
            items_by_key[k] = record
            new_count += 1
        else:
            if UPDATE_EXISTING_ITEMS:
                old_version = items_by_key[k].get("version")
                if record.get("version") != old_version:
                    items_by_key[k] = record
                    updated_count += 1
                else:
                    skipped_count += 1
            else:
                skipped_count += 1

    # Write new snapshot
    now = datetime.now()
    stamp = now.strftime("%Y%m%d_%H%M%S")
    snapshot_file = EXPORT_DIR / f"zotero_export_{stamp}.json"

    snapshot = {
        "schema": "zotero-export-snapshot-v1",
        "generatedAt": now.isoformat(timespec="seconds"),
        "library": {"libraryType": LIBRARY_TYPE, "libraryId": LIBRARY_ID},
        "collectionKey": collection_key,
        "collectionName": COLLECTION_NAME,
        "filters": {
            "excludeItemTypes": sorted(EXCLUDE_ITEM_TYPES),
        },
        "stats": {
            "baselineItems": len(items_by_key) - new_count - (updated_count if UPDATE_EXISTING_ITEMS else 0),
            "fetchedRawFromCollection": len(raw_items),
            "keptLiteratureItems": len(current_items),
            "newAdded": new_count,
            "updated": updated_count if UPDATE_EXISTING_ITEMS else 0,
            "skippedExisting": skipped_count,
            "totalNow": len(items_by_key),
        },
        "items": sorted(
            items_by_key.values(),
            key=lambda r: (r.get("year") or "", r.get("title") or "", r.get("key") or ""),
        ),
    }

    save_json(snapshot_file, snapshot)

    print("\nDone.")
    print(f"New items added this run: {new_count}")
    if UPDATE_EXISTING_ITEMS:
        print(f"Existing items updated: {updated_count}")
    print(f"Snapshot written: {snapshot_file}")


if __name__ == "__main__":
    run_export()