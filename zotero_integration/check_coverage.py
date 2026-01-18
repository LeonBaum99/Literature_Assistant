"""
Check which PDFs have Zotero metadata coverage.

Compares PDF filenames in data/testPDFs/ against titles in the latest Zotero export.
"""

import json
from pathlib import Path
from difflib import SequenceMatcher


def normalize_title(title: str) -> str:
    """Normalize title for fuzzy matching."""
    return title.lower().replace("‐", "-").replace("–", "-").replace("  ", " ").strip()


def similarity(a: str, b: str) -> float:
    """Compute similarity ratio between two strings."""
    return SequenceMatcher(None, a, b).ratio()


def main():
    # Load latest Zotero export
    export_dir = Path("zotero_integration/exports")
    exports = sorted(export_dir.glob("zotero_export_*.json"), 
                    key=lambda p: p.stat().st_mtime, reverse=True)
    
    if not exports:
        print("❌ No Zotero export files found!")
        return
    
    latest_export = exports[0]
    print(f"Loading: {latest_export.name}\n")
    
    with open(latest_export, "r", encoding="utf-8") as f:
        zotero_data = json.load(f)
    
    # Extract titles from Zotero
    zotero_titles = {normalize_title(item["title"]): item["title"] 
                    for item in zotero_data["items"]}
    
    print(f"Found {len(zotero_titles)} items in Zotero export\n")
    print("=" * 80)
    
    # Check PDFs
    pdf_dir = Path("data/testPDFs")
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    print(f"\nChecking {len(pdf_files)} PDFs against Zotero metadata:\n")
    
    covered = []
    not_covered = []
    
    for pdf in pdf_files:
        # Extract title from filename (format: "Author et al. - Year - Title.pdf")
        parts = pdf.stem.split(" - ", 2)
        if len(parts) == 3:
            pdf_title = parts[2]
        else:
            pdf_title = pdf.stem
        
        pdf_title_norm = normalize_title(pdf_title)
        
        # Try exact match first
        if pdf_title_norm in zotero_titles:
            covered.append(pdf.name)
            print(f"✅ {pdf.name}")
            print(f"   → Zotero: {zotero_titles[pdf_title_norm]}")
        else:
            # Try fuzzy match (>90% similarity)
            best_match = None
            best_score = 0.0
            
            for z_title_norm, z_title_orig in zotero_titles.items():
                score = similarity(pdf_title_norm, z_title_norm)
                if score > best_score:
                    best_score = score
                    best_match = z_title_orig
            
            if best_score > 0.9:
                covered.append(pdf.name)
                print(f"✅ {pdf.name}")
                print(f"   → Zotero: {best_match} ({best_score:.1%} match)")
            else:
                not_covered.append(pdf.name)
                print(f"❌ {pdf.name}")
                if best_match and best_score > 0.7:
                    print(f"   ? Best match: {best_match} ({best_score:.1%} similarity)")
        print()
    
    # Summary
    print("=" * 80)
    print(f"\nSUMMARY:")
    print(f"  PDFs with Zotero metadata: {len(covered)}/{len(pdf_files)}")
    print(f"  PDFs missing from Zotero:  {len(not_covered)}/{len(pdf_files)}")
    
    if not_covered:
        print(f"\n⚠️  Missing PDFs need to be added to Zotero:")
        for pdf in not_covered:
            print(f"     - {pdf}")
        print(f"\n📚 To add them:")
        print(f"   1. Open Zotero desktop app")
        print(f"   2. Drag these PDFs into the 'GenAI' collection")
        print(f"   3. Let Zotero extract metadata (or manually enter)")
        print(f"   4. Run: python zotero_integration/zotero.py")


if __name__ == "__main__":
    main()
