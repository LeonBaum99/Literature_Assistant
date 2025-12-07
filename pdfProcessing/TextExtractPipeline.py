import os
import sys
import json
import fitz  # PyMuPDF
import pymupdf4llm

try:
    import pymupdf.layout
except ImportError:
    pass  # Graceful fallback if layout isn't licensed/installed

# Unstructured imports
from unstructured.partition.text import partition_text
from unstructured.cleaners.core import group_broken_paragraphs, clean_bullets

# --- TESSERACT FIX FOR WINDOWS/CONDA ---
# Only needed if using Unstructured's OCR features, but good to have safety
conda_base = os.environ.get('CONDA_PREFIX') or sys.prefix
possible_tess_paths = [
    os.path.join(conda_base, "Library", "bin", "tessdata"),
    os.path.join(conda_base, "share", "tessdata")
]
for path in possible_tess_paths:
    if os.path.exists(os.path.join(path, "eng.traineddata")):
        os.environ['TESSDATA_PREFIX'] = path
        break


# ---------------------------------------

def process_with_unstructured(pdf_path):
    """
    Hybrid Approach: Uses PyMuPDF for extraction -> Unstructured for NLP classification.
    Pros: Fast, granular text labeling (Narrative vs Title).
    Cons: Loses table structure (sees tables as text).
    """
    print("   ↳ Running Unstructured Hybrid Pipeline...")
    doc = fitz.open(pdf_path)
    structured_data = []

    for page_num, page in enumerate(doc):
        # 1. Extract raw text blocks (preserves column order)
        blocks = page.get_text("blocks", sort=True)
        page_text = ""
        for block in blocks:
            if block[6] == 0:  # Text only
                page_text += block[4] + "\n"

        # 2. Semantic Partitioning
        # We classify text based on structure (caps, length, bullets)
        elements = partition_text(text=page_text, min_partition=64)

        for element in elements:
            # Clean up the text
            clean_text = group_broken_paragraphs(element.text)

            structured_data.append({
                "page": page_num + 1,
                "type": element.category,  # e.g. "NarrativeText", "Title", "ListItem"
                "content": clean_text,
                "metadata": {"method": "unstructured_hybrid"}
            })

    return structured_data


def process_with_pymupdf4llm(pdf_path, output_dir):
    """
    Layout Approach: Uses PyMuPDF4LLM + Layout Analysis.
    Includes strict type-checking to prevent 'int is not iterable' errors.
    """
    print("   ↳ Running PyMuPDF4LLM Layout Pipeline...")

    img_path = os.path.join(output_dir, "extracted_images")
    os.makedirs(img_path, exist_ok=True)

    try:
        # 1. Extract Chunks
        chunks = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
            write_images=True,
            image_path=img_path,
            image_format="png"
        )
    except Exception as e:
        print(f"   ❌ Critical Error during to_markdown: {e}")
        return []

    structured_data = []

    for chunk in chunks:
        # Safety check for metadata
        if not isinstance(chunk, dict): continue
        page_num = chunk.get('metadata', {}).get('page', 0)

        # --- A. Process Tables (Safe Mode) ---
        for table in chunk.get('tables', []):
            # 1. Try to find pre-rendered markdown
            table_md = table.get('markdown') or table.get('content') or table.get('text')

            # 2. If no markdown, try to build from rows -- BUT CHECK TYPES FIRST
            if not table_md:
                rows = table.get('rows')
                # Strict Check: "rows" must be a list, and the first item must be a list (not an int)
                if isinstance(rows, list) and len(rows) > 0 and isinstance(rows[0], (list, tuple)):
                    try:
                        table_md = "\n".join(["| " + " | ".join(map(str, row)) + " |" for row in rows])
                    except Exception:
                        table_md = "[Error building table from rows]"
                else:
                    # If rows contains ints or is empty, we skip reconstruction
                    table_md = "[Table detected but content structure is complex/missing]"

            structured_data.append({
                "page": page_num,
                "type": "Table",
                "content": table_md,
                "metadata": {"raw_table_keys": list(table.keys()), "method": "pymupdf4llm"}
            })

        # --- B. Process Images (Safe Mode) ---
        for i, img in enumerate(chunk.get('images', [])):
            if not isinstance(img, dict): continue

            img_name = img.get('name', img.get('filename'))
            if not img_name:
                img_name = f"image_p{page_num}_{i}.png"

            structured_data.append({
                "page": page_num,
                "type": "Image",
                "content": f"Image extracted: {img_name}",
                "metadata": {"method": "pymupdf4llm"}
            })

        # --- C. Process Text ---
        # Fallback to empty string if 'text' key is missing
        raw_text = chunk.get('text', '')
        text_lines = raw_text.split('\n')

        for line in text_lines:
            line = line.strip()
            if not line: continue

            if line.startswith('#'):
                b_type = "Title"
                content = line.lstrip('#').strip()
            elif line.startswith(('-', '*', '1.')):
                b_type = "ListItem"
                content = line.lstrip('-*1. ').strip()
            elif line.startswith('|'):
                continue
            else:
                b_type = "NarrativeText"
                content = line

            structured_data.append({
                "page": page_num,
                "type": b_type,
                "content": content,
                "metadata": {"method": "pymupdf4llm"}
            })

    return structured_data


def extract_semantic_blocks(pdf_path, method="pymupdf4llm", output_dir="../data/testPDFOutput"):
    """
    Master function to extract semantic blocks using the chosen backend.

    Args:
        pdf_path (str): Path to PDF.
        method (str): "pymupdf4llm" (default) or "unstructured".
        output_dir (str): Folder to save results.
    """
    if not os.path.exists(pdf_path):
        return {"error": "File not found"}

    os.makedirs(output_dir, exist_ok=True)
    print(f"--- Processing {os.path.basename(pdf_path)} using [{method.upper()}] ---")

    if method.lower() == "unstructured":
        blocks = process_with_unstructured(pdf_path)
    elif method.lower() == "pymupdf4llm":
        blocks = process_with_pymupdf4llm(pdf_path, output_dir)
    else:
        raise ValueError("Method must be 'unstructured' or 'pymupdf4llm'")

    # Save Result
    out_file = os.path.join(output_dir, f"output_{method}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(blocks, f, indent=2)

    print(f"✅ Extracted {len(blocks)} blocks.")
    print(f"   Saved to: {out_file}\n")
    return blocks


# --- Usage Example ---
if __name__ == "__main__":
    pdf_file = "../data/testPDFs/Attention is all you need.pdf"

    # 1. Run with PyMuPDF4LLM (Best for tables/layout)
    extract_semantic_blocks(pdf_file, method="pymupdf4llm")

    # 2. Run with Unstructured (Best for pure text classification)
    extract_semantic_blocks(pdf_file, method="unstructured")