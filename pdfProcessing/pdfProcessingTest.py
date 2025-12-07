import os
import sys

# --- TESSERACT FIX FOR CONDA/WINDOWS ---
# 1. Get the base directory of your current Conda environment
conda_base = os.environ.get('CONDA_PREFIX') or sys.prefix

# 2. Define likely paths where Conda puts the 'tessdata' folder
possible_paths = [
    os.path.join(conda_base, "Library", "bin", "tessdata"),
    os.path.join(conda_base, "Library", "share", "tessdata"),
    os.path.join(conda_base, "share", "tessdata")
]

# 3. Search for the folder and set the variable
tessdata_path = None
for path in possible_paths:
    if os.path.exists(os.path.join(path, "eng.traineddata")):
        tessdata_path = path
        break

if tessdata_path:
    os.environ['TESSDATA_PREFIX'] = tessdata_path
    print(f"✅ Found Tesseract data at: {tessdata_path}")
else:
    print("❌ Could not auto-locate 'eng.traineddata'.")
    print(f"   Checked paths: {possible_paths}")
    print("   Please verify you ran: conda install -c conda-forge tesseract")
import time
import pathlib
import json
import fitz  # PyMuPDF
import pymupdf4llm

try:
    import pymupdf.layout
except ImportError:
    pass

from unstructured.partition.pdf import partition_pdf
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.output import text_from_rendered
from docling.document_converter import DocumentConverter  # <--- NEW IMPORT

# --- Configuration ---
# REPLACE THIS with the path to your actual PDF file
INPUT_PDF = "../data/testPDFs/Attention is all you need.pdf"
BASE_OUTPUT_DIR = "../data/testPDFOutput"


def setup_directories(base_dir):
    """Creates the base output directory if it doesn't exist."""
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        print(f"Created directory: {base_dir}")
    else:
        print(f"Using existing directory: {base_dir}")


def benchmark_fitz(pdf_path, output_dir):
    """Method 1: Standard PyMuPDF (Fitz) - Raw Blocks"""
    start_time = time.time()

    doc = fitz.open(pdf_path)
    full_text = []

    for page in doc:
        # Get text blocks (preserves some structure)
        blocks = page.get_text("blocks")
        for block in blocks:
            full_text.append(block[4])  # index 4 is the text

    output_path = os.path.join(output_dir, "1_fitz_raw.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(full_text))

    duration = time.time() - start_time
    return duration, output_path


def benchmark_pymupdf4llm(pdf_path, output_dir):
    """Method 2: PyMuPDF4LLM (Layout-Enhanced) - Markdown"""
    start_time = time.time()

    images_path = os.path.join(output_dir, "images_pymupdf")
    os.makedirs(images_path, exist_ok=True)

    # pymupdf.layout is already imported globally to activate the hook
    md_text = pymupdf4llm.to_markdown(
        pdf_path,
        write_images=True,
        image_path=images_path,
        image_format="png"
    )

    output_path = os.path.join(output_dir, "2_pymupdf_layout.md")
    pathlib.Path(output_path).write_bytes(md_text.encode())

    duration = time.time() - start_time
    return duration, output_path


def benchmark_unstructured(pdf_path, output_dir):
    """Method 3: Unstructured - Elements & Text"""
    start_time = time.time()

    # strategy="hi_res" is critical for academic papers but slower
    elements = partition_pdf(
        filename=pdf_path,
        strategy="hi_res",
        infer_table_structure=True
    )

    # Reconstruct text
    text_content = "\n\n".join([str(el) for el in elements])

    output_path = os.path.join(output_dir, "3_unstructured.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)

    duration = time.time() - start_time
    return duration, output_path


def benchmark_marker(pdf_path, output_dir, converter):
    """Method 4: Marker - Deep Learning Markdown"""
    # Note: We pass the converter in to measure processing time, not model loading time.
    start_time = time.time()

    rendered = converter(pdf_path)
    text, _, images = text_from_rendered(rendered)

    # Marker output folder
    marker_dir = os.path.join(output_dir, "4_marker_output")
    os.makedirs(marker_dir, exist_ok=True)

    output_path = os.path.join(marker_dir, "output.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Save images
    for filename, image in images.items():
        image.save(os.path.join(marker_dir, filename))

    duration = time.time() - start_time
    return duration, output_path


def benchmark_docling(pdf_path, output_dir):
    """Method 5: Docling - Hybrid Layout Analysis"""
    start_time = time.time()

    # Docling initialization is generally fast enough to include,
    # but strictly speaking, we are benchmarking the conversion process.
    converter = DocumentConverter()
    result = converter.convert(pdf_path)

    # Export to markdown
    md_text = result.document.export_to_markdown()

    output_path = os.path.join(output_dir, "5_docling.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(md_text)

    duration = time.time() - start_time
    return duration, output_path


def main():
    if not os.path.exists(INPUT_PDF):
        print(f"Error: Input file '{INPUT_PDF}' not found. Please place a PDF file there or update the script.")
        return

    setup_directories(BASE_OUTPUT_DIR)

    results = []

    print("\n--- Starting Benchmark ---\n")

    # 1. Run Fitz
    print(f"Running Method 1: PyMuPDF (Standard)...")
    try:
        t1, p1 = benchmark_fitz(INPUT_PDF, BASE_OUTPUT_DIR)
        results.append({"Method": "PyMuPDF (Raw)", "Time (s)": t1, "Output": p1})
        print(f"   Done in {t1:.4f}s")
    except Exception as e:
        print(f"    PyMuPDF failed: {e}")
        results.append({"Method": "PyMuPDF (Raw)", "Time (s)": -1, "Output": "FAILED"})

    # 2. Run PyMuPDF4LLM
    print(f"Running Method 2: PyMuPDF4LLM (Layout)...")
    try:
        t2, p2 = benchmark_pymupdf4llm(INPUT_PDF, BASE_OUTPUT_DIR)
        results.append({"Method": "PyMuPDF4LLM (Layout)", "Time (s)": t2, "Output": p2})
        print(f"   Done in {t2:.4f}s")
    except Exception as e:
        print(f"    PyMuPDF4LLM failed: {e}")
        results.append({"Method": "PyMuPDF4LLM (Layout)", "Time (s)": -1, "Output": "FAILED"})

    # 3. Run Unstructured
    print(f"Running Method 3: Unstructured (Hi-Res)...")
    try:
        t3, p3 = benchmark_unstructured(INPUT_PDF, BASE_OUTPUT_DIR)
        results.append({"Method": "Unstructured", "Time (s)": t3, "Output": p3})
        print(f"   Done in {t3:.4f}s")
    except Exception as e:
        print(f"    Unstructured failed: {e}")
        results.append({"Method": "Unstructured", "Time (s)": -1, "Output": "FAILED"})

    # 4. Run Marker
    # print(f"Running Method 4: Marker (Deep Learning)...")
    # print("   Loading Marker models (excluded from timing)...")
    # # Load model outside timing loop to be fair to processing speed
    # try:
    #     converter = PdfConverter(artifact_dict=create_model_dict())
    #     t4, p4 = benchmark_marker(INPUT_PDF, BASE_OUTPUT_DIR, converter)
    #     results.append({"Method": "Marker", "Time (s)": t4, "Output": p4})
    #     print(f"   Done in {t4:.4f}s")
    # except Exception as e:
    #     print(f"    Marker failed: {e}")
    #     results.append({"Method": "Marker", "Time (s)": -1, "Output": "FAILED"})

    # 5. Run Docling
    print(f"Running Method 5: Docling...")
    try:
        t5, p5 = benchmark_docling(INPUT_PDF, BASE_OUTPUT_DIR)
        results.append({"Method": "Docling", "Time (s)": t5, "Output": p5})
        print(f"   Done in {t5:.4f}s")
    except Exception as e:
        print(f"    Docling failed: {e}")
        results.append({"Method": "Docling", "Time (s)": -1, "Output": "FAILED"})

    # --- Summary ---
    print("\n\n" + "=" * 60)
    print(f"{'METHOD':<25} | {'TIME (s)':<10} | {'OUTPUT PATH'}")
    print("-" * 60)
    for res in results:
        time_str = f"{res['Time (s)']:.4f}" if res['Time (s)'] != -1 else "FAIL"
        print(f"{res['Method']:<25} | {time_str:<10} | {res['Output']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()