import os
import glob
import json
import re
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice
)

# --- 1. Regex Configuration ---
# Matches: arXiv:1706.03762 or arXiv:1706.03762v1
# TODO: turn into class
def setup_docling_converter():
    """Configures Docling with GPU if available."""
    pipeline_options = PdfPipelineOptions()
    if torch.cuda.is_available():
        print("✅ CUDA detected. Using GPU.")
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CUDA
        )
    else:
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4, device=AcceleratorDevice.CPU
        )

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


def extract_sections_from_doc(doc):
    """Parses Docling output into a section dictionary."""
    sections = {}
    current_header = "Preamble"
    sections[current_header] = []

    for item in doc.texts:
        text = item.text.strip()
        if not text: continue

        if item.label in ["section_header", "title"]:
            current_header = text
            if current_header not in sections:
                sections[current_header] = []
        else:
            sections[current_header].append(text)

    # Join lists into single strings
    return {k: "\n".join(v) for k, v in sections.items() if v}


def extract_metadata(sections):
    """
    Heuristic function to extract Metadata (Title, Authors, arXiv ID).
    """
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
            metadata["arxiv_id"] = match.group(0)  # e.g. "arXiv:1706.03762v7"

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

        # Get the content associated with the title (which contains authors)
        author_block = sections.get(title_key, "")

        # Split by newline and clean up
        if author_block:
            # splitting by \n as requested
            raw_authors = author_block.split('\n')

            # Filter out empty strings and potential affiliations (optional)
            # For now, we take everything that isn't empty
            clean_authors = [a.strip() for a in raw_authors if a.strip()]
            metadata["authors"] = clean_authors

    return metadata


def main():
    input_folder = "../data/testPDFs"
    output_folder = "../data/testPDFOutput"
    os.makedirs(output_folder, exist_ok=True)

    converter = setup_docling_converter()
    pdf_files = glob.glob(os.path.join(input_folder, "*.pdf"))

    print(f"🚀 Processing {len(pdf_files)} PDFs...")

    for pdf_path in pdf_files:
        file_stem = os.path.splitext(os.path.basename(pdf_path))[0]

        try:
            # 1. Convert
            result = converter.convert(pdf_path)

            # 2. Get Sections
            sections = extract_sections_from_doc(result.document)

            # 3. Extract Specific Metadata
            metadata = extract_metadata(sections)

            # 4. Construct Final JSON Structure
            final_output = {
                "filename": os.path.basename(pdf_path),
                "metadata": metadata,
                "sections": sections
            }

            # 5. Save
            out_path = os.path.join(output_folder, f"{file_stem}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(final_output, f, indent=2)

            print(f"✅ Processed: {file_stem}")
            print(f"   found ID: {metadata.get('arxiv_id')}")
            print(f"   found {len(metadata.get('authors', []))} authors")

        except Exception as e:
            print(f"❌ Failed {file_stem}: {e}")


if __name__ == "__main__":
    main()