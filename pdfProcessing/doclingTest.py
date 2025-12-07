import time
import torch
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    AcceleratorOptions,
    AcceleratorDevice
)


def run_docling_with_gpu(pdf_path):
    print(f"--- Docling GPU Setup ---")

    # 1. Configure the pipeline options
    pipeline_options = PdfPipelineOptions()

    # 2. Set the accelerator to CUDA (or MPS for Mac, or AUTO)
    # Options: AcceleratorDevice.CUDA, AcceleratorDevice.MPS, AcceleratorDevice.CPU, AcceleratorDevice.AUTO
    if torch.cuda.is_available():
        print("✅ CUDA detected. Using GPU.")
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CUDA
        )
    else:
        print("⚠️ CUDA not detected. Falling back to CPU.")
        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )

    # 3. Apply the options specifically for PDF files
    doc_converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

    # 4. Run Conversion
    print(f"Converting {pdf_path}...")
    start_time = time.time()

    result = doc_converter.convert(pdf_path)

    duration = time.time() - start_time
    print(f"✅ Conversion complete in {duration:.2f} seconds.")

    return result


if __name__ == "__main__":
    # Update with your path
    pdf_path = "../data/testPDFs/Attention is all you need.pdf"

    result = run_docling_with_gpu(pdf_path)

    # Just to verify output
    print(f"Extracted {len(result.document.texts)} text items.")