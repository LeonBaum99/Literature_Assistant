from pdfProcessing.doclingTest import setup_docling_converter, extract_sections_from_doc, extract_metadata

class PDFProcessorService:
    def __init__(self):
        print("🔧 Initializing PDF Converter...")
        self.converter = setup_docling_converter()

    def process_pdf(self, file_path: str):
        """
        Returns a structured dictionary with metadata and sections.
        """
        result = self.converter.convert(file_path)
        sections = extract_sections_from_doc(result.document)
        metadata = extract_metadata(sections)
        return metadata, sections