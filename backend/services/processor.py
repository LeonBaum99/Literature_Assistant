# Updated Import
from pdfProcessing.docling_PDF_processor import DoclingPDFProcessor

class PDFProcessorService:
    def __init__(self):
        self.processor = DoclingPDFProcessor()

    def process_pdf(self, file_path: str):
        """
        Delegates processing to the DoclingPDFProcessor.
        """
        metadata, sections = self.processor.process_pdf(file_path)
        return metadata, sections