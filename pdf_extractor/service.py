from typing import Optional, Callable

from .config import ExtractorConfig
from .extractor import PDFExtractor
from .models import ExtractionResult
from .storage import ExtractionStorage


class ExtractionService:
    def __init__(self, config: ExtractorConfig | None = None):
        self.config = config or ExtractorConfig()
        self.extractor = PDFExtractor(config=self.config.processing)
        self.storage = ExtractionStorage(self.config.data_dir)

    def extract(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> ExtractionResult:
        return self.extractor.extract(pdf_path, progress_callback=progress_callback)

    def extract_and_save(
        self,
        pdf_path: str,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> tuple[ExtractionResult, str, str]:
        result = self.extract(pdf_path, progress_callback=progress_callback)
        paths = self.storage.save(result)
        return result, paths.document_id, str(paths.extraction_file)
