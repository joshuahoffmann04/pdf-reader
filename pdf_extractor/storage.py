from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .models import ExtractionResult


@dataclass
class ExtractionPaths:
    document_id: str
    extraction_dir: Path
    extraction_file: Path


class ExtractionStorage:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def build_paths(self, source_file: str) -> ExtractionPaths:
        document_id = Path(source_file).stem
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        extraction_dir = self.data_dir / document_id / "extraction"
        extraction_dir.mkdir(parents=True, exist_ok=True)
        extraction_file = extraction_dir / f"{document_id}_{timestamp}.json"
        return ExtractionPaths(
            document_id=document_id,
            extraction_dir=extraction_dir,
            extraction_file=extraction_file,
        )

    def save(self, result: ExtractionResult) -> ExtractionPaths:
        paths = self.build_paths(result.source_file)
        result.save(str(paths.extraction_file))
        return paths
