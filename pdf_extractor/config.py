from dataclasses import dataclass, field

from .models import ProcessingConfig


@dataclass
class ExtractorConfig:
    data_dir: str = "data/pdf_extractor"
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
