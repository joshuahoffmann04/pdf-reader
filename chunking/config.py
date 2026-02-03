from dataclasses import dataclass, field

from .models import ChunkingConfig


@dataclass
class ChunkingServiceConfig:
    data_dir: str = "data/chunking"
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
