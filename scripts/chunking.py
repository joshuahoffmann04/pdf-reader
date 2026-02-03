from pdf_extractor import ExtractionResult
from chunking import DocumentChunker, ChunkingConfig

result = ExtractionResult.load("stpo_bsc-informatik_25-01-23_lese.json")
chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
chunks = chunker.chunk(result)
chunks.save("stpo_bsc-informatik_25-01-23_lese-chunks.json")