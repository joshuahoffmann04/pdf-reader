# Chunking Evaluation Schema

The evaluation uses:
- An extraction JSON (input to chunking)
- A generated chunking result (output from chunking)

No special reference file is required. The evaluation compares:
- Source text coverage
- Chunk size constraints
- Metadata consistency
- Sentence boundary heuristics

## Required Input
- Extraction JSON that can be loaded by `ExtractionResult.load(...)`

## Output
- `report.json`: Detailed metrics and per-chunk issues
- `summary.json`: Compact summary with pass/fail
