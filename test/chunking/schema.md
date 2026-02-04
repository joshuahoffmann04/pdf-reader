# Chunking – Evaluations-Schema

Diese Test-Pipeline evaluiert die **Chunking-Komponente** gegen eine Extraktions-Datei (Output von `pdf_extractor`).

Ziel: sicherstellen, dass die Chunks
- den Originaltext nahezu vollstaendig abdecken,
- Zahlen verlaesslich enthalten,
- nicht uebermaessig duplizieren,
- sauber an Satzgrenzen schneiden,
- konsistente Metadaten enthalten.

## Input

### Extraktion (Pfad)

Die Pipeline erwartet eine JSON-Datei im Format von `pdf_extractor.ExtractionResult`, z. B.:

- `data/pdf_extractor/<document_id>/extraction/<timestamp>.json`

In `test/chunking/main.py` ist der Pfad standardmaessig als `EXTRACTION_PATH` hinterlegt.

## Output

Im Output-Ordner (Standard: `test/chunking/output/`) werden geschrieben:

- `report.json` – Vollreport inkl. Details
- `summary.json` – Kurzsummary (Pass/Fail + Kernmetriken)
- optional `chunks.json` – erzeugte Chunks (`chunking.ChunkingResult`)

## Report (Kurzuebersicht)

Wichtige Felder in `summary.json`:

- `pass`: Gesamtbewertung
- `token_recall`: Token-Recall (Source -> Chunks)
- `number_recall`: Zahlen-Recall (Source -> Chunks)
- `duplication_ratio`: Duplikationsgrad (Chunk-Tokens / Source-Tokens)
- `sentence_boundary_ratio`: Anteil an Chunk-Grenzen, die an Satzgrenzen liegen
- `too_small_chunks` / `too_large_chunks`: Chunk-Groessenverletzungen
- `metadata_errors`: Metadaten-Integritaet (Index/IDs/Seiten)

