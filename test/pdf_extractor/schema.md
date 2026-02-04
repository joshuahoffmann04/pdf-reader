# PDF Extractor – Evaluations-Schema

Diese Test-Pipeline evaluiert die **pdf_extractor-Komponente** gegen eine manuell kuratierte Referenz-Extraktion.

Ziel: messen, wie gut die Extraktion den Referenztext pro Seite trifft (Textqualitaet + strukturierte Marker).

## Input

### PDF (Pfad)

Eine PDF-Datei, z. B.:

- `pdfs/2-aend-19-02-25_msc-computer-science_lese.pdf`

In `test/pdf_extractor/main.py` ist der Pfad als `PDF_PATH` hinterlegt.

### Referenz (JSON, Pfad)

Eine JSON-Datei mit Referenzseiten im Format der Extractor-Ausgabe (mindestens `pages[].page_number` und `pages[].content`), z. B.:

- `reference/pdf_extractor/<document_id>/extraction/<file>.json`

In `test/pdf_extractor/main.py` ist der Pfad als `REFERENCE_PATH` hinterlegt.

## Output

Im Output-Ordner (Standard: `test/pdf_extractor/output/`) werden geschrieben:

- `report.json` – Vollreport inkl. Seitenmetriken und Diffs
- `summary.json` – Kurzsummary

## Report (Kurzuebersicht)

Wichtige Felder in `summary.json`:

- `pass`
- `pages`
- `cer_avg` (Character Error Rate)
- `wer_avg` (Word Error Rate)
- `token_recall_avg`
- `number_recall_avg`
- zusaetzlich: aggregierte Precision/Recall/F1 fuer `section_numbers`, `paragraph_numbers`, `internal_references`

