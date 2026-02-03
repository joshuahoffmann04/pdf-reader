# Chunking Module

Satzbasiertes Sliding-Window-Chunking für die RAG-Pipeline. Nimmt die Ausgabe des PDF-Extraktors (`ExtractionResult`) und erzeugt überlappende Chunks mit Metadaten, optimiert für ein lokales LLM mit 2048 Token Kontextfenster.

## Architektur

```
ExtractionResult (JSON)          chunking/
├── pages[].content         →    ├── sentence_splitter.py   # Deutsche Satzerkennung
├── pages[].page_number     →    ├── token_counter.py       # tiktoken Token-Zählung
├── context (Dokument-Meta) →    ├── chunker.py             # DocumentChunker (Hauptklasse)
└── source_file             →    ├── models.py              # Datenmodelle (Pydantic v2)
                                 └── __init__.py            # Public API
```

## Quick Start

```python
from pdf_extractor import ExtractionResult
from chunking import DocumentChunker, ChunkingConfig

# 1. Extraktionsergebnis laden
result = ExtractionResult.load("stpo_bsc-informatik_25-01-23_lese.json")

# 2. Chunker konfigurieren und ausführen
chunker = DocumentChunker(ChunkingConfig(max_chunk_tokens=512))
chunks = chunker.chunk(result)

# 3. Ergebnis speichern
chunks.save("stpo_bsc-informatik_25-01-23_lese-chunks.json")
```

Alternativ direkt aus einer JSON-Datei:

```python
from chunking import DocumentChunker

chunker = DocumentChunker()
chunks = chunker.chunk_from_file("output.json")
chunks.save("chunks.json")
```

## Konfiguration

`ChunkingConfig` steuert das Verhalten:

```python
from chunking import ChunkingConfig

config = ChunkingConfig(
    max_chunk_tokens=512,   # Hartes Limit pro Chunk (Default: 512)
    overlap_tokens=100,     # Overlap zwischen aufeinanderfolgenden Chunks (Default: 100)
    min_chunk_tokens=50,    # Minimale Chunk-Größe, verhindert Mikro-Chunks (Default: 50)
)
```

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `max_chunk_tokens` | 512 | Maximale Token-Anzahl pro Chunk. Kein Chunk überschreitet dieses Limit. |
| `overlap_tokens` | 100 | Ziel-Overlap in Tokens zwischen aufeinanderfolgenden Chunks (Sliding Window). |
| `min_chunk_tokens` | 50 | Chunks unter diesem Wert werden entfernt (außer der letzte). |

## Algorithmus

1. **Page Merging**: Alle Seitentexte werden zu einem durchgehenden Textfluss zusammengeführt. Für jeden Zeichenindex wird die Ursprungsseite getrackt.

2. **Satz-Splitting**: Der Text wird an Satzgrenzen aufgeteilt (`.`, `!`, `?` + Großbuchstabe). Deutsche Abkürzungen (`Abs.`, `Nr.`, `z.B.`, `d.h.`, `i.d.R.`) und Paragraphenreferenzen (`§ 5 Abs. 2`) werden geschützt.

3. **Chunk-Bildung (Sliding Window)**: Sätze werden akkumuliert, bis das Token-Limit erreicht ist. Der nächste Chunk startet einige Sätze zurück, sodass ca. `overlap_tokens` Tokens überlappen. Kein Satz wird je zerteilt.

4. **Metadaten-Anreicherung**: Jeder Chunk bekommt Dokument-Metadaten, Seitennummern und Nachbar-Pointer (`prev_chunk_id`/`next_chunk_id`).

## Ausgabeformat

### Chunk

```json
{
  "chunk_id": "stpo_bsc-informatik_25-01-23_lese_chunk_0003",
  "text": "Zentrale Bedeutung haben die Befähigung zu wissenschaftlicher ...",
  "token_count": 502,
  "metadata": {
    "document_id": "stpo_bsc-informatik_25-01-23_lese",
    "document_title": "Studien- und Prüfungsordnung ...",
    "document_type": "pruefungsordnung",
    "institution": "Philipps-Universität Marburg",
    "degree_program": "Informatik B.Sc.",
    "page_numbers": [4, 5],
    "chunk_index": 3,
    "total_chunks": 57,
    "prev_chunk_id": "stpo_bsc-informatik_25-01-23_lese_chunk_0002",
    "next_chunk_id": "stpo_bsc-informatik_25-01-23_lese_chunk_0004"
  }
}
```

### ChunkingResult (vollständig)

```json
{
  "source_file": "pdfs/stpo_bsc-informatik_25-01-23_lese.pdf",
  "document_id": "stpo_bsc-informatik_25-01-23_lese",
  "config": {
    "max_chunk_tokens": 512,
    "overlap_tokens": 100,
    "min_chunk_tokens": 50
  },
  "chunks": [ ... ],
  "stats": {
    "total_chunks": 57,
    "total_tokens": 27723,
    "avg_chunk_tokens": 486.4,
    "min_chunk_tokens": 257,
    "max_chunk_tokens": 512,
    "total_sentences": 561,
    "total_pages_processed": 50
  },
  "created_at": "2026-02-03T09:32:23.756677"
}
```

## API-Referenz

### DocumentChunker

Die Hauptklasse des Moduls.

```python
from chunking import DocumentChunker, ChunkingConfig

chunker = DocumentChunker(config=ChunkingConfig())
```

**Methoden:**

| Methode | Parameter | Rückgabe | Beschreibung |
|---------|-----------|----------|-------------|
| `chunk(result)` | `ExtractionResult` | `ChunkingResult` | Chunked ein Extraktionsergebnis |
| `chunk_from_file(path)` | `str` (JSON-Pfad) | `ChunkingResult` | Lädt JSON und chunked es |

### ChunkingResult

Das Ergebnisobjekt mit allen Chunks und Statistiken.

```python
result = chunker.chunk(extraction_result)
```

**Properties und Methoden:**

| Zugriff | Rückgabe | Beschreibung |
|---------|----------|-------------|
| `result.total_chunks` | `int` | Gesamtanzahl der Chunks |
| `result.chunks` | `list[Chunk]` | Alle Chunks mit Metadaten |
| `result.stats` | `ChunkingStats` | Statistiken (Token-Durchschnitt, Min, Max, etc.) |
| `result.document_id` | `str` | Dokument-ID (aus Dateiname) |
| `result.get_chunk_by_id(id)` | `Chunk \| None` | Chunk per ID suchen |
| `result.get_neighbors(id)` | `(Chunk, Chunk)` | Vorherigen und nächsten Chunk holen |
| `result.save(path)` | `None` | Als JSON speichern |
| `ChunkingResult.load(path)` | `ChunkingResult` | Aus JSON laden |
| `result.to_json()` | `str` | Als JSON-String |
| `result.to_dict()` | `dict` | Als Dictionary |

### Einzelne Komponenten

Die internen Module können auch direkt verwendet werden:

```python
from chunking.sentence_splitter import split_sentences
from chunking.token_counter import count_tokens, count_tokens_batch

# Sätze splitten
sentences = split_sentences("Dies ist gem. § 5 Abs. 2 korrekt. Der zweite Satz.")
# -> ["Dies ist gem. § 5 Abs. 2 korrekt.", "Der zweite Satz."]

# Tokens zählen
n = count_tokens("Prüfungsordnung für den Bachelorstudiengang")
# -> 14

# Batch-Zählung
counts = count_tokens_batch(["Satz eins.", "Satz zwei."])
# -> [3, 3]
```

## Tests

```bash
# Alle Chunking-Tests
pytest tests/test_chunker.py tests/test_chunking_models.py tests/test_sentence_splitter.py tests/test_token_counter.py -v

# Gesamtes Projekt
pytest tests/ -v
```

Abgedeckte Testbereiche:
- Token-Limits (kein Chunk > 512 Tokens)
- Sliding-Window-Overlap (Textüberlappung prüfen)
- Satzgrenzen (keine Sätze zerteilt)
- Deutsche Abkürzungen (Abs., Nr., z.B., d.h., i.d.R.)
- Paragraphenreferenzen (§ 5 Abs. 2)
- Metadaten-Konsistenz (Neighbor-Pointer, Page-Tracking)
- Serialisierung (Save/Load JSON)
- Edge Cases (leere Seiten, einzelne Sätze > 512 Tokens)

## Abhängigkeiten

- `tiktoken>=0.7.0` (Token-Zählung)
- `pydantic>=2.0.0` (Datenmodelle, bereits im Projekt)
- `pdf_extractor` (Eingabeformat, bereits im Projekt)
