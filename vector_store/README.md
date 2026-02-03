# Vector Store Module

Lokale Embedding- und Retrieval-Pipeline auf Basis von ChromaDB und Ollama. Nimmt die Ausgabe des Chunking-Moduls (`ChunkingResult`) und speichert die Chunks als Vektor-Embeddings in einer lokalen ChromaDB-Datenbank. Ermöglicht semantische Ähnlichkeitssuche mit Nachbar-Expansion und Token-Budget-Management für den RAG-Chatbot.

## Architektur

```
ChunkingResult (JSON)              vector_store/
├── chunks[].text             →    ├── embedder.py      # OllamaEmbedder (lokale Embeddings)
├── chunks[].metadata         →    ├── store.py         # DocumentStore (ChromaDB-Wrapper)
├── chunks[].chunk_id         →    ├── retriever.py     # ChunkRetriever (Nachbar-Expansion + Token-Budget)
├── document_id               →    ├── models.py        # Datenmodelle (Pydantic v2)
└── config                    →    └── __init__.py      # Public API
```

### Datenfluss

```
ChunkingResult → DocumentStore.ingest() → OllamaEmbedder → ChromaDB (Disk)
                                                              ↓
User-Query → ChunkRetriever.retrieve() → OllamaEmbedder → ChromaDB.query()
                                                              ↓
                                         Nachbar-Expansion → Dedup → Sort → Token-Budget
                                                              ↓
                                         RetrievalResult.context_text → LLM Prompt
```

## Voraussetzungen

1. **Ollama** installiert und gestartet:
   ```bash
   # Installation: https://ollama.com
   ollama serve
   ```

2. **Embedding-Modell** herunterladen:
   ```bash
   ollama pull nomic-embed-text
   ```

## Quick Start

```python
from vector_store import DocumentStore, ChunkRetriever

# 1. Store erstellen (ChromaDB wird auf Disk persistiert)
store = DocumentStore()

# 2. Chunks aus JSON einlesen und embedden
stats = store.ingest_from_file("stpo_bsc-informatik_25-01-23_lese-chunks.json")
print(f"{stats.chunks_stored} Chunks gespeichert in {stats.total_time_seconds}s")

# 3. Semantische Suche
results = store.search("Bachelorarbeit Leistungspunkte", n_results=3)
for r in results:
    print(f"  [{r.similarity:.3f}] {r.text[:80]}...")

# 4. Kontext für den Chatbot abrufen (mit Nachbar-Expansion)
retriever = ChunkRetriever(store)
result = retriever.retrieve(
    query="Wie viele LP hat die Bachelorarbeit?",
    n_results=3,
    max_context_tokens=1024,
)
print(result.context_text)  # Direkt in den LLM-Prompt einfügbar
```

## Konfiguration

`StoreConfig` steuert alle Einstellungen:

```python
from vector_store import StoreConfig, DocumentStore

config = StoreConfig(
    collection_name="documents",         # ChromaDB-Collection
    persist_directory="./chroma_db",     # Speicherort auf Disk
    embedding_model="nomic-embed-text",  # Ollama-Modell
    ollama_base_url="http://localhost:11434",
    distance_metric="cosine",           # cosine, l2, ip
)
store = DocumentStore(config=config)
```

| Parameter | Default | Beschreibung |
|-----------|---------|-------------|
| `collection_name` | `"documents"` | Name der ChromaDB-Collection |
| `persist_directory` | `"./chroma_db"` | Verzeichnis für die ChromaDB-Datenbank |
| `embedding_model` | `"nomic-embed-text"` | Ollama-Modell für Embeddings (768 Dimensionen) |
| `ollama_base_url` | `"http://localhost:11434"` | Ollama API-Adresse |
| `distance_metric` | `"cosine"` | Distanzmetrik (cosine empfohlen für Texte) |

## Chunk-Verknüpfung (Neighbor Expansion)

Die Chunks werden maximal miteinander verknüpft. Das Retrieval-System nutzt die `prev_chunk_id` / `next_chunk_id`-Pointer aus dem Chunking-Modul:

1. **Suche**: Top-N relevanteste Chunks per Cosine-Similarity
2. **Expansion**: Für jeden Treffer werden `prev_chunk_id` und `next_chunk_id` aus den Metadaten geladen
3. **Batch-Fetch**: Alle Nachbar-Chunks werden in einem einzigen ChromaDB-Aufruf geholt
4. **Deduplizierung**: Überlappende Chunks (durch Overlap im Chunking) werden nur einmal aufgenommen
5. **Sortierung**: Alle Chunks werden nach Dokument-Reihenfolge sortiert (`document_id`, `chunk_index`)
6. **Token-Budget**: Chunks werden hinzugefügt, bis das Token-Budget erschöpft ist
7. **Separatoren**: Zwischen nicht-aufeinanderfolgenden Chunks wird `---` eingefügt

### Beispiel

Suche nach "Bachelorarbeit" findet Chunk 42. Der Retriever lädt auch Chunk 41 und 43. Die drei Chunks werden zusammengefügt und bilden einen kohärenten Kontext-Block.

```python
retriever = ChunkRetriever(store)

# Mit Nachbar-Expansion (Default)
result = retriever.retrieve("Bachelorarbeit LP", n_results=3)
print(f"{result.total_results} direkte Treffer → {len(result.context_chunks)} Chunks im Kontext")
print(f"Token-Verbrauch: {result.token_count}")

# Ohne Nachbar-Expansion
result = retriever.retrieve("Bachelorarbeit LP", n_results=3, expand_neighbors=False)

# Mit Metadaten-Filter
result = retriever.retrieve(
    "Bachelorarbeit",
    where={"document_type": "pruefungsordnung"},
)
```

## Metadaten in ChromaDB

ChromaDB unterstützt nur flache Key-Value-Metadaten. Die Chunk-Metadaten werden entsprechend transformiert:

| Feld | Typ | Beispiel |
|------|-----|---------|
| `document_id` | string | `"stpo_bsc-informatik_25-01-23_lese"` |
| `document_title` | string | `"Studien- und Prüfungsordnung..."` |
| `document_type` | string | `"pruefungsordnung"` |
| `institution` | string | `"Philipps-Universität Marburg"` |
| `degree_program` | string | `"Informatik B.Sc."` |
| `chunk_index` | int | `3` |
| `total_chunks` | int | `57` |
| `token_count` | int | `502` |
| `page_numbers` | string | `"4,5"` (kommasepariert) |
| `prev_chunk_id` | string | `"..._chunk_0002"` (nur wenn vorhanden) |
| `next_chunk_id` | string | `"..._chunk_0004"` (nur wenn vorhanden) |

## API-Referenz

### DocumentStore

Haupt-Speicher-Klasse. Verwaltet Ingestion, Suche und Chunk-Verwaltung.

```python
from vector_store import DocumentStore, StoreConfig

store = DocumentStore(config=StoreConfig())
```

**Methoden:**

| Methode | Parameter | Rückgabe | Beschreibung |
|---------|-----------|----------|-------------|
| `ingest(chunking_result)` | `ChunkingResult`, opt. `progress_callback` | `IngestStats` | Chunks embedden und speichern (Upsert) |
| `ingest_from_file(path)` | `str` (JSON-Pfad) | `IngestStats` | ChunkingResult aus JSON laden und einlesen |
| `search(query, n_results)` | `str`, opt. `int`, `where` | `list[SearchResult]` | Semantische Ähnlichkeitssuche |
| `get_chunk_by_id(id)` | `str` | `SearchResult \| None` | Einzelnen Chunk per ID holen |
| `get_chunks_by_ids(ids)` | `list[str]` | `list[SearchResult]` | Mehrere Chunks per ID holen |
| `get_document_ids()` | — | `list[str]` | Alle Dokument-IDs auflisten |
| `delete_document(doc_id)` | `str` | `int` | Alle Chunks eines Dokuments löschen |
| `count()` | — | `int` | Gesamtanzahl gespeicherter Chunks |
| `health_check()` | — | `dict` | Status von ChromaDB und Ollama |

### ChunkRetriever

Intelligenter Retriever mit Nachbar-Expansion und Token-Budget.

```python
from vector_store import ChunkRetriever

retriever = ChunkRetriever(store)
```

**Methoden:**

| Methode | Parameter | Rückgabe | Beschreibung |
|---------|-----------|----------|-------------|
| `retrieve(query)` | `str`, opt. `n_results`, `max_context_tokens`, `expand_neighbors`, `where` | `RetrievalResult` | Suche + Expansion + Token-Budget |
| `retrieve_context_string(query)` | `str`, opt. `n_results`, `max_context_tokens`, `where` | `str` | Nur den Kontext-String zurückgeben |

### OllamaEmbedder

Lokale Embedding-Generierung via Ollama.

```python
from vector_store import OllamaEmbedder

embedder = OllamaEmbedder(model="nomic-embed-text")
vector = embedder.embed("Ein Beispieltext")           # list[float] (768 Dim.)
vectors = embedder.embed_batch(["Text 1", "Text 2"])   # list[list[float]]
health = embedder.health_check()                        # dict
```

### Datenmodelle

```python
from vector_store import SearchResult, RetrievalResult, IngestStats, StoreConfig
```

| Modell | Beschreibung |
|--------|-------------|
| `StoreConfig` | Konfiguration (Collection, Pfad, Modell, URL, Metrik) |
| `SearchResult` | Ein Suchtreffer mit `chunk_id`, `text`, `metadata`, `distance`, `similarity` |
| `RetrievalResult` | Vollständiges Retrieval-Ergebnis mit `context_text`, `context_chunks`, `token_count` |
| `IngestStats` | Ingestion-Statistiken mit `chunks_stored`, `embedding_time_seconds`, `total_time_seconds` |

## Tests

```bash
# Alle Vector-Store-Tests
pytest tests/test_embedder.py tests/test_store.py tests/test_retriever.py tests/test_vector_store_models.py -v

# Gesamtes Projekt
pytest tests/ -v
```

**53 Tests** decken ab:
- Embedding: Einzel- und Batch-Embedding, leere Texte, Verbindungsfehler, Health-Check
- Store: Ingestion, Upsert, Suche, Metadaten-Filter, Chunk-Lookup, Dokument-Verwaltung
- Retriever: Nachbar-Expansion, Deduplizierung, Token-Budget, Kontext-Formatierung, Sortierung
- Modelle: Defaults, Validierung, Serialisierung

Tests verwenden `chromadb.EphemeralClient()` (In-Memory) und gemockte Embeddings — kein Ollama nötig.

## Abhängigkeiten

- `chromadb>=1.0.0` (Vektor-Datenbank)
- `ollama>=0.4.0` (Embedding-Client)
- `pydantic>=2.0.0` (Datenmodelle, bereits im Projekt)
- `tiktoken>=0.7.0` (Token-Zählung, bereits im Projekt)
- `chunking` (Eingabeformat, bereits im Projekt)
