#!/usr/bin/env python3
"""
Vector Store Demo - Teste Embedding, Ingestion und Retrieval

Voraussetzungen:
    1. Ollama läuft: ollama serve
    2. Modell vorhanden: ollama pull nomic-embed-text
    3. Chunks-JSON existiert (vom Chunking-Modul)

Aufruf:
    python -m vector_store.main                              # Standard-Chunks
    python -m vector_store.main pfad/zu/chunks.json          # Eigene Chunks
    python -m vector_store.main --query "Bachelorarbeit LP"  # Nur Suche
"""

import argparse
import logging
import sys
from pathlib import Path

from vector_store import DocumentStore, ChunkRetriever, StoreConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_health(store: DocumentStore) -> bool:
    """Prüfe ob Ollama und ChromaDB erreichbar sind."""
    print("\n=== Health Check ===")
    health = store.health_check()

    for key, value in health.items():
        status = "OK" if value is True else ("FEHLER" if value is False else value)
        print(f"  {key}: {status}")

    if not health.get("healthy", False):
        print("\nOllama ist nicht erreichbar oder das Modell fehlt.")
        print("  1. Ollama starten: ollama serve")
        print("  2. Modell laden:   ollama pull nomic-embed-text")
        return False

    return True


def ingest_chunks(store: DocumentStore, chunks_path: str) -> None:
    """Lade und embedde Chunks aus einer JSON-Datei."""
    print(f"\n=== Ingestion: {chunks_path} ===")

    def progress(current, total, status):
        print(f"  [{current}/{total}] {status}")

    stats = store.ingest_from_file(chunks_path, progress_callback=progress)

    print(f"\n  Dokument:        {stats.document_id}")
    print(f"  Chunks:          {stats.chunks_stored}")
    print(f"  Embedding-Zeit:  {stats.embedding_time_seconds}s")
    print(f"  Gesamt-Zeit:     {stats.total_time_seconds}s")


def search_demo(store: DocumentStore, query: str, n_results: int = 5) -> None:
    """Führe eine einfache semantische Suche durch."""
    print(f"\n=== Suche: \"{query}\" (Top {n_results}) ===")

    results = store.search(query, n_results=n_results)

    if not results:
        print("  Keine Ergebnisse gefunden.")
        return

    for i, r in enumerate(results, 1):
        print(f"\n  --- Treffer {i} (Similarity: {r.similarity:.4f}) ---")
        print(f"  Chunk:  {r.chunk_id}")
        print(f"  Seiten: {r.metadata.get('page_numbers', '?')}")
        text_preview = r.text[:150].replace("\n", " ")
        print(f"  Text:   {text_preview}...")


def retrieve_demo(store: DocumentStore, query: str) -> None:
    """Zeige das vollständige Retrieval mit Nachbar-Expansion."""
    print(f"\n=== Retrieval mit Kontext: \"{query}\" ===")

    retriever = ChunkRetriever(store)
    result = retriever.retrieve(
        query=query,
        n_results=3,
        max_context_tokens=1024,
        expand_neighbors=True,
    )

    print(f"\n  Direkte Treffer:  {result.total_results}")
    print(f"  Kontext-Chunks:   {len(result.context_chunks)}")
    print(f"  Token-Verbrauch:  {result.token_count}")

    print(f"\n  Chunks im Kontext:")
    for chunk in result.context_chunks:
        idx = chunk["metadata"].get("chunk_index", "?")
        pages = chunk["metadata"].get("page_numbers", "?")
        print(f"    - Chunk {idx} (Seiten: {pages})")

    print(f"\n  === Kontext-Text (bereit für LLM) ===")
    print()
    print(result.context_text)
    print()


def show_store_info(store: DocumentStore) -> None:
    """Zeige Informationen über den aktuellen Store."""
    print(f"\n=== Store-Status ===")
    print(f"  Chunks gesamt:    {store.count()}")

    doc_ids = store.get_document_ids()
    print(f"  Dokumente:        {len(doc_ids)}")
    for doc_id in doc_ids:
        print(f"    - {doc_id}")


def main():
    parser = argparse.ArgumentParser(
        description="Vector Store Demo - Embedding, Ingestion und Retrieval testen",
    )
    parser.add_argument(
        "chunks_file",
        nargs="?",
        default=None,
        help="Pfad zur Chunks-JSON-Datei (optional, wenn bereits ingested)",
    )
    parser.add_argument(
        "--query", "-q",
        default="Wie viele Leistungspunkte hat die Bachelorarbeit?",
        help="Suchanfrage (Default: 'Wie viele Leistungspunkte hat die Bachelorarbeit?')",
    )
    parser.add_argument(
        "--n-results", "-n",
        type=int,
        default=5,
        help="Anzahl Suchergebnisse (Default: 5)",
    )
    parser.add_argument(
        "--persist-dir",
        default="./chroma_db",
        help="ChromaDB-Speicherort (Default: ./chroma_db)",
    )
    parser.add_argument(
        "--search-only",
        action="store_true",
        help="Nur Suche, keine Ingestion",
    )
    args = parser.parse_args()

    # Store konfigurieren
    config = StoreConfig(persist_directory=args.persist_dir)
    store = DocumentStore(config=config)

    # Health Check
    if not check_health(store):
        sys.exit(1)

    # Ingestion (optional)
    if args.chunks_file and not args.search_only:
        path = Path(args.chunks_file)
        if not path.exists():
            print(f"\nFEHLER: Datei nicht gefunden: {path}")
            sys.exit(1)
        ingest_chunks(store, str(path))

    # Store-Info
    show_store_info(store)

    if store.count() == 0:
        print("\nKeine Chunks im Store. Bitte zuerst Chunks einlesen:")
        print("  python -m vector_store.main chunks.json")
        sys.exit(0)

    # Suche + Retrieval
    search_demo(store, args.query, n_results=args.n_results)
    retrieve_demo(store, args.query)


if __name__ == "__main__":
    main()
