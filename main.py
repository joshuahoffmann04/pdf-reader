from pdf_extractor import PDFExtractor, NoTableOfContentsError
from dotenv import load_dotenv
import logging

# .env laden
load_dotenv()

# Logging aktivieren um Retries zu sehen
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s: %(message)s'
)

# Extractor initialisieren
extractor = PDFExtractor()

# PDF verarbeiten
def progress(current, total, status):
    print(f"Sektion {current}/{total}: {status}", flush=True)

try:
    result = extractor.extract(
        "pdfs/stpo_bsc-informatik_25-01-23_lese.pdf",
        progress_callback=progress
    )

    # Ergebnisse nutzen (v2.0 API: sections statt pages)
    print(f"\n{'='*60}")
    print(f"Titel: {result.context.title}")
    print(f"Institution: {result.context.institution}")
    print(f"Sektionen: {len(result.sections)}")

    # Statistiken
    stats = result.get_stats()
    print(f"\nStatistiken:")
    print(f"  Paragraphen (§§): {stats['paragraphs']}")
    print(f"  Anlagen: {stats['anlagen']}")
    print(f"  Mit Tabellen: {stats['sections_with_tables']}")
    print(f"  Fehlgeschlagen: {stats['failed_sections']}")

    # Erste 3 Sektionen anzeigen
    print(f"\nErste 3 Sektionen:")
    for section in result.sections[:3]:
        print(f"  {section.identifier}: {section.content[:80]}...")

    # Fehler anzeigen falls vorhanden
    if result.errors:
        print(f"\nFehler ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")

    # In Datei speichern
    result.save("output.json")
    print(f"\nGespeichert: output.json")

except NoTableOfContentsError as e:
    print(f"FEHLER: Kein Inhaltsverzeichnis gefunden!")
    print(f"Details: {e}")
except FileNotFoundError as e:
    print(f"FEHLER: PDF nicht gefunden: {e}")
