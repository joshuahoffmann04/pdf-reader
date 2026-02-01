from pdf_extractor import PDFExtractor
from dotenv import load_dotenv

#.env laden
load_dotenv()

# Extractor initialisieren
extractor = PDFExtractor()

# PDF verarbeiten
def progress(current, total, status):
    print(f"Seite {current}/{total}: {status}")

result = extractor.extract("pdfs/stpo_bsc-informatik_25-01-23_lese.pdf", progress_callback=progress)

# Ergebnisse nutzen
print(f"Titel: {result.context.title}")
print(f"Seiten: {len(result.pages)}")

# In Datei speichern
result.save("output.json")