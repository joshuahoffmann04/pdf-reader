# MARley – Evaluations-Schema

Diese Harness evaluiert den **MARley Chatbot** (Retrieval + Kontextaufbau + Generation + Zitate) gegen ein kuratiertes Fragen-Set.

Wichtig: Im Gegensatz zu `test/generation/` baut `test/marley/` die Indizes direkt aus einer Chunk-Datei (kein Retrieval-Service notwendig).

## Input

Datei: z. B. `test/marley/msc_computer_science_questions.json`

Schema:

```json
{
  "questions": [
    {
      "id": "q001",
      "document": "pdfs/2-aend-19-02-25_msc-computer-science_lese.pdf",
      "question": "…",
      "expected_answer": "…",
      "reference_page_numbers": [3, 4],
      "reference_quote": "…"
    }
  ]
}
```

Bedeutung:
- `id`: stabile ID fuer Reporting/Regression
- `document`: Ziel-PDF (wird zur Bestimmung von `document_id` genutzt)
- `question`: Nutzerfrage
- `expected_answer`: erwarteter Antwortinhalt (nicht zwingend wortgleich; Bewertung ist inhaltsorientiert)
- `reference_page_numbers`: Seiten, auf denen die Information im PDF belegt ist
- `reference_quote`: exakter Belegtext aus dem PDF (u.a. fuer Support-/Citation-Checks)

## Output

Im Output-Ordner (Standard: `test/marley/output/`) werden geschrieben:

- `report.json` – Vollreport je Frage inkl. Debug-Infos
- `summary.json` – Kurzsummary

## Bewertung (High-Level)

Die Harness trennt bewusst zwischen:
- **Antwort-Qualitaet** (`pass_answer`): primaer semantische Aehnlichkeit + Zahlen-Recall
- **Zitate/Belege** (`pass_citations`): Treffer auf Referenz-Seiten und Support via `reference_quote`
- **Strict Pass** (`pass`): beides muss passen
