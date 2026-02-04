SYSTEM_PROMPT = """Du bist ein praeziser Assistent fuer akademische Dokumente.

Regeln:
1) Antworte nur mit Informationen aus dem Kontext.
2) Keine Halluzinationen. Wenn etwas fehlt, setze missing_info.
3) Antworte ausschliesslich im JSON-Format (kein Markdown).
4) Zitiere mit chunk_id und Seitenzahlen aus dem Kontext.
"""


USER_PROMPT_TEMPLATE = """Frage:
{query}

Kontext:
{context}

Gib eine JSON-Antwort im folgenden Schema:
{{
  "answer": "...",
  "citations": [
    {{"chunk_id": "...", "page_numbers": [12, 13], "snippet": "..."}}
  ],
  "missing_info": ""
}}
"""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "chunk_id": {"type": "string"},
                    "page_numbers": {"type": "array", "items": {"type": "integer"}},
                    "snippet": {"type": "string"},
                },
                "required": ["chunk_id", "page_numbers", "snippet"],
            },
        },
        "missing_info": {"type": "string"},
    },
    "required": ["answer", "citations", "missing_info"],
}
