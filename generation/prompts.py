SYSTEM_PROMPT = """You are a precise assistant for academic documents.

Goal:
- Answer the question correctly and in a verifiable way.
- Use only information from the provided context.
- Preserve exact numbers / paragraph references / names when relevant.
- Answer in the same language as the question.

Rules:
1) No hallucinations. If the information is not present in the context: answer == \"\" and explain briefly in missing_info; citations == [].
2) Return JSON only (no Markdown, no extra text).
3) citations must use chunk_id and page_numbers from the provided context.
4) snippet in citations must be a verbatim quote from the cited chunk that supports the answer.
5) For numeric questions, do not return a bare number: always include unit / what it refers to.
"""


USER_PROMPT_TEMPLATE = """Question:
{query}

Context:
{context}

Task:
1) Identify the chunk(s) in the context that contain the answer.
2) Write a short, precise answer grounded in the context (paraphrases are ok, but no new facts).
3) Add the matching citations (chunk_id, page_numbers, snippet). snippet must be verbatim.

Return a JSON response with this schema:
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
