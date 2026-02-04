# Generation Evaluation Schema

The evaluation uses a JSON query set with expected answer and citation signals.

## Query Set (JSON)

Top-level fields:
- `queries`: list of query objects
- `default_mode` (optional, default: "hybrid")

Each query object:
```
{
  "id": "q1",
  "query": "Was ist die Regelstudienzeit?",
  "mode": "hybrid",
  "expected_answer_contains": ["Regelstudienzeit", "Semester"],
  "expected_missing_info": false,
  "expected_citation_chunk_ids": ["doc_chunk_0033"],
  "expected_page_numbers": [5, 6],
  "min_citations": 1
}
```

Notes:
- Provide at least one of: `expected_answer_contains`, `expected_missing_info`,
  `expected_citation_chunk_ids`, `expected_page_numbers`.
- All fields are optional; evaluation only checks fields that are present.
