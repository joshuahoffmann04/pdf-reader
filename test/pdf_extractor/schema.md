# Reference Schema

The evaluation expects a JSON file that follows the `ExtractionResult` shape.
Only a subset is required. The pipeline evaluates whatever is provided.

## Required
- `pages`: list of objects with:
  - `page_number` (int, 1-based)
  - `content` (str)

## Optional (used for structure metrics)
- `section_numbers` (list of str)
- `paragraph_numbers` (list of str)
- `internal_references` (list of str)

## Example (minimal)
```
{
  "pages": [
    {"page_number": 1, "content": "Title page content ..."},
    {"page_number": 2, "content": "Table of contents ..."}
  ]
}
```

## Example (with structure)
```
{
  "pages": [
    {
      "page_number": 5,
      "content": "Section content ...",
      "section_numbers": ["Section 1"],
      "paragraph_numbers": ["(1)", "(2)"],
      "internal_references": ["Section 5 Abs. 2"]
    }
  ]
}
```
