"""
Unit tests for PDFExtractor class.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from pdf_extractor import (
    PDFExtractor,
    ExtractionConfig,
    DocumentContext,
    StructureEntry,
    ExtractedSection,
    ExtractionResult,
    DocumentType,
    SectionType,
    NoTableOfContentsError,
)
from pdf_extractor.extractor import estimate_api_cost


class TestPDFExtractorInit:
    """Tests for PDFExtractor initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")
            assert extractor.api_key == "test-key"

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                PDFExtractor()

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}):
            with patch('pdf_extractor.extractor.OpenAI'):
                extractor = PDFExtractor()
                assert extractor.api_key == "env-key"

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = ExtractionConfig(model="gpt-4o-mini", max_images_per_request=10)
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(config=config, api_key="test-key")
            assert extractor.config.model == "gpt-4o-mini"
            assert extractor.config.max_images_per_request == 10


class TestSlidingWindow:
    """Tests for sliding window logic."""

    def test_single_window(self):
        """Test that single window is returned for small sections."""
        config = ExtractionConfig(max_images_per_request=5)
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(config=config, api_key="test-key")

            pages = [1, 2, 3]
            windows = extractor._create_sliding_windows(pages, 5)

            assert len(windows) == 1
            assert windows[0] == [1, 2, 3]

    def test_exact_fit(self):
        """Test that exact fit returns single window."""
        config = ExtractionConfig(max_images_per_request=5)
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(config=config, api_key="test-key")

            pages = [1, 2, 3, 4, 5]
            windows = extractor._create_sliding_windows(pages, 5)

            assert len(windows) == 1
            assert windows[0] == [1, 2, 3, 4, 5]

    def test_two_windows_with_overlap(self):
        """Test that two windows overlap by 1 page."""
        config = ExtractionConfig(max_images_per_request=5)
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(config=config, api_key="test-key")

            pages = [1, 2, 3, 4, 5, 6, 7]
            windows = extractor._create_sliding_windows(pages, 5)

            assert len(windows) == 2
            assert windows[0] == [1, 2, 3, 4, 5]
            assert windows[1] == [5, 6, 7]
            # Check overlap
            assert windows[0][-1] == windows[1][0]

    def test_three_windows_with_overlap(self):
        """Test that three windows each overlap by 1 page."""
        config = ExtractionConfig(max_images_per_request=5)
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(config=config, api_key="test-key")

            pages = list(range(1, 13))  # 1-12
            windows = extractor._create_sliding_windows(pages, 5)

            assert len(windows) == 3
            assert windows[0] == [1, 2, 3, 4, 5]
            assert windows[1] == [5, 6, 7, 8, 9]
            assert windows[2] == [9, 10, 11, 12]
            # Check overlaps
            assert windows[0][-1] == windows[1][0]
            assert windows[1][-1] == windows[2][0]

    def test_all_pages_covered(self):
        """Test that all pages are covered by at least one window."""
        config = ExtractionConfig(max_images_per_request=3)
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(config=config, api_key="test-key")

            pages = list(range(1, 11))  # 1-10
            windows = extractor._create_sliding_windows(pages, 3)

            # Flatten windows and check all pages present
            covered = set()
            for window in windows:
                covered.update(window)

            assert covered == set(pages)


class TestJSONExtraction:
    """Tests for JSON extraction from responses."""

    def test_extract_json_code_block(self):
        """Test extracting JSON from code block."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            response = '''Here is the result:
```json
{"key": "value", "number": 42}
```
That's all.'''

            result = extractor._extract_json(response)
            data = json.loads(result)
            assert data["key"] == "value"
            assert data["number"] == 42

    def test_extract_json_raw(self):
        """Test extracting raw JSON."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            response = '{"key": "value"}'
            result = extractor._extract_json(response)
            data = json.loads(result)
            assert data["key"] == "value"

    def test_extract_nested_json(self):
        """Test extracting nested JSON."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            response = '''```json
{"outer": {"inner": {"deep": "value"}}, "list": [1, 2, 3]}
```'''

            result = extractor._extract_json(response)
            data = json.loads(result)
            assert data["outer"]["inner"]["deep"] == "value"
            assert data["list"] == [1, 2, 3]


class TestRefusalDetection:
    """Tests for API refusal detection."""

    def test_detects_refusal(self):
        """Test that refusals are detected."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            refusal_messages = [
                "I'm sorry, I can't assist with that request.",
                "I cannot assist with this.",
                "I'm not able to process this document.",
                "I am unable to help with this.",
            ]

            for msg in refusal_messages:
                assert extractor._is_refusal(msg), f"Should detect refusal: {msg}"

    def test_not_refusal(self):
        """Test that normal content is not flagged as refusal."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            normal_messages = [
                '{"content": "This is a valid response"}',
                "The document contains regulations about...",
                "§ 1 Geltungsbereich: Diese Ordnung regelt...",
            ]

            for msg in normal_messages:
                assert not extractor._is_refusal(msg), f"Should not flag: {msg}"

    def test_empty_not_refusal(self):
        """Test that empty/None content is not flagged as refusal."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            # Empty and None should not be flagged as refusal
            assert not extractor._is_refusal("")
            assert not extractor._is_refusal(None)


class TestEstimateAPICost:
    """Tests for API cost estimation."""

    def test_estimate_gpt4o(self):
        """Test cost estimation for gpt-4o."""
        estimate = estimate_api_cost(50, "gpt-4o")

        assert "estimated_input_tokens" in estimate
        assert "estimated_output_tokens" in estimate
        assert "estimated_cost_usd" in estimate
        assert estimate["model"] == "gpt-4o"
        assert estimate["page_count"] == 50

    def test_estimate_gpt4o_mini(self):
        """Test cost estimation for gpt-4o-mini."""
        estimate = estimate_api_cost(50, "gpt-4o-mini")

        assert estimate["model"] == "gpt-4o-mini"
        # Mini should be cheaper
        gpt4o_cost = estimate_api_cost(50, "gpt-4o")["estimated_cost_usd"]
        assert estimate["estimated_cost_usd"] < gpt4o_cost

    def test_estimate_unknown_model(self):
        """Test cost estimation for unknown model."""
        estimate = estimate_api_cost(50, "unknown-model")
        assert "error" in estimate


class TestParseStructureResponse:
    """Tests for structure response parsing."""

    def test_parse_valid_response(self, mock_structure_response):
        """Test parsing a valid structure response."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            json_str = json.dumps(mock_structure_response)
            context, structure = extractor._parse_structure_response(
                json_str, total_pages=56, document_path="test.pdf"
            )

            assert context.document_type == DocumentType.PRUEFUNGSORDNUNG
            assert len(structure) == 3
            assert structure[0].section_type == SectionType.OVERVIEW
            assert structure[1].section_number == "§ 1"

    def test_parse_no_toc_response(self):
        """Test parsing response with no ToC."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            response = json.dumps({
                "has_toc": False,
                "context": None,
                "structure": None
            })

            with pytest.raises(NoTableOfContentsError):
                extractor._parse_structure_response(
                    response, total_pages=56, document_path="test.pdf"
                )


class TestParseSectionResponse:
    """Tests for section response parsing."""

    def test_parse_valid_response(self, mock_section_response):
        """Test parsing a valid section response."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            entry = StructureEntry(
                section_type=SectionType.PARAGRAPH,
                section_number="§ 1",
                section_title="Geltungsbereich",
                start_page=3,
                end_page=3,
            )

            json_str = json.dumps(mock_section_response)
            section = extractor._parse_section_response(
                json_str, entry, pages=[3]
            )

            assert section.section_number == "§ 1"
            assert "Geltungsbereich" in section.content
            assert section.extraction_confidence == 1.0

    def test_parse_invalid_json(self):
        """Test parsing invalid JSON returns fallback."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            entry = StructureEntry(
                section_type=SectionType.PARAGRAPH,
                section_number="§ 1",
                start_page=3,
                end_page=3,
            )

            section = extractor._parse_section_response(
                "Not valid JSON", entry, pages=[3]
            )

            assert section.section_number == "§ 1"
            assert section.extraction_confidence == 0.5
            assert "JSON parsing failed" in section.extraction_notes


class TestCreateFailedSection:
    """Tests for failed section placeholder creation."""

    def test_create_failed_section(self):
        """Test creating a failed section placeholder."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            entry = StructureEntry(
                section_type=SectionType.PARAGRAPH,
                section_number="§ 5",
                section_title="Test Section",
                start_page=10,
                end_page=12,
            )

            section = extractor._create_failed_section(entry, "API error")

            assert section.section_number == "§ 5"
            assert section.extraction_confidence == 0.0
            assert "API error" in section.extraction_notes
            assert section.pages == [10, 11, 12]


class TestParseContext:
    """Tests for context parsing."""

    def test_parse_context_with_abbreviations(self):
        """Test parsing context with abbreviations."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            data = {
                "document_type": "pruefungsordnung",
                "title": "Test PO",
                "institution": "Test Uni",
                "abbreviations": [
                    {"short": "LP", "long": "Leistungspunkte"},
                    {"short": "ECTS", "long": "European Credit Transfer System"},
                ],
            }

            context = extractor._parse_context(data, total_pages=50)

            assert context.document_type == DocumentType.PRUEFUNGSORDNUNG
            assert len(context.abbreviations) == 2
            assert context.abbreviations[0].short == "LP"

    def test_parse_context_with_unknown_type(self):
        """Test parsing context with unknown document type."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            data = {
                "document_type": "unknown_type",
                "title": "Test Doc",
                "institution": "Test Uni",
            }

            context = extractor._parse_context(data, total_pages=50)

            assert context.document_type == DocumentType.OTHER


class TestParseStructureEntries:
    """Tests for structure entry parsing."""

    def test_parse_entries_sorted(self):
        """Test that entries are sorted by start page."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            data = [
                {"section_type": "paragraph", "section_number": "§ 2", "start_page": 5, "end_page": 6},
                {"section_type": "overview", "section_number": None, "start_page": 1, "end_page": 2},
                {"section_type": "paragraph", "section_number": "§ 1", "start_page": 3, "end_page": 4},
            ]

            entries = extractor._parse_structure_entries(data, total_pages=50)

            assert len(entries) == 3
            assert entries[0].start_page == 1
            assert entries[1].start_page == 3
            assert entries[2].start_page == 5

    def test_parse_entries_page_clamping(self):
        """Test that invalid pages are clamped."""
        with patch('pdf_extractor.extractor.OpenAI'):
            extractor = PDFExtractor(api_key="test-key")

            data = [
                {"section_type": "paragraph", "section_number": "§ 1", "start_page": 0, "end_page": 100},
            ]

            entries = extractor._parse_structure_entries(data, total_pages=50)

            assert entries[0].start_page == 1
            assert entries[0].end_page == 50
