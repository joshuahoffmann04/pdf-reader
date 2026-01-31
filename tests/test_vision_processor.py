"""
Tests for VisionProcessor with mocked OpenAI API.
"""

import pytest
import json
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

from src.llm_processor.vision_processor import VisionProcessor, VisionProcessorResult
from src.llm_processor.models import (
    ProcessingConfig,
    DocumentContext,
    ExtractedPage,
    DocumentType,
)


class TestVisionProcessorInit:
    """Tests for VisionProcessor initialization."""

    def test_init_with_api_key(self):
        """Test initialization with API key."""
        with patch('src.llm_processor.vision_processor.OpenAI'):
            processor = VisionProcessor(api_key="test-key")
            assert processor.api_key == "test-key"

    def test_init_without_api_key_raises(self):
        """Test initialization without API key raises error."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="OpenAI API key required"):
                VisionProcessor()

    def test_init_with_env_var(self):
        """Test initialization with environment variable."""
        with patch.dict('os.environ', {'OPENAI_API_KEY': 'env-key'}):
            with patch('src.llm_processor.vision_processor.OpenAI'):
                processor = VisionProcessor()
                assert processor.api_key == "env-key"

    def test_init_with_custom_config(self):
        """Test initialization with custom config."""
        config = ProcessingConfig(model="gpt-4o-mini")
        with patch('src.llm_processor.vision_processor.OpenAI'):
            processor = VisionProcessor(config=config, api_key="test-key")
            assert processor.config.model == "gpt-4o-mini"


class TestVisionProcessorMocked:
    """Tests for VisionProcessor with mocked API calls."""

    @pytest.fixture
    def mock_openai(self):
        """Create a mocked OpenAI client."""
        with patch('src.llm_processor.vision_processor.OpenAI') as mock:
            yield mock

    @pytest.fixture
    def mock_pdf_converter(self):
        """Create a mocked PDF converter."""
        with patch('src.llm_processor.vision_processor.PDFToImages') as mock:
            converter = Mock()
            converter.get_document_info.return_value = {"page_count": 3}
            converter.get_page_count.return_value = 3

            # Mock image rendering
            mock_image = Mock()
            mock_image.image_base64 = "base64data"
            mock_image.mime_type = "image/png"

            converter.render_page.return_value = mock_image
            converter.render_pages_batch.return_value = [mock_image]

            mock.return_value = converter
            yield mock

    @pytest.fixture
    def processor(self, mock_openai, mock_pdf_converter, mock_context_response, mock_openai_response):
        """Create a VisionProcessor with mocked dependencies."""
        # Setup mock responses
        mock_client = Mock()

        # Create mock response objects
        context_response = Mock()
        context_response.choices = [Mock()]
        context_response.choices[0].message.content = json.dumps(mock_context_response)
        context_response.usage = Mock(prompt_tokens=100, completion_tokens=50)

        page_response = Mock()
        page_response.choices = [Mock()]
        page_response.choices[0].message.content = json.dumps(mock_openai_response)
        page_response.usage = Mock(prompt_tokens=200, completion_tokens=100)

        # Return different responses for context vs page extraction
        mock_client.chat.completions.create.side_effect = [
            context_response,  # Context analysis
            page_response,     # Page 1
            page_response,     # Page 2
            page_response,     # Page 3
        ]

        mock_openai.return_value = mock_client

        return VisionProcessor(api_key="test-key")

    def test_process_document(self, processor, pdf_path):
        """Test full document processing."""
        if pdf_path is None:
            pytest.skip("Test PDF not available")

        result = processor.process_document(pdf_path)

        assert isinstance(result, VisionProcessorResult)
        assert result.context is not None
        assert len(result.pages) == 3  # Mocked page count

    def test_process_document_with_callback(self, processor, pdf_path):
        """Test processing with progress callback."""
        if pdf_path is None:
            pytest.skip("Test PDF not available")

        callback_calls = []

        def callback(current, total, status):
            callback_calls.append((current, total, status))

        result = processor.process_document(pdf_path, progress_callback=callback)

        # Should have been called for context + each page
        assert len(callback_calls) >= 3

    def test_parse_context_response(self, processor, mock_context_response):
        """Test context response parsing."""
        response_json = json.dumps(mock_context_response)
        context = processor._parse_context_response(response_json, 10)

        assert isinstance(context, DocumentContext)
        assert context.document_type == DocumentType.PRUEFUNGSORDNUNG
        assert context.title == mock_context_response["title"]
        assert len(context.abbreviations) == 2

    def test_parse_context_response_invalid(self, processor):
        """Test handling of invalid context response."""
        context = processor._parse_context_response("invalid json", 10)

        assert context.document_type == DocumentType.OTHER
        assert context.title == "Unknown"

    def test_parse_page_response(self, processor, mock_openai_response):
        """Test page response parsing."""
        response_json = json.dumps(mock_openai_response)
        page = processor._parse_page_response(response_json, 1)

        assert isinstance(page, ExtractedPage)
        assert page.page_number == 1
        assert page.content == mock_openai_response["content"]
        assert len(page.sections) == 1

    def test_parse_page_response_with_json_block(self, processor, mock_openai_response):
        """Test parsing response with markdown code block."""
        response = f"Here is the JSON:\n```json\n{json.dumps(mock_openai_response)}\n```"
        page = processor._parse_page_response(response, 1)

        assert page.content == mock_openai_response["content"]

    def test_parse_page_response_invalid(self, processor):
        """Test handling of invalid page response."""
        page = processor._parse_page_response("invalid response", 1)

        assert page.page_number == 1
        assert page.extraction_confidence == 0.5
        assert "invalid response" in page.content

    def test_extract_json_from_code_block(self, processor):
        """Test JSON extraction from code blocks."""
        text = 'Some text\n```json\n{"key": "value"}\n```\nMore text'
        result = processor._extract_json(text)

        assert result == '{"key": "value"}'

    def test_extract_json_raw(self, processor):
        """Test JSON extraction from raw text."""
        text = 'Before {"key": "value"} after'
        result = processor._extract_json(text)

        assert result == '{"key": "value"}'

    def test_extract_json_nested(self, processor):
        """Test JSON extraction with nested objects."""
        text = '{"outer": {"inner": "value"}}'
        result = processor._extract_json(text)

        assert '"inner"' in result
        assert json.loads(result)["outer"]["inner"] == "value"

    def test_token_tracking(self, processor, pdf_path):
        """Test token usage tracking."""
        if pdf_path is None:
            pytest.skip("Test PDF not available")

        result = processor.process_document(pdf_path)

        assert result.total_input_tokens > 0
        assert result.total_output_tokens > 0

    def test_error_handling(self, mock_openai, mock_pdf_converter):
        """Test error handling during processing."""
        mock_client = Mock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        mock_openai.return_value = mock_client

        processor = VisionProcessor(api_key="test-key")

        # Use a mock path
        with patch.object(Path, 'exists', return_value=True):
            mock_path = Path("test.pdf")
            result = processor.process_document(mock_path)

            # Should have errors but not crash
            assert len(result.errors) > 0


class TestVisionProcessorResult:
    """Tests for VisionProcessorResult class."""

    def test_result_attributes(self, sample_context, sample_pages):
        """Test result has all required attributes."""
        result = VisionProcessorResult(
            context=sample_context,
            pages=sample_pages,
            processing_time_seconds=10.5,
            total_input_tokens=1000,
            total_output_tokens=500,
            errors=["Test error"],
        )

        assert result.context == sample_context
        assert len(result.pages) == len(sample_pages)
        assert result.processing_time_seconds == 10.5
        assert result.total_input_tokens == 1000
        assert result.total_output_tokens == 500
        assert "Test error" in result.errors
