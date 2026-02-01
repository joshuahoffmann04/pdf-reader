#!/usr/bin/env python3
"""
Debug script for Context Analysis phase.

This script helps diagnose issues with the document context extraction,
specifically:
1. Which pages are being sent to the LLM
2. What the LLM returns
3. How the response is parsed

Usage:
    python scripts/debug_context_analysis.py pdfs/your_document.pdf
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dotenv import load_dotenv
load_dotenv()

from llm_processor.pdf_to_images import PDFToImages
from llm_processor.prompts import CONTEXT_ANALYSIS_SYSTEM, CONTEXT_ANALYSIS_USER


def debug_context_analysis(pdf_path: str):
    """Debug the context analysis phase."""

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        print(f"❌ File not found: {pdf_path}")
        return

    print("=" * 60)
    print("CONTEXT ANALYSIS DEBUG")
    print("=" * 60)

    # Initialize
    pdf_converter = PDFToImages(dpi=150)
    doc_info = pdf_converter.get_document_info(pdf_path)
    total_pages = doc_info["page_count"]

    print(f"\n📄 Document: {pdf_path.name}")
    print(f"📊 Total pages: {total_pages}")

    # Current sample page selection
    sample_pages_current = [1]
    if total_pages > 2:
        sample_pages_current.append(total_pages // 2)
    if total_pages > 1:
        sample_pages_current.append(total_pages)

    print(f"\n🔍 CURRENT sample pages: {sample_pages_current}")
    print("   ⚠️  This is the PROBLEM - page 2 and 3 (table of contents) are NOT included!")

    # Better sample page selection
    sample_pages_better = [1, 2, 3]  # Include table of contents
    if total_pages > 5:
        sample_pages_better.append(total_pages // 2)
    if total_pages > 3:
        sample_pages_better.append(total_pages)

    print(f"\n✅ BETTER sample pages: {sample_pages_better}")
    print("   This includes pages 2-3 where the table of contents usually is!")

    # Check what's on each sample page
    print("\n" + "-" * 60)
    print("PAGE CONTENT PREVIEW")
    print("-" * 60)

    # We can't easily preview without calling the API, but we can show the images
    print("\nRendering sample pages for visual inspection...")

    for page_num in [1, 2, 3, 4, 5][:min(5, total_pages)]:
        try:
            img = pdf_converter.render_page(pdf_path, page_num)
            print(f"  ✅ Page {page_num}: {len(img.image_base64)} bytes (base64)")
        except Exception as e:
            print(f"  ❌ Page {page_num}: Error - {e}")

    # Show prompts
    print("\n" + "-" * 60)
    print("PROMPTS BEING USED")
    print("-" * 60)

    print("\n📝 SYSTEM PROMPT (first 500 chars):")
    print(CONTEXT_ANALYSIS_SYSTEM[:500] + "...")

    print("\n📝 USER PROMPT (first 1000 chars):")
    print(CONTEXT_ANALYSIS_USER[:1000] + "...")

    # Test with API if key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n⚠️  No OPENAI_API_KEY found - skipping API test")
        print("   Set OPENAI_API_KEY to test the actual API response")
        return

    print("\n" + "-" * 60)
    print("API TEST")
    print("-" * 60)

    from openai import OpenAI
    client = OpenAI(api_key=api_key)

    # Use better sample pages
    print(f"\n🔄 Calling API with pages: {sample_pages_better[:4]}")

    images = pdf_converter.render_pages_batch(pdf_path, sample_pages_better[:4])

    content = []
    for i, img in enumerate(images):
        print(f"   Adding image {i+1}: page {sample_pages_better[i]}")
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:{img.mime_type};base64,{img.image_base64}"
            }
        })

    content.append({
        "type": "text",
        "text": CONTEXT_ANALYSIS_USER,
    })

    print("\n⏳ Calling OpenAI API...")

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            max_tokens=4096,
            temperature=0.1,
            messages=[
                {"role": "system", "content": CONTEXT_ANALYSIS_SYSTEM},
                {"role": "user", "content": content},
            ],
        )

        raw_response = response.choices[0].message.content

        print(f"\n✅ API Response received!")
        print(f"   Input tokens: {response.usage.prompt_tokens}")
        print(f"   Output tokens: {response.usage.completion_tokens}")

        print("\n" + "-" * 60)
        print("RAW RESPONSE")
        print("-" * 60)
        print(raw_response)

        # Try to parse
        print("\n" + "-" * 60)
        print("PARSED RESPONSE")
        print("-" * 60)

        # Extract JSON
        if "```json" in raw_response:
            start = raw_response.find("```json") + 7
            end = raw_response.find("```", start)
            json_str = raw_response[start:end].strip()
        elif "{" in raw_response:
            start = raw_response.find("{")
            depth = 0
            for i, char in enumerate(raw_response[start:], start):
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        json_str = raw_response[start:i+1]
                        break
        else:
            json_str = raw_response

        data = json.loads(json_str)

        print(f"\n📋 Document Type: {data.get('document_type')}")
        print(f"📋 Title: {data.get('title')}")
        print(f"📋 Institution: {data.get('institution')}")
        print(f"📋 Faculty: {data.get('faculty')}")
        print(f"📋 Degree Program: {data.get('degree_program')}")
        print(f"📋 Version Date: {data.get('version_date')}")

        print(f"\n📚 CHAPTERS ({len(data.get('chapters', []))} entries):")
        for ch in data.get('chapters', []):
            print(f"   - {ch}")

        print(f"\n📖 ABBREVIATIONS ({len(data.get('abbreviations', []))} entries):")
        abbrevs = data.get('abbreviations', [])
        if isinstance(abbrevs, list):
            for a in abbrevs[:10]:
                if isinstance(a, dict):
                    print(f"   - {a.get('short')}: {a.get('long')}")
                else:
                    print(f"   - {a}")
        elif isinstance(abbrevs, dict):
            for k, v in list(abbrevs.items())[:10]:
                print(f"   - {k}: {v}")

        print(f"\n🔑 KEY TERMS ({len(data.get('key_terms', []))} entries):")
        for term in data.get('key_terms', [])[:10]:
            print(f"   - {term}")

        # Check for issues
        print("\n" + "-" * 60)
        print("ISSUE DETECTION")
        print("-" * 60)

        chapters = data.get('chapters', [])
        if len(chapters) < 5:
            print(f"⚠️  PROBLEM: Only {len(chapters)} chapters found (expected 10-20)")
        else:
            print(f"✅ Good: {len(chapters)} chapters found")

        has_teile = any("Teil" in ch or ch.startswith("I.") for ch in chapters)
        if not has_teile:
            print("⚠️  PROBLEM: No 'Teil I/II/III/IV' found in chapters")
        else:
            print("✅ Good: 'Teil' entries found in chapters")

        has_anlagen = any("Anlage" in ch for ch in chapters)
        if not has_anlagen:
            print("⚠️  PROBLEM: No 'Anlage' found in chapters")
        else:
            print("✅ Good: 'Anlage' entries found in chapters")

        abbrevs = data.get('abbreviations', [])
        if not abbrevs:
            print("⚠️  PROBLEM: No abbreviations found")
        else:
            print(f"✅ Good: {len(abbrevs)} abbreviations found")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/debug_context_analysis.py <pdf_path>")
        print("\nExample:")
        print("  python scripts/debug_context_analysis.py pdfs/stpo_bsc-informatik_25-01-23_lese.pdf")
        sys.exit(1)

    debug_context_analysis(sys.argv[1])
