import pytest
from app.core.preprocessing import preprocess_prompt


def test_strip_leading_and_trailing_whitespace():
    raw_prompt = "   \t\n  Hello, this is a test prompt!   \n\t  "
    cleaned = preprocess_prompt(raw_prompt)
    assert cleaned == "Hello, this is a test prompt!"


def test_reduce_consecutive_newlines():
    raw_prompt = "Paragraph 1\n\n\n\n\nParagraph 2\n\n\nParagraph 3"
    cleaned = preprocess_prompt(raw_prompt)
    assert cleaned == "Paragraph 1\n\nParagraph 2\n\nParagraph 3"


def test_windows_crlf_normalization():
    raw_prompt = "Line 1\r\n\r\n\r\n\r\nLine 2\r\nLine 3"
    cleaned = preprocess_prompt(raw_prompt)
    assert cleaned == "Line 1\n\nLine 2\nLine 3"


def test_unprintable_characters_removal():
    # \x00 (null byte) and \x07 (bell) should be stripped; \n and \t should remain
    raw_prompt = "Safe text\x00 with unprintable\x07 chars\nand normal tabs\there."
    cleaned = preprocess_prompt(raw_prompt)
    assert cleaned == "Safe text with unprintable chars\nand normal tabs\there."


def test_empty_and_whitespace_only():
    assert preprocess_prompt("") == ""
    assert preprocess_prompt("   \n\n\t  ") == ""
    assert preprocess_prompt(None) == ""
