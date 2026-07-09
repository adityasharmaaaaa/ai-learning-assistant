import pytest

from app.utils.json_parser import JSONExtractionError, extract_json


def test_extract_clean_json():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_from_markdown_fence():
    raw = '```json\n{"a": 1, "b": [1,2,3]}\n```'
    assert extract_json(raw) == {"a": 1, "b": [1, 2, 3]}


def test_extract_json_with_preamble_and_trailing_text():
    raw = 'Sure! Here is the JSON you asked for:\n{"a": 1}\nLet me know if you need anything else.'
    assert extract_json(raw) == {"a": 1}


def test_extract_json_raises_on_garbage():
    with pytest.raises(JSONExtractionError):
        extract_json("this is not json at all")
