"""Tests for memocore.core.llm_adapter — parse_llm_json and detect_provider"""

import json
import pytest

from memocore.core.llm_adapter import parse_llm_json, _detect_provider


class TestParseLlmJson:
    def test_plain_json_object(self):
        result = parse_llm_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_plain_json_array(self):
        result = parse_llm_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_markdown_fence(self):
        text = '```json\n{"action": "merge", "keep_uuid": "abc"}\n```'
        result = parse_llm_json(text)
        assert result["action"] == "merge"

    def test_markdown_fence_no_language(self):
        text = '```\n{"x": 1}\n```'
        result = parse_llm_json(text)
        assert result == {"x": 1}

    def test_prose_preamble(self):
        text = 'Here is the result:\n\n{"action": "skip", "reason": "insufficient data"}'
        result = parse_llm_json(text)
        assert result["action"] == "skip"

    def test_no_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            parse_llm_json("No JSON here at all.")

    def test_whitespace_handling(self):
        result = parse_llm_json('  \n  {"a": 1}  \n  ')
        assert result == {"a": 1}

    def test_nested_json(self):
        text = '{"outer": {"inner": 42}}'
        result = parse_llm_json(text)
        assert result["outer"]["inner"] == 42


class TestDetectProvider:
    def test_explicit_anthropic(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LLM_PROVIDER", "anthropic")
        assert _detect_provider() == "anthropic"

    def test_explicit_openai(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LLM_PROVIDER", "openai")
        assert _detect_provider() == "openai"

    def test_auto_detect_anthropic(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_LLM_PROVIDER", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert _detect_provider() == "anthropic"

    def test_auto_detect_openai(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert _detect_provider() == "openai"

    def test_fallback_anthropic(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_LLM_PROVIDER", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        assert _detect_provider() == "anthropic"
