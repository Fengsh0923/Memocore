"""Tests for memocore.core.locale"""

import pytest
from memocore.core.locale import t


class TestLocale:
    def test_chinese_default(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_LANG", raising=False)
        result = t("ui.scope_personal")
        assert result == "\u4e2a\u4eba"

    def test_chinese(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LANG", "zh")
        result = t("ui.scope_personal")
        assert result == "\u4e2a\u4eba"

    def test_format_args(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LANG", "en")
        result = t("ui.recall_header_session", name="TestBot")
        assert "TestBot" in result

    def test_missing_key(self):
        result = t("nonexistent.key")
        assert "MISSING_LOCALE" in result

    def test_all_keys_have_en(self):
        from memocore.core.locale import _STRINGS
        for key, entry in _STRINGS.items():
            assert "en" in entry, f"Key {key} is missing 'en' translation"

    def test_all_keys_have_zh(self):
        from memocore.core.locale import _STRINGS
        for key, entry in _STRINGS.items():
            assert "zh" in entry, f"Key {key} is missing 'zh' translation"

    def test_dream_compile_entity_format(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LANG", "en")
        result = t(
            "dream.compile_entity",
            entity_name="Alice",
            entity_type="Person",
            confidence_label="high",
            fact_count=5,
            facts_text="fact1\nfact2",
        )
        assert "Alice" in result
        assert "Person" in result

    def test_privacy_prompt_format(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LANG", "en")
        result = t("privacy.llm_check_prompt", text="some text here")
        assert "some text here" in result
