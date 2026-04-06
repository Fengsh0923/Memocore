"""Tests for memocore.core.privacy"""

import pytest
from memocore.core.privacy import PrivacyFilter, PrivacyReport


class TestRedactPatterns:
    def test_openai_key_redacted(self):
        f = PrivacyFilter()
        text = "My key is sk-abcdefghij1234567890abcdef"
        result, report = f.process(text)
        assert "[OPENAI_KEY]" in result
        assert "sk-abcdefghij" not in result
        assert report.redacted_count >= 1

    def test_anthropic_key_redacted_correctly(self):
        """Anthropic key (sk-ant-...) should be labeled ANTHROPIC_KEY, not OPENAI_KEY."""
        f = PrivacyFilter()
        text = "key: sk-ant-api03-abcdefghij1234567890abcdef"
        result, report = f.process(text)
        assert "[ANTHROPIC_KEY]" in result
        assert "[OPENAI_KEY]" not in result
        assert "anthropic_key" in report.redacted_types

    def test_github_token(self):
        f = PrivacyFilter()
        text = "token: ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        result, report = f.process(text)
        assert "[GITHUB_TOKEN]" in result

    def test_phone_number(self):
        f = PrivacyFilter()
        text = "联系方式 13812345678 备用"
        result, report = f.process(text)
        assert "[PHONE]" in result
        assert "13812345678" not in result

    def test_id_number(self):
        f = PrivacyFilter()
        text = "身份证 110101199001011234"
        result, report = f.process(text)
        assert "[ID_NUMBER]" in result

    def test_pem_key_skips(self):
        f = PrivacyFilter()
        text = "-----BEGIN PRIVATE KEY-----\nMIIE...\n-----END PRIVATE KEY-----"
        _, report = f.process(text)
        assert report.should_skip is True

    def test_clean_text_passes(self):
        f = PrivacyFilter()
        text = "今天天气不错，我们讨论了项目架构。"
        result, report = f.process(text)
        assert result == text
        assert report.redacted_count == 0
        assert not report.should_skip


class TestBlacklist:
    def test_blacklist_triggers_skip(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_PRIVACY_BLACKLIST", "secret_project,classified")
        import memocore.core.privacy as mod
        mod._default_filter = None
        f = PrivacyFilter()
        _, report = f.process("This is about the secret_project deployment")
        assert report.should_skip is True
        assert "blacklist" in report.skip_reason


class TestDisabled:
    def test_disabled_passes_through(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_PRIVACY_ENABLED", "false")
        f = PrivacyFilter()
        text = "sk-ant-api03-abcdefghij1234567890abcdef"
        result, report = f.process(text)
        assert result == text
        assert report.redacted_count == 0


class TestPrivacyReport:
    def test_str_clean(self):
        assert str(PrivacyReport()) == "CLEAN"

    def test_str_skip(self):
        r = PrivacyReport(should_skip=True, skip_reason="test")
        assert "SKIP" in str(r)

    def test_str_redacted(self):
        r = PrivacyReport(redacted_count=2, redacted_types=["openai_key", "phone"])
        assert "REDACTED 2" in str(r)
