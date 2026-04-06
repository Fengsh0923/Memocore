"""Tests for memocore.core.config"""

import pytest
from pathlib import Path

from memocore.core.config import (
    get_state_dir,
    get_sessions_dir,
    get_agent_id,
    get_lang,
    get_neo4j_config,
    get_dream_interval,
    get_dream_ttl_days,
    increment_session_counter,
    reset_session_counter,
    should_run_dream,
    cleanup_old_session_flags,
    is_privacy_enabled,
    validate_agent_id,
)


class TestPaths:
    def test_state_dir_created(self, tmp_path, monkeypatch):
        d = tmp_path / "mc_state"
        monkeypatch.setenv("MEMOCORE_STATE_DIR", str(d))
        result = get_state_dir()
        assert result == d
        assert d.is_dir()

    def test_sessions_dir_is_subdir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMOCORE_STATE_DIR", str(tmp_path))
        sd = get_sessions_dir()
        assert sd.parent == tmp_path
        assert sd.name == "sessions"
        assert sd.is_dir()


class TestAgentId:
    def test_default(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_AGENT_ID", raising=False)
        assert get_agent_id() == "default"

    def test_custom(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_AGENT_ID", "my-agent")
        assert get_agent_id() == "my-agent"


class TestNeo4jConfig:
    def test_returns_dict(self):
        cfg = get_neo4j_config()
        assert isinstance(cfg, dict)
        assert "uri" in cfg
        assert "user" in cfg
        assert "password" in cfg

    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("NEO4J_URI", raising=False)
        monkeypatch.delenv("NEO4J_USER", raising=False)
        monkeypatch.delenv("NEO4J_PASSWORD", raising=False)
        cfg = get_neo4j_config()
        assert cfg["uri"] == "bolt://localhost:7687"
        assert cfg["user"] == "neo4j"


class TestDreamConfig:
    def test_dream_interval_default(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_DREAM_INTERVAL", raising=False)
        assert get_dream_interval() == 5

    def test_dream_interval_custom(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_DREAM_INTERVAL", "10")
        assert get_dream_interval() == 10

    def test_dream_interval_invalid(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_DREAM_INTERVAL", "bad")
        assert get_dream_interval() == 5

    def test_ttl_default(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_DREAM_TTL_DAYS", raising=False)
        assert get_dream_ttl_days() == 90


class TestSessionCounter:
    def test_increment_and_reset(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMOCORE_STATE_DIR", str(tmp_path))
        assert increment_session_counter("test") == 1
        assert increment_session_counter("test") == 2
        reset_session_counter("test")
        assert increment_session_counter("test") == 1

    def test_should_run_dream(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MEMOCORE_STATE_DIR", str(tmp_path))
        monkeypatch.setenv("MEMOCORE_DREAM_INTERVAL", "3")
        assert not should_run_dream("test")  # 1
        assert not should_run_dream("test")  # 2
        assert should_run_dream("test")       # 3 -> triggers, resets
        assert not should_run_dream("test")  # 1 again


class TestCleanupSessionFlags:
    def test_deletes_old_flags(self, tmp_path, monkeypatch):
        import time
        monkeypatch.setenv("MEMOCORE_STATE_DIR", str(tmp_path))
        sessions = get_sessions_dir()
        # Create a "stale" flag
        old_flag = sessions / "old.flag"
        old_flag.touch()
        # Backdate it 72 hours
        import os
        old_time = time.time() - (72 * 3600)
        os.utime(old_flag, (old_time, old_time))
        # Create a "fresh" flag
        new_flag = sessions / "new.flag"
        new_flag.touch()
        deleted = cleanup_old_session_flags(max_age_hours=48)
        assert deleted == 1
        assert not old_flag.exists()
        assert new_flag.exists()


class TestPrivacyConfig:
    def test_enabled_by_default(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_PRIVACY_ENABLED", raising=False)
        assert is_privacy_enabled() is True

    def test_disabled(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_PRIVACY_ENABLED", "false")
        assert is_privacy_enabled() is False


class TestLang:
    def test_default_zh(self, monkeypatch):
        monkeypatch.delenv("MEMOCORE_LANG", raising=False)
        assert get_lang() == "zh"

    def test_custom_zh(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LANG", "zh")
        assert get_lang() == "zh"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("MEMOCORE_LANG", "ZH")
        assert get_lang() == "zh"


class TestValidateAgentId:
    def test_valid_simple(self):
        assert validate_agent_id("my-agent") == "my-agent"

    def test_valid_with_colon_dot(self):
        assert validate_agent_id("org:tenant.123") == "org:tenant.123"

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_agent_id("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            validate_agent_id("   ")

    def test_too_long_raises(self):
        with pytest.raises(ValueError, match="too long"):
            validate_agent_id("a" * 200)

    def test_invalid_chars_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_agent_id("../../etc/passwd")

    def test_spaces_raises(self):
        with pytest.raises(ValueError, match="invalid characters"):
            validate_agent_id("my agent")

    def test_strips_whitespace(self):
        assert validate_agent_id("  valid-id  ") == "valid-id"
