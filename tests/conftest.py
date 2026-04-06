"""
Shared fixtures for Memocore unit tests.
All tests are pure unit tests — no Neo4j or LLM connections required.

NOTE: Tests import specific submodules (e.g. memocore.core.config) directly,
NOT via memocore.core — the __init__.py re-exports require graphiti_core.
"""

import os
import sys
import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Isolate each test from the real environment."""
    monkeypatch.setenv("MEMOCORE_STATE_DIR", str(tmp_path / "memocore_state"))
    monkeypatch.setenv("MEMOCORE_PRIVACY_ENABLED", "true")
    monkeypatch.setenv("MEMOCORE_PRIVACY_BLACKLIST", "")
    # Clear singleton caches that depend on env
    import memocore.core.privacy as priv_mod
    priv_mod._default_filter = None


@pytest.fixture
def mock_chat_complete(monkeypatch):
    """Replace chat_complete with a controllable mock."""
    import memocore.core.llm_adapter as llm

    responses = []

    async def _fake_chat_complete(prompt, **kwargs):
        if responses:
            return responses.pop(0)
        return '{"action": "skip", "reason": "mocked"}'

    monkeypatch.setattr(llm, "chat_complete", _fake_chat_complete)
    return responses
