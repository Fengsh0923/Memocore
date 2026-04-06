"""Tests for memocore.agents.registry"""

import pytest

from memocore.agents.registry import (
    register_profile,
    get_profile,
    get_entity_types,
    is_registered,
    list_registered,
    _registry,
)
from memocore.agents.default.schema import DEFAULT_ENTITY_TYPES, get_default_profile


@pytest.fixture(autouse=True)
def _clean_registry():
    """Save and restore registry state around each test."""
    saved = dict(_registry)
    yield
    _registry.clear()
    _registry.update(saved)


class TestRegistration:
    def test_register_and_retrieve(self):
        profile = {"user_display_name": "Test"}
        types = {"Foo": object}
        register_profile("test-agent", profile, types)
        assert get_profile("test-agent") == profile
        assert get_entity_types("test-agent") == types
        assert is_registered("test-agent")

    def test_unregistered_returns_defaults(self):
        assert get_profile("nonexistent") == get_default_profile()
        assert get_entity_types("nonexistent") == DEFAULT_ENTITY_TYPES
        assert not is_registered("nonexistent")

    def test_register_without_entity_types(self):
        profile = {"user_display_name": "NoTypes"}
        register_profile("no-types-agent", profile)
        assert get_profile("no-types-agent") == profile
        # No entity_types registered → should return DEFAULT
        assert get_entity_types("no-types-agent") == DEFAULT_ENTITY_TYPES


class TestBootstrap:
    def test_no_auto_registered_profiles(self):
        """No profiles should be auto-registered at import time."""
        assert len(list_registered()) == 0

    def test_manual_registration_works(self):
        """Profiles can be registered manually at application startup."""
        register_profile("custom-agent", {"user_display_name": "Custom"})
        assert is_registered("custom-agent")
        assert get_profile("custom-agent")["user_display_name"] == "Custom"


class TestListRegistered:
    def test_list_includes_registered(self):
        register_profile("list-test", {"x": 1})
        assert "list-test" in list_registered()
