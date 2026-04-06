"""
Agent Profile Registry

Profiles define how memory extraction and recall behave for a given agent_id.
Enterprise operators register profiles at startup; all other agents get
DEFAULT_PROFILE with DEFAULT_ENTITY_TYPES.

Usage:
    from memocore.agents.registry import get_profile, get_entity_types, register_profile

    # Register at app startup (enterprise: loop over 4000+ agents)
    register_profile("my-agent-id", profile=MY_PROFILE, entity_types=MY_TYPES)

    # Use in extraction / retrieval (returns default if unregistered)
    profile = get_profile("any-agent-id")
    types   = get_entity_types("any-agent-id")
"""

from typing import Optional

from memocore.agents.default.schema import DEFAULT_ENTITY_TYPES, get_default_profile

# Registry storage: agent_id -> (profile_dict, entity_types_dict)
_registry: dict[str, tuple[dict, dict]] = {}




def register_profile(
    agent_id: str,
    profile: dict,
    entity_types: Optional[dict] = None,
) -> None:
    """
    Register a custom profile for agent_id.
    Call this at application startup before any extraction/retrieval.
    """
    _registry[agent_id] = (profile, entity_types or {})


def get_profile(agent_id: str) -> dict:
    """Return the registered profile, or language-aware default if unregistered."""
    entry = _registry.get(agent_id)
    return entry[0] if entry else get_default_profile()


def get_entity_types(agent_id: str) -> dict:
    """Return the registered entity types, or DEFAULT_ENTITY_TYPES if unregistered."""
    entry = _registry.get(agent_id)
    if entry:
        return entry[1] if entry[1] else DEFAULT_ENTITY_TYPES
    return DEFAULT_ENTITY_TYPES


def is_registered(agent_id: str) -> bool:
    return agent_id in _registry


def list_registered() -> list[str]:
    return list(_registry.keys())
