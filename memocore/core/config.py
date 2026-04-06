"""
Memocore Central Configuration

Configuration loading priority (high to low):
  1. Environment variables
  2. Project directory .env
  3. Global config file ~/.memocore/config.env (for persistent API keys etc.)

Supported configuration items:
  MEMOCORE_AGENT_ID          — Agent namespace (default "default")
  MEMOCORE_LANG              — Language: "zh" | "en" (default "zh", affects LLM prompts and output)
  MEMOCORE_LLM_PROVIDER      — LLM provider: "anthropic" | "openai" (default: auto-detect)
  MEMOCORE_ANTHROPIC_MODEL   — Anthropic model (default claude-haiku-4-5-20251001)
  MEMOCORE_OPENAI_MODEL      — OpenAI model (default gpt-4o-mini)
  MEMOCORE_EMBED_PROVIDER    — Embedding provider: "openai" | "local" | "auto"
  MEMOCORE_EMBED_MODEL       — OpenAI embedding model
  MEMOCORE_LOCAL_EMBED_MODEL — Local embedding model name
  MEMOCORE_DREAM_INTERVAL    — Dream trigger interval (session count, default 5)
  MEMOCORE_DREAM_TTL_DAYS    — Memory TTL (expire after this many days if unreferenced, default 90)
  MEMOCORE_PRIVACY_ENABLED   — Enable privacy filtering (default true)
  MEMOCORE_STATE_DIR         — State file directory (default ~/.memocore)
  NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
  ANTHROPIC_API_KEY / OPENAI_API_KEY
"""

import fcntl
import hashlib
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

_logger = logging.getLogger("memocore.config")


def _get_raw_state_dir() -> Path:
    """Return path without creating directory (used during config init)."""
    custom = os.getenv("MEMOCORE_STATE_DIR")
    if custom:
        return Path(custom)
    try:
        return Path.home() / ".memocore"
    except (RuntimeError, KeyError):
        # Fallback for containerised environments with no HOME
        return Path("/tmp/.memocore")


def _load_global_config():
    """
    Load all .env config centrally (other modules do not need their own load_dotenv).
    Loading order (override=False, first-come-first-served, env vars take priority):
      1. ~/.memocore/config.env (global persistent config)
      2. Project root .env (for development)
    """
    try:
        config_file = _get_raw_state_dir() / "config.env"
        if config_file.exists():
            load_dotenv(config_file, override=False)
    except Exception as e:
        _logger.debug(f"Failed to load global config: {e}")
    # Try loading project root .env (relative to this file's package)
    try:
        pkg_root = Path(__file__).parent.parent
        project_env = pkg_root / ".env"
        if project_env.exists():
            load_dotenv(project_env, override=False)
    except Exception as e:
        _logger.debug(f"Failed to load project .env: {e}")


# Auto-load global config at module import time
_load_global_config()


# ── Base paths ────────────────────────────────────────────────────────────────

def get_state_dir() -> Path:
    """Return memocore state directory, auto-creating if needed."""
    base = _get_raw_state_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_sessions_dir() -> Path:
    """Session flag file directory (replaces /tmp, persists across restarts)."""
    d = get_state_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logs_dir() -> Path:
    """Log directory."""
    d = get_state_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_global_config_path() -> Path:
    """Global config file path."""
    return get_state_dir() / "config.env"


# ── Agent configuration ──────────────────────────────────────────────────────

def get_agent_id() -> str:
    """Read agent_id from env var / config file, default 'default'."""
    return os.getenv("MEMOCORE_AGENT_ID", "default")


def get_lang() -> str:
    """Return language setting: 'zh' | 'en' (default 'zh')."""
    return os.getenv("MEMOCORE_LANG", "zh").lower()


def validate_agent_id(agent_id: str) -> str:
    """Validate and sanitize agent_id for safe use in file paths and queries."""
    import re
    agent_id = agent_id.strip()
    if not agent_id:
        raise ValueError("agent_id cannot be empty")
    if len(agent_id) > 128:
        raise ValueError(f"agent_id too long ({len(agent_id)} > 128)")
    if not re.match(r'^[a-zA-Z0-9_\-:.]+$', agent_id):
        raise ValueError(f"agent_id contains invalid characters: {agent_id!r}")
    return agent_id


def validate_scope_id(scope_id: str, label: str = "scope_id") -> str:
    """Validate team_id or tenant_id — same rules as agent_id."""
    import re
    scope_id = scope_id.strip()
    if not scope_id:
        raise ValueError(f"{label} cannot be empty")
    if len(scope_id) > 128:
        raise ValueError(f"{label} too long ({len(scope_id)} > 128)")
    if not re.match(r'^[a-zA-Z0-9_\-:.]+$', scope_id):
        raise ValueError(f"{label} contains invalid characters: {scope_id!r}")
    return scope_id


def validate_identifier(value: str, label: str = "identifier") -> str:
    """
    Generic identifier validator for agent_id, team_id, tenant_id, and similar fields.
    Applies the same rules as validate_agent_id / validate_scope_id:
      - Non-empty after stripping whitespace
      - At most 128 characters
      - Only alphanumeric characters plus _ - : .
    Returns the stripped value on success; raises ValueError otherwise.
    """
    import re
    value = value.strip()
    if not value:
        raise ValueError(f"{label} cannot be empty")
    if len(value) > 128:
        raise ValueError(f"{label} too long ({len(value)} > 128)")
    if not re.match(r'^[a-zA-Z0-9_\-:.]+$', value):
        raise ValueError(f"{label} contains invalid characters: {value!r}")
    return value


def make_safe_agent_key(agent_id: str) -> str:
    """
    Return a short, collision-resistant key suitable for use in session flag filenames.

    Uses a SHA-256 hash of the full agent_id (hex, first 16 chars) instead of
    raw truncation, which would collide for UUID-prefixed multi-tenant IDs that
    share a common prefix.

    Example:
        make_safe_agent_key("tenant-abc/agent-123") -> "e3b0c44298fc1c14"
    """
    return hashlib.sha256(agent_id.encode()).hexdigest()[:16]


# ── General getter ───────────────────────────────────────────────────────────

def get(key: str, default: str = "") -> str:
    """Read any config item (env var or global config.env)."""
    return os.getenv(key, default)


def get_neo4j_config() -> dict:
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", ""),
    }


# ── Dream configuration ─────────────────────────────────────────────────────

def get_dream_interval() -> int:
    """Dream trigger interval (number of sessions)."""
    try:
        return int(os.getenv("MEMOCORE_DREAM_INTERVAL", "5"))
    except ValueError:
        return 5


def get_dream_ttl_days() -> int:
    """Memory TTL: expire after this many days if unreferenced."""
    try:
        return int(os.getenv("MEMOCORE_DREAM_TTL_DAYS", "90"))
    except ValueError:
        return 90


def _get_counter_path(agent_id: str) -> Path:
    safe_key = make_safe_agent_key(agent_id)
    return get_state_dir() / f"dream_counter_{safe_key}.txt"


def increment_session_counter(agent_id: str) -> int:
    """Atomically increment session counter using file locking."""
    path = _get_counter_path(agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Open in read+write mode, creating if needed
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = os.read(fd, 64).decode().strip()
        current = int(data) if data else 0
        current += 1
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(current).encode())
        return current
    except (ValueError, OSError):
        return 1
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def reset_session_counter(agent_id: str) -> None:
    """Atomically reset session counter using file locking."""
    path = _get_counter_path(agent_id)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, b"0")
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def should_run_dream(agent_id: str) -> bool:
    """Check if dream should run. Atomic increment+check+reset under a single file lock."""
    interval = get_dream_interval()
    path = _get_counter_path(agent_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        data = os.read(fd, 64).decode().strip()
        current = int(data) if data else 0
        current += 1
        if current >= interval:
            # Reset and signal dream
            current = 0
            trigger = True
        else:
            trigger = False
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, str(current).encode())
        return trigger
    except (ValueError, OSError):
        return False
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


# ── Privacy configuration ────────────────────────────────────────────────────

def is_privacy_enabled() -> bool:
    return os.getenv("MEMOCORE_PRIVACY_ENABLED", "true").lower() != "false"


def cleanup_old_session_flags(max_age_hours: int = 48) -> int:
    """Delete session flag files older than max_age_hours. Returns count deleted."""
    import time
    sessions_dir = get_sessions_dir()
    cutoff = time.time() - (max_age_hours * 3600)
    deleted = 0
    for flag in sessions_dir.glob("*.flag"):
        try:
            if flag.stat().st_mtime < cutoff:
                flag.unlink()
                deleted += 1
        except OSError:
            pass
    return deleted


def get_privacy_blacklist() -> list[str]:
    """Read custom blacklist words from config (comma-separated)."""
    raw = os.getenv("MEMOCORE_PRIVACY_BLACKLIST", "")
    return [w.strip() for w in raw.split(",") if w.strip()]


# ── Config writing (for memocore init) ───────────────────────────────────────

def write_global_config(values: dict) -> Path:
    """
    Write key=value pairs to global config file ~/.memocore/config.env.
    Does not overwrite existing config lines, only appends/updates.
    """
    config_path = get_global_config_path()
    existing: dict[str, str] = {}

    if config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    existing.update(values)

    lines = [
        "# Memocore global configuration",
        "# Generated by memocore init, can be edited manually",
        "# Priority: lower than env vars, higher than defaults",
        "",
    ]
    for k, v in existing.items():
        # Strip newlines/carriage returns to prevent config injection
        safe_v = v.replace('\n', '').replace('\r', '')
        lines.append(f"{k}={safe_v}")

    config_path.write_text("\n".join(lines) + "\n")
    return config_path
