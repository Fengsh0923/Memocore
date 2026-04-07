#!/usr/bin/env python3
"""
Memocore Mid-Session Hook for Claude Code
Trigger: every N user messages (UserPromptSubmit), configurable via MEMOCORE_MID_SESSION_INTERVAL
Function: read partial session transcript -> extract memory -> write to knowledge graph
          This ensures memory is preserved even if session is interrupted (e.g. /new)

Environment variables:
  MEMOCORE_MID_SESSION_INTERVAL  — how many user messages between writes (default: 5)
  MEMOCORE_AGENT_ID              — agent namespace (default "default")
"""

import sys
import json
import asyncio
import logging
import os
import re
import subprocess
from pathlib import Path
from datetime import datetime

from memocore.core.config import get_agent_id, validate_agent_id, get_sessions_dir, get_logs_dir
from memocore.agents.registry import get_profile as _get_profile

logger = logging.getLogger("memocore.mid_session_hook")

PROJECTS_DIR = Path.home() / ".claude" / "projects"
DEFAULT_INTERVAL = 5  # write every N user messages


def _configure_logging():
    log_file = get_logs_dir() / "mid_session_hook.log"
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [mid_session_hook] %(levelname)s %(message)s"
    ))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def get_interval() -> int:
    try:
        return int(os.environ.get("MEMOCORE_MID_SESSION_INTERVAL", DEFAULT_INTERVAL))
    except (ValueError, TypeError):
        return DEFAULT_INTERVAL


def get_counter_path(session_id: str, agent_id: str) -> Path:
    safe_sid = re.sub(r'[^a-zA-Z0-9_\-]', '', session_id)[:32]
    safe_aid = re.sub(r'[^a-zA-Z0-9_\-]', '', agent_id)[:16]
    return get_sessions_dir() / f"midcount-{safe_aid}-{safe_sid}.txt"


def increment_counter(session_id: str, agent_id: str) -> int:
    """Atomically increment message counter, return new count."""
    path = get_counter_path(session_id, agent_id)
    try:
        count = int(path.read_text().strip()) if path.exists() else 0
    except (ValueError, OSError):
        count = 0
    count += 1
    path.write_text(str(count))
    return count


def find_session_jsonl(session_id: str) -> Path | None:
    """Search all project dirs for the session jsonl file."""
    safe_sid = re.sub(r'[^a-zA-Z0-9_\-]', '', session_id)
    for p in PROJECTS_DIR.rglob(f"{safe_sid}.jsonl"):
        return p
    return None


def parse_session_jsonl(jsonl_path: Path, profile: dict) -> str:
    """Parse Claude Code session jsonl into transcript text."""
    user_name = profile.get("user_display_name", "User")
    assistant_name = profile.get("assistant_display_name", "Assistant")

    lines = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for raw in f:
                try:
                    d = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                turn_type = d.get("type", "")
                if turn_type not in ("user", "assistant"):
                    continue
                msg = d.get("message", {})
                content = msg.get("content", "")
                if isinstance(content, list):
                    text = " ".join(
                        c.get("text", "")
                        for c in content
                        if isinstance(c, dict) and c.get("type") == "text"
                    )
                else:
                    text = str(content)

                text = text.strip()
                if not text or len(text) < 5:
                    continue
                # Truncate very long turns
                if len(text) > 1500:
                    text = text[:1500] + "..."
                prefix = user_name if turn_type == "user" else assistant_name
                lines.append(f"{prefix}: {text}")
    except OSError as e:
        logger.warning(f"Cannot read jsonl: {e}")

    return "\n".join(lines)


async def run_extraction(session_id: str, agent_id: str):
    """Read session transcript and extract to Memocore."""
    from memocore.core.extractor import extract_and_store

    profile = _get_profile(agent_id)
    jsonl_path = find_session_jsonl(session_id)

    if not jsonl_path:
        logger.warning(f"session={session_id[:16]} jsonl not found, skipping mid-session write")
        return

    transcript_text = parse_session_jsonl(jsonl_path, profile)

    if len(transcript_text) < 200:
        logger.info(f"session={session_id[:16]} transcript too short ({len(transcript_text)} chars), skipping")
        return

    logger.info(f"session={session_id[:16]} mid-session extraction start, chars={len(transcript_text)}")

    result = await extract_and_store(
        conversation=transcript_text,
        agent_id=agent_id,
        source_description=(
            f"mid-session | session={session_id[:16]} | "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M')}"
        ),
    )

    if result["success"]:
        logger.info(
            f"session={session_id[:16]} mid-session extraction succeeded: "
            f"episode={result['episode_name']} entities={result.get('entities_extracted', '?')}"
        )
    else:
        logger.error(f"session={session_id[:16]} mid-session extraction failed: {result.get('error')}")


def spawn_extraction_background(session_id: str, agent_id: str):
    """Launch extraction as a detached background subprocess so hook returns fast."""
    script = Path(__file__)
    try:
        subprocess.Popen(
            [sys.executable, str(script), "--extract", session_id, agent_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env={**os.environ},
        )
        logger.info(f"session={session_id[:16]} spawned background extraction subprocess")
    except Exception as e:
        logger.warning(f"session={session_id[:16]} failed to spawn background extraction: {e}")


def main():
    _configure_logging()

    # Sub-mode: called as background extraction worker
    if len(sys.argv) >= 4 and sys.argv[1] == "--extract":
        session_id = sys.argv[2]
        agent_id = sys.argv[3]
        try:
            asyncio.run(run_extraction(session_id, agent_id))
        except Exception as e:
            logger.error(f"background extraction failed: {e}")
        return

    # Normal mode: called by UserPromptSubmit hook
    try:
        raw = sys.stdin.read()
        hook_input = json.loads(raw) if raw.strip() else {}
    except Exception:
        hook_input = {}

    session_id = hook_input.get("session_id", "unknown")
    agent_id = validate_agent_id(get_agent_id())
    interval = get_interval()

    count = increment_counter(session_id, agent_id)
    logger.info(f"session={session_id[:16]} msg_count={count} interval={interval}")

    if count % interval == 0:
        logger.info(f"session={session_id[:16]} interval reached ({count}), spawning background extraction")
        spawn_extraction_background(session_id, agent_id)

    # Always exit 0 — never block the conversation
    sys.exit(0)


if __name__ == "__main__":
    main()
