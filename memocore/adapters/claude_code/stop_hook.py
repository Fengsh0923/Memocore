#!/usr/bin/env python3
"""
Memocore Stop Hook for Claude Code
Trigger: end of each conversation (Claude Code Stop event)
Function: read conversation transcript -> extract memory -> write to knowledge graph
          Trigger Dream consolidation in background every MEMOCORE_DREAM_INTERVAL sessions.

Config (in ~/.claude/settings.json):
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "python3 /path/to/memocore/adapters/claude_code/stop_hook.py",
        "timeout": 60
      }]
    }]
  }
}

Environment variables:
  MEMOCORE_AGENT_ID         — Agent namespace (default "default")
  MEMOCORE_DREAM_INTERVAL   — Dream trigger interval in sessions (default 5)
  MEMOCORE_LLM_PROVIDER     — LLM provider (default: auto-detect)
"""

import sys
import json
import asyncio
import logging
import subprocess
from pathlib import Path
from datetime import datetime

from memocore.core.config import get_agent_id, should_run_dream, validate_agent_id
from memocore.agents.registry import get_profile as _get_profile

logger = logging.getLogger("memocore.stop_hook")


def _configure_logging():
    from memocore.core.config import get_logs_dir
    log_file = get_logs_dir() / "stop_hook.log"
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s [stop_hook] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def parse_transcript(transcript: list, profile: dict) -> str:
    """Convert transcript list to extractable plain text."""
    user_name = profile.get("user_display_name", "User")
    assistant_name = profile.get("assistant_display_name", "Assistant")

    lines = []
    for turn in transcript:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")

        # content may be a string or a list (tool_use etc.)
        if isinstance(content, list):
            text_parts = [
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = " ".join(text_parts)

        if content and content.strip():
            # Rough token estimate: ~4 chars per token, limit 300 tokens/turn
            text = content.strip()
            if len(text) > 1200:
                text = text[:1200] + "..."
            prefix = user_name if role == "user" else assistant_name
            lines.append(f"{prefix}: {text}")

    return "\n".join(lines)


def should_extract(transcript_text: str) -> bool:
    """Determine if extraction is worthwhile (filter short/trivial conversations)."""
    if len(transcript_text) < 200:
        return False
    import re
    skip_keywords = {"hello", "test", "hi", "ping"}
    first_line = transcript_text.split("\n")[0].lower()
    # Word-boundary match to avoid false positives (e.g. "hi" inside "philosopher")
    words = set(re.findall(r'\b[a-z]+\b', first_line))
    if words & skip_keywords:
        return False
    return True


def _spawn_dream_background(agent_id: str):
    """Spawn Dream as a detached subprocess, non-blocking."""
    dream_script = Path(__file__).parent.parent.parent / "core" / "dream.py"
    try:
        subprocess.Popen(
            [sys.executable, str(dream_script), "--agent-id", agent_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        logger.info(f"[dream] spawned background Dream subprocess agent={agent_id}")
    except Exception as e:
        logger.warning(f"[dream] failed to spawn background Dream (non-fatal): {e}")


async def run(hook_input: dict):
    from memocore.core.extractor import extract_and_store

    agent_id = validate_agent_id(get_agent_id())
    profile = _get_profile(agent_id)
    transcript = hook_input.get("transcript", [])
    session_id = hook_input.get("session_id", "unknown")

    if not transcript:
        logger.info(f"session={session_id} transcript empty, skipping")
        return

    transcript_text = parse_transcript(transcript, profile)

    if not should_extract(transcript_text):
        logger.info(f"session={session_id} content too short or trivial, skipping")
        return

    logger.info(f"session={session_id} starting extraction, text_len={len(transcript_text)}")

    # Step 1: Extract and store
    result = await extract_and_store(
        conversation=transcript_text,
        agent_id=agent_id,
        source_description=f"conversation | session={session_id[:16]} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )

    if result["success"]:
        logger.info(
            f"session={session_id} extraction succeeded: "
            f"episode={result['episode_name']} "
            f"entities={result.get('entities_extracted', '?')}"
        )
    else:
        logger.error(f"session={session_id} extraction failed: {result.get('error')}")
        return  # Don't trigger Dream on extraction failure

    # Step 2: Dream consolidation (background subprocess, non-blocking)
    if should_run_dream(agent_id):
        logger.info(f"session={session_id} counter reached threshold, triggering background Dream...")
        _spawn_dream_background(agent_id)
    else:
        logger.info(f"session={session_id} Dream counter below threshold, skipping")


def main():
    _configure_logging()

    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        hook_input = json.loads(raw)
    except Exception as e:
        logger.error(f"failed to read hook input: {e}")
        return

    try:
        asyncio.run(run(hook_input))
    except Exception as e:
        logger.error(f"stop_hook main flow failed: {e}")


if __name__ == "__main__":
    main()
