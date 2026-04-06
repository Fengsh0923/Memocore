"""
Memocore Bridge Write Hook
Called by IM bridge after each conversation to persist memory.

Input (stdin JSON):
  {
    "session_id": "...",
    "prompt": "user message",
    "response": "AI response",
    "channel": "slack" | "teams" | ...
  }

Function:
  - Compose prompt+response into conversation text
  - Call MemoryExtractor to write to knowledge graph
  - Skip conversations shorter than 100 chars
"""

import asyncio
import json
import logging
import sys
from datetime import datetime

from memocore.core.config import get_agent_id, validate_agent_id
from memocore.agents.registry import get_profile

logger = logging.getLogger("memocore.bridge_write")


def _get_role_names(agent_id: str) -> tuple[str, str]:
    p = get_profile(agent_id)
    return p.get("user_display_name", "User"), p.get("assistant_display_name", "Assistant")


MIN_LENGTH = 100


async def run(data: dict):
    session_id = data.get("session_id", "unknown")
    prompt = data.get("prompt", "").strip()
    response = data.get("response", "").strip()
    channel = data.get("channel", "bridge")
    agent_id = validate_agent_id(get_agent_id())

    if not prompt and not response:
        logger.info(f"session={session_id[:16]} prompt and response both empty, skipping")
        return

    user_name, assistant_name = _get_role_names(agent_id)
    conversation = f"{user_name}: {prompt}\n{assistant_name}: {response}"

    if len(conversation) < MIN_LENGTH:
        logger.info(f"session={session_id[:16]} conversation too short ({len(conversation)} chars), skipping")
        return

    from memocore.core.extractor import extract_and_store

    source = f"bridge | channel={channel} | session={session_id[:16]} | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    logger.info(f"session={session_id[:16]} starting write, conversation_len={len(conversation)}")

    result = await extract_and_store(
        conversation=conversation,
        agent_id=agent_id,
        source_description=source,
    )

    if result["success"]:
        logger.info(
            f"session={session_id[:16]} write succeeded: "
            f"episode={result['episode_name']} "
            f"entities={result.get('entities_extracted', '?')}"
        )
    else:
        logger.error(f"session={session_id[:16]} write failed: {result.get('error')}")


def _configure_logging():
    from memocore.core.config import get_logs_dir
    log_file = get_logs_dir() / "bridge_write.log"
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s [bridge_write] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main():
    _configure_logging()

    stdin_data = sys.stdin.read()
    try:
        data = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        logger.error("invalid JSON input")
        sys.exit(0)

    asyncio.run(run(data))


if __name__ == "__main__":
    main()
