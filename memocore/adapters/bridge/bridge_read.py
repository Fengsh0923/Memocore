"""
Memocore Bridge Read Hook
Called by IM bridge on each incoming user message to recall historical memory for system prompt injection.

Input (stdin JSON):
  {
    "prompt": "user message",
    "session_id": "...",
    "is_first_message": true | false  (optional, default false)
  }

Output (stdout):
  Plain text to append to system prompt; empty string if no memory.
"""

import asyncio
import json
import logging
import sys

from memocore.core.config import get_agent_id, get_sessions_dir, validate_agent_id, make_safe_agent_key

logger = logging.getLogger("memocore.bridge_read")


def is_first_message(session_id: str, agent_id: str) -> bool:
    """Check if this is the first message in the session (isolated by agent_id)."""
    agent_key = make_safe_agent_key(agent_id)
    flag = get_sessions_dir() / f"bridge-{agent_key}-{session_id[:16]}.flag"
    if flag.exists():
        return False
    flag.touch()
    return True


def should_retrieve(prompt: str) -> bool:
    p = prompt.strip()
    if len(p) < 5:
        return False
    greetings = {"hi", "hello", "hey", "ok", "yes", "no", "thanks", "bye"}
    if p.lower() in greetings:
        return False
    return True


async def run(data: dict) -> str:
    prompt = data.get("prompt", "").strip()
    session_id = data.get("session_id", "unknown")
    force_first = data.get("is_first_message", False)
    agent_id = validate_agent_id(get_agent_id())

    from memocore.core.retriever import MemoryRetriever

    retriever = MemoryRetriever()
    context_text = ""

    try:
        first_msg = force_first or is_first_message(session_id, agent_id)

        if first_msg:
            logger.info(f"session={session_id[:16]} first message, triggering full recall")
            context_text = await retriever.retrieve_for_session_start(
                agent_id=agent_id,
                top_k=10,
            )
            logger.info(
                f"session={session_id[:16]} full recall complete, chars={len(context_text)}"
            )

        elif should_retrieve(prompt):
            logger.info(f"session={session_id[:16]} subsequent message, fast recall")
            context_text = await retriever.retrieve(
                query=prompt,
                agent_id=agent_id,
                top_k=5,
                use_rerank=False,
            )
            logger.info(
                f"session={session_id[:16]} fast recall complete, chars={len(context_text)}"
            )
        else:
            logger.info(f"session={session_id[:16]} prompt too short or greeting, skipping recall")

    except Exception as e:
        logger.error(f"session={session_id[:16]} recall failed: {e}")
        context_text = ""

    finally:
        await retriever.close()

    return context_text


def _configure_logging():
    from memocore.core.config import get_logs_dir
    log_file = get_logs_dir() / "bridge_read.log"
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s [bridge_read] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def main():
    _configure_logging()

    stdin_data = sys.stdin.read()
    try:
        data = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        logger.error("invalid JSON input")
        sys.stdout.write("")
        sys.exit(0)

    try:
        result = asyncio.run(run(data))
    except Exception as e:
        logger.error(f"bridge_read main flow failed: {e}")
        result = ""

    from memocore.core.locale import t

    # Output to stdout for daemon.mjs
    if result and result.strip():
        sys.stdout.write(
            t("ui.hook_memory_start")
            + result
            + t("ui.hook_memory_end")
        )
    else:
        sys.stdout.write("")

    sys.exit(0)


if __name__ == "__main__":
    main()
