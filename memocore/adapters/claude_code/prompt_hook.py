"""
Memocore UserPromptSubmit Hook — auto-recall historical memory on each user message.

Trigger: every user message (UserPromptSubmit)
Strategy:
  - First message in session: retrieve_for_session_start (parallel queries, top10)
  - Subsequent messages: fast recall based on user prompt, top5 (no rerank, low latency)

Usage (register in settings.json):
  {
    "type": "command",
    "command": "python3 /path/to/memocore/adapters/claude_code/prompt_hook.py",
    "timeout": 25
  }
"""

import asyncio
import json
import logging
import sys

from memocore.core.config import get_agent_id, get_sessions_dir, get_logs_dir, validate_agent_id, make_safe_agent_key

logger = logging.getLogger("memocore.prompt_hook")


def is_first_message(session_id: str, agent_id: str) -> bool:
    """Check if this is the first user message in the session (isolated by agent_id)."""
    agent_key = make_safe_agent_key(agent_id)
    flag = get_sessions_dir() / f"session-{agent_key}-{session_id[:16]}.flag"
    if flag.exists():
        return False
    flag.touch()
    return True


def should_retrieve(prompt: str) -> bool:
    """Determine if a prompt-based recall is needed. Filter short/greeting messages."""
    p = prompt.strip()
    if len(p) < 5:
        return False
    greetings = {"hi", "hello", "hey", "ok", "yes", "no", "thanks", "bye"}
    if p.lower() in greetings:
        return False
    return True


async def run(hook_input: dict) -> str:
    """Main flow, returns additionalContext text."""
    prompt = hook_input.get("prompt", "")
    session_id = hook_input.get("session_id", "unknown")
    agent_id = validate_agent_id(get_agent_id())

    from memocore.core.retriever import MemoryRetriever

    retriever = MemoryRetriever()
    context_text = ""

    try:
        first_message = is_first_message(session_id, agent_id)

        if first_message:
            logger.info(f"session={session_id[:16]} first message, triggering full recall")
            context_text = await retriever.retrieve_for_session_start(
                agent_id=agent_id,
                top_k=10,
            )
            logger.info(
                f"session={session_id[:16]} full recall complete, "
                f"chars={len(context_text)}"
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
                f"session={session_id[:16]} fast recall complete, "
                f"chars={len(context_text)}"
            )
        else:
            logger.info(f"session={session_id[:16]} prompt too short or greeting, skip recall")

    except Exception as e:
        logger.error(f"session={session_id[:16]} recall failed: {e}")
        context_text = ""

    finally:
        await retriever.close()

    return context_text


def main():
    # Configure logging for hook process
    log_file = get_logs_dir() / "prompt_hook.log"
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s [prompt_hook] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    stdin_data = sys.stdin.read()
    try:
        hook_input = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        hook_input = {}

    session_id = hook_input.get("session_id", "unknown")

    try:
        context_text = asyncio.run(run(hook_input))
    except Exception as e:
        logger.error(f"session={session_id[:16]} hook main flow failed: {e}")
        sys.exit(0)

    if not context_text or not context_text.strip():
        sys.exit(0)

    from memocore.core.locale import t

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                t("ui.hook_memory_start")
                + context_text
                + t("ui.hook_memory_end")
            ),
        }
    }
    print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
