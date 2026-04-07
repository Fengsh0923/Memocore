"""
Memocore Lite — Bridge adapter.

Drop-in replacements for memocore.adapters.bridge.{bridge_read,bridge_write}
that use the SQLite KV store instead of Graphiti/Neo4j.

Why a separate adapter:
- The original bridge adapters depend on MemoryRetriever (Neo4j+Graphiti).
- This adapter has zero external dependencies and uses MemoryStore directly.
- Same stdin/stdout protocol as the originals, so the bridge hook shell
  scripts only need to swap the python -m target.

bridge_read protocol (stdin → stdout):
    in:  {"prompt": str, "session_id": str, "is_first_message": bool?}
    out: plain text to inject into system prompt (empty string = no recall)

bridge_write protocol (stdin → stdout):
    in:  {"transcript": [{"role": "user|assistant", "content": str}, ...],
          "session_id": str}
    out: ignored

Recall strategy (matches the Karpathy spirit — no LLM extraction):
    - First message: search_pages(prompt) returns top-K most-relevant pages.
    - Fast recall (subsequent): same.
    - The LLM gets snippets (FTS5 bm25-ranked) plus the option to ask for
      full pages by name. Empty result returns empty string.

Write strategy (deliberately minimal):
    - No LLM extraction (which is what poisoned the old graph).
    - Appends each session transcript to a single rolling page named
      `bridge_sessions/YYYY-MM-DD.md`. The LLM (or a periodic batch job)
      can later compile these raw conversations into wiki pages — that's
      the Karpathy ingest step, done out-of-band.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from memocore.lite import MemoryStore


# ── config helpers ────────────────────────────────────────────────


def get_db_path() -> Path:
    """Resolve the SQLite DB path. Override with MEMOCORE_LITE_DB env var."""
    override = os.environ.get("MEMOCORE_LITE_DB")
    if override:
        return Path(override).expanduser()
    return Path("~/.memocore-lite/aoxia.db").expanduser()


def get_agent_id() -> str:
    """Resolve agent_id. Override with MEMOCORE_AGENT_ID env var."""
    return os.environ.get("MEMOCORE_AGENT_ID", "aoxia")


def should_retrieve(prompt: str) -> bool:
    p = prompt.strip()
    if len(p) < 5:
        return False
    greetings = {"hi", "hello", "hey", "ok", "yes", "no", "thanks", "bye"}
    return p.lower() not in greetings


# ── bridge_read ───────────────────────────────────────────────────


def format_recall(hits: list[dict]) -> str:
    """Render search hits as a markdown block for system prompt injection."""
    if not hits:
        return ""
    lines = [
        "\n--- memocore-lite recall ---",
        "## Relevant historical memory (FTS5 trigram match)",
        "",
    ]
    for i, h in enumerate(hits, 1):
        lines.append(f"{i}. **{h['page_path']}** — {h['snippet']}")
    lines.append("")
    lines.append("--- end of recall ---")
    return "\n".join(lines)


def bridge_read(data: dict) -> str:
    prompt = data.get("prompt", "").strip()
    if not should_retrieve(prompt):
        return ""

    db_path = get_db_path()
    if not db_path.exists():
        return ""

    store = MemoryStore(str(db_path), agent_id=get_agent_id())
    try:
        hits = store.search_pages(prompt, limit=5)
        return format_recall(hits)
    finally:
        store.close()


def main_read():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}
    out = bridge_read(data)
    if out:
        sys.stdout.write(out)
    sys.exit(0)


# ── bridge_write ──────────────────────────────────────────────────


def bridge_write(data: dict) -> int:
    """Append session transcript to today's bridge_sessions page.

    Returns number of turns appended. Idempotent on session_id: re-appending
    the same session_id within the same day will append duplicate turns —
    callers should only invoke this once per session end.
    """
    transcript = data.get("transcript", [])
    session_id = data.get("session_id", "unknown")
    if not transcript:
        return 0

    db_path = get_db_path()
    store = MemoryStore(str(db_path), agent_id=get_agent_id())

    try:
        today = datetime.now().strftime("%Y-%m-%d")
        page_path = f"bridge_sessions/{today}.md"

        existing = store.read_page(page_path) or f"# Bridge sessions — {today}\n"

        block = [f"\n## session {session_id[:16]} @ {datetime.now().strftime('%H:%M:%S')}\n"]
        for turn in transcript:
            role = turn.get("role", "?")
            content = turn.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") for c in content
                    if isinstance(c, dict) and c.get("type") == "text"
                )
            content = content.strip()
            if not content:
                continue
            # truncate long messages to keep the page browsable
            if len(content) > 2000:
                content = content[:2000] + " ..."
            prefix = "**user**" if role == "user" else "**assistant**"
            block.append(f"{prefix}: {content}\n")

        store.write_page(page_path, existing + "\n".join(block))
        return len(transcript)
    finally:
        store.close()


def main_write():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}
    bridge_write(data)
    sys.exit(0)


# ── Claude Code UserPromptSubmit hook ─────────────────────────────
#
# Claude Code hooks expect a JSON envelope on stdout:
#   {"hookSpecificOutput": {"hookEventName": "UserPromptSubmit",
#                           "additionalContext": "..."}}
# Rather than a bare string (which is what the IM bridge consumes).
# Same search logic, different output shape.


def main_cc_prompt():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}

    recall_text = bridge_read(data)
    if not recall_text or not recall_text.strip():
        sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": recall_text,
        }
    }
    print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)


def main_cc_stop():
    # Claude Code Stop hook payload is the same shape as bridge_write input
    # (transcript + session_id), so reuse bridge_write directly.
    main_write()


# ── module entry points ───────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    if mode == "write":
        main_write()
    elif mode == "cc_prompt":
        main_cc_prompt()
    elif mode == "cc_stop":
        main_cc_stop()
    else:
        main_read()
