"""
Memocore Lite — Bridge + Claude Code hook adapter.

Drop-in replacement for memocore.adapters.bridge.{bridge_read,bridge_write}
and memocore.adapters.claude_code.{prompt_hook,stop_hook,mid_session_hook}
that uses the SQLite KV store instead of Graphiti/Neo4j.

Why one adapter for everything:
- Read path is identical: stdin prompt → FTS5 search → system prompt injection.
  The only difference is the output envelope (plain text for IM bridge,
  hookSpecificOutput JSON for Claude Code).
- Write path is identical: stdin transcript → markdown page.
- Having one module means one code path to audit, one place to fix bugs,
  and argv-based mode selection keeps the shell hooks trivial.

Modes (selected by argv[1]):
    (default)    bridge read      — plain text recall
    write        bridge write     — append transcript to bridge_sessions/
    cc_prompt    Claude Code UserPromptSubmit — JSON-wrapped recall
    cc_mid       Claude Code mid-session sync — persists in-progress session
                 to SQLite on every user prompt, so a crash / disconnect
                 before Stop doesn't lose the conversation.
    cc_stop      Claude Code Stop — final session persist (same path as cc_mid)

Persistence model for Claude Code sessions:
    Claude Code already writes every turn to ~/.claude/projects/<cwd>/<sid>.jsonl
    synchronously. We treat that JSONL file as the source of truth and
    mirror it into memocore.lite on every UserPromptSubmit and Stop. One
    SQLite page per session, path = sessions/<date>/<sid>.md.
    Overwrites on every sync — JSONL is append-only so the latest dump is
    always the full history.
"""

from __future__ import annotations

import glob
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
    """Persist an IM bridge transcript as a per-session markdown page.

    Uses the same sessions/<date>/<sid>.md layout as the Claude Code cc_stop
    path, so session-oriented recall works uniformly regardless of whether
    the session came from the IM bridge or a local Claude Code run.

    Each call overwrites the page — the caller is expected to pass the
    full transcript so far, not an incremental delta.
    """
    transcript = data.get("transcript", [])
    session_id = data.get("session_id", "unknown")
    if not transcript:
        return 0

    # Normalize the transcript into the same shape as parse_session_turns
    turns = []
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
        turns.append({"role": role, "text": content, "ts": ""})

    if not turns:
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    page_path = f"sessions/{today}/{session_id[:16]}.md"
    markdown = format_session_markdown(turns, session_id)

    store = MemoryStore(str(get_db_path()), agent_id=get_agent_id())
    try:
        store.write_page(page_path, markdown)
    finally:
        store.close()
    return len(turns)


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
#
# cc_prompt injects TWO pieces of context:
#   1. FTS5-based recall of long-term memory (same as bridge_read)
#   2. A short-term context window: the tail of the most recently updated
#      OTHER sessions. This is what lets a new local Claude Code session
#      know what was just discussed in an IM bridge conversation (or vice
#      versa) — continuity across entry points without relying on FTS
#      keyword overlap.


def _tail_session_page(content: str, turn_limit: int = 4) -> str:
    """Extract the last `turn_limit` '## Turn' blocks from a session page.

    Session pages use markdown headers of the form '## Turn N — **role**',
    so we split on those and keep the tail. This is more stable than
    character-based truncation because it respects turn boundaries.
    """
    if not content:
        return ""
    marker = "\n## Turn "
    parts = content.split(marker)
    if len(parts) <= 1:
        # Unexpected shape — fall back to character tail
        return content[-1500:]
    # parts[0] is the header block; parts[1:] are turns
    tail = parts[-turn_limit:]
    return marker.lstrip("\n") + marker.join(tail)


def load_recent_session_context(
    current_session_id: str,
    session_limit: int = 2,
    turns_per_session: int = 4,
) -> str:
    """Return a markdown block summarizing the tail of recent OTHER sessions.

    Excludes the current session (which the model already has in context)
    and returns at most `session_limit` sessions' tails.
    """
    db_path = get_db_path()
    if not db_path.exists():
        return ""

    short_sid = (current_session_id or "")[:16]
    store = MemoryStore(str(db_path), agent_id=get_agent_id())
    try:
        # Exclude the current session by constructing its sessions/*/sid.md
        # prefix pattern. Cross-date is rare but possible — we fall back
        # to filtering in Python.
        recent = store.recent_pages(prefix="sessions/", limit=session_limit + 2)
        out_blocks: list[str] = []
        for p in recent:
            if short_sid and short_sid in p["page_path"]:
                continue  # skip current session
            content = store.read_page(p["page_path"]) or ""
            tail = _tail_session_page(content, turn_limit=turns_per_session)
            if not tail.strip():
                continue
            out_blocks.append(
                f"### from `{p['page_path']}`\n\n{tail}"
            )
            if len(out_blocks) >= session_limit:
                break
    finally:
        store.close()

    if not out_blocks:
        return ""

    return (
        "\n--- recent session context ---\n"
        "## Tail of recent other sessions (cross-entry-point continuity)\n\n"
        + "\n\n".join(out_blocks)
        + "\n\n--- end of session context ---"
    )


def main_cc_prompt():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}

    session_id = data.get("session_id", "")

    # Long-term memory recall (FTS5 over all pages)
    recall_text = bridge_read(data)

    # Short-term context — tails of the most recent other sessions
    session_context = load_recent_session_context(session_id)

    combined = "\n".join(x for x in (recall_text, session_context) if x.strip())
    if not combined.strip():
        sys.exit(0)

    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": combined,
        }
    }
    print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)


# ── Claude Code session JSONL sync ────────────────────────────────
#
# Claude Code persists every turn to a session file as JSONL. We mirror
# that file into memocore.lite on every UserPromptSubmit (cc_mid) and Stop
# (cc_stop), so a crash / disconnect before Stop fires still leaves an
# up-to-date copy in the shared memory layer.


def find_session_file(session_id: str, cwd: Optional[str] = None) -> Optional[Path]:
    """Locate the JSONL file for a given session_id under ~/.claude/projects/.

    Claude Code stores sessions at ~/.claude/projects/<encoded-cwd>/<sid>.jsonl
    where <encoded-cwd> replaces / with -. We try the cwd-derived path first
    (fast path), then fall back to a glob across all projects (slow path).
    """
    if not session_id or session_id == "unknown":
        return None

    projects = Path("~/.claude/projects").expanduser()
    if not projects.exists():
        return None

    # Fast path: derive the encoded cwd
    if cwd:
        encoded = cwd.replace("/", "-")
        candidate = projects / encoded / f"{session_id}.jsonl"
        if candidate.exists():
            return candidate

    # Slow path: glob (still fast in practice — <1000 projects)
    matches = list(projects.glob(f"*/{session_id}.jsonl"))
    return matches[0] if matches else None


def parse_session_turns(jsonl_path: Path) -> list[dict]:
    """Parse a Claude Code session JSONL into a flat list of {role, text, ts}.

    Filters out thinking blocks, tool uses, tool results, system entries,
    file-history-snapshots, and other internal bookkeeping. Keeps only the
    text that would appear in a human-readable transcript.
    """
    turns = []
    try:
        with jsonl_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                t = obj.get("type")
                if t not in ("user", "assistant"):
                    continue

                msg = obj.get("message") or {}
                role = msg.get("role", t)
                content = msg.get("content")
                ts = obj.get("timestamp", "")

                # Flatten content into a single text string
                text_parts: list[str] = []
                if isinstance(content, str):
                    text_parts.append(content)
                elif isinstance(content, list):
                    for c in content:
                        if not isinstance(c, dict):
                            continue
                        ctype = c.get("type")
                        if ctype == "text":
                            text = c.get("text") or ""
                            if text.strip():
                                text_parts.append(text)
                        # deliberately skip: thinking, tool_use, tool_result,
                        # image, document — these are not human transcript

                text = "\n".join(text_parts).strip()
                if not text:
                    continue

                turns.append({"role": role, "text": text, "ts": ts})
    except (FileNotFoundError, PermissionError):
        return []

    return turns


def format_session_markdown(turns: list[dict], session_id: str) -> str:
    """Render turns as a human-browsable markdown page."""
    lines = [
        f"# Session {session_id[:16]}",
        f"",
        f"*{len(turns)} turns, last updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "---",
        "",
    ]
    for i, t in enumerate(turns, 1):
        prefix = "**user**" if t["role"] == "user" else "**assistant**"
        ts = t.get("ts", "")
        ts_short = ts[:19].replace("T", " ") if ts else ""
        lines.append(f"## Turn {i} — {prefix} {ts_short}")
        lines.append("")
        # truncate very long turns to keep the page browsable
        text = t["text"]
        if len(text) > 4000:
            text = text[:4000] + "\n\n... [truncated]"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def sync_session(session_id: str, cwd: Optional[str] = None) -> int:
    """Mirror a Claude Code session JSONL into memocore.lite.

    Returns number of turns persisted, or 0 if session file not found
    or contains no user-visible content.
    """
    session_file = find_session_file(session_id, cwd=cwd)
    if not session_file:
        return 0

    turns = parse_session_turns(session_file)
    if not turns:
        return 0

    today = datetime.now().strftime("%Y-%m-%d")
    page_path = f"sessions/{today}/{session_id[:16]}.md"
    content = format_session_markdown(turns, session_id)

    store = MemoryStore(str(get_db_path()), agent_id=get_agent_id())
    try:
        store.write_page(page_path, content)
    finally:
        store.close()

    return len(turns)


def main_cc_mid():
    """UserPromptSubmit hook — sync current session JSONL to SQLite.

    This is the "incremental persistence" path. Called on every user prompt,
    so even if the session crashes before Stop fires, the SQLite copy is
    at most one turn behind.
    """
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}

    session_id = data.get("session_id", "unknown")
    cwd = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")

    try:
        sync_session(session_id, cwd=cwd)
    except Exception:
        # Never block the user prompt flow on a sync error.
        pass
    sys.exit(0)


def main_cc_stop():
    """Stop hook — final session sync. Prefer JSONL source-of-truth over
    the stdin transcript, falling back to bridge_write if we can't find
    the session file (degenerate case)."""
    raw = sys.stdin.read()
    try:
        data = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError:
        data = {}

    session_id = data.get("session_id", "unknown")
    cwd = data.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR")

    try:
        n = sync_session(session_id, cwd=cwd)
        if n == 0:
            # Session file not found — fall back to raw stdin transcript.
            bridge_write(data)
    except Exception:
        pass
    sys.exit(0)


# ── module entry points ───────────────────────────────────────────

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "read"
    if mode == "write":
        main_write()
    elif mode == "cc_prompt":
        main_cc_prompt()
    elif mode == "cc_mid":
        main_cc_mid()
    elif mode == "cc_stop":
        main_cc_stop()
    else:
        main_read()
