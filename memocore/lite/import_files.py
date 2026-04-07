"""
Bulk-import a directory of markdown files into a MemoryStore.

Usage:
    python -m memocore.lite.import_files \
        --source ~/.claude/projects/-Users-shenfeng/memory \
        --db ~/.memocore-lite/aoxia.db \
        --agent-id aoxia

Behavior:
- Walks the source directory recursively for *.md files.
- Uses the path *relative to source* as the page_path (preserves subdirs).
- Idempotent: re-running over the same source updates pages whose content
  changed (version bumps), leaves unchanged ones alone.
- Skips files larger than --max-size (default 1 MB) to avoid blob bloat.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from memocore.lite import MemoryStore


def import_directory(
    source: Path,
    db_path: Path,
    agent_id: str,
    max_size: int = 1_000_000,
) -> dict:
    if not source.exists():
        raise FileNotFoundError(f"source directory does not exist: {source}")

    store = MemoryStore(str(db_path), agent_id=agent_id)
    stats = {"scanned": 0, "imported": 0, "updated": 0, "unchanged": 0, "skipped_size": 0}

    try:
        for md_path in sorted(source.rglob("*.md")):
            stats["scanned"] += 1
            size = md_path.stat().st_size
            if size > max_size:
                stats["skipped_size"] += 1
                print(f"  SKIP (too big, {size} bytes): {md_path.relative_to(source)}")
                continue

            page_path = str(md_path.relative_to(source))
            content = md_path.read_text(encoding="utf-8", errors="replace")

            existing = store.read_page(page_path)
            if existing is None:
                store.write_page(page_path, content)
                stats["imported"] += 1
            elif existing != content:
                store.write_page(page_path, content)
                stats["updated"] += 1
            else:
                stats["unchanged"] += 1
    finally:
        store.close()

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Directory of *.md files")
    parser.add_argument("--db", type=Path, required=True, help="SQLite DB path")
    parser.add_argument("--agent-id", type=str, required=True, help="Agent namespace")
    parser.add_argument("--max-size", type=int, default=1_000_000, help="Skip files > N bytes")
    args = parser.parse_args()

    print(f"Importing from: {args.source}")
    print(f"Into DB:        {args.db}")
    print(f"Agent ID:       {args.agent_id}")
    print()

    stats = import_directory(args.source, args.db, args.agent_id, args.max_size)

    print()
    print("Done.")
    for k, v in stats.items():
        print(f"  {k:15s} {v}")


if __name__ == "__main__":
    main()
