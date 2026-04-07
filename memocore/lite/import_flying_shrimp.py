"""
Bulk-import every Flying Shrimp's memory directory from the Nutstore backup
mirror into a single central DB, one agent_id per shrimp.

This is the "central reader" data path:
- Each shrimp's memory lives on its own host as plain markdown files.
- Each shrimp mirrors its memory directory to Nutstore automatically (via
  its own backup job — memocore does not manage that mirror).
- This script runs on the central shrimp (aoxia), walks the Nutstore mirror,
  and imports each shrimp's files into aoxia.db under the shrimp's agent_id.
- Subsequent runs are incremental: MemoryStore.write_page() is a no-op when
  content is unchanged.

After running this, aoxia.db contains every shrimp's markdown content under
its namespace, and search_all_agents() can surface hits from any shrimp in a
single FTS5 query.

Paths are configured in SHRIMP_SOURCES below. Add a new shrimp by appending
to that list.
"""

from __future__ import annotations

import sys
from pathlib import Path

from memocore.lite.import_files import import_directory


# ── Configuration ────────────────────────────────────────────────

NUTSTORE_BACKUP = Path(
    "~/Nutstore Files/FS_KM/05_Backup"
).expanduser()

CENTRAL_DB = Path("~/.memocore-lite/aoxia.db").expanduser()

# Each tuple: (agent_id, relative path under NUTSTORE_BACKUP)
#
# Some shrimps keep their latest memory at a date-suffixed subdirectory
# (e.g. Tianxia/20260408). When that is the case we let the path include
# a wildcard '*' and pick the lexicographically highest match, which
# corresponds to the most recent date.
SHRIMP_SOURCES: list[tuple[str, str]] = [
    ("aoxia",    str(Path("~/.claude/projects/-Users-shenfeng/memory").expanduser())),
    ("tianxia",  "Tianxia/*"),
    ("longxia",  "Longxia/*"),
    ("maixia",   "Maixia/*"),
    ("huxia",    "Huxia"),
    ("hexia",    "Hexia"),
    ("mingxia",  "Mingxia"),
]


def resolve_source(relative: str) -> Path | None:
    """Resolve a SHRIMP_SOURCES entry to an actual directory path.

    Absolute paths (used for aoxia's own memory) are returned as-is.
    Relative paths are joined with NUTSTORE_BACKUP; if they contain a
    wildcard, we pick the highest-sorting match (most recent date suffix).
    """
    path = Path(relative)
    if path.is_absolute():
        return path if path.exists() else None

    if "*" in relative:
        parent = NUTSTORE_BACKUP / relative.split("/*")[0]
        if not parent.exists():
            return None
        subdirs = sorted(
            (d for d in parent.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        return subdirs[0] if subdirs else None

    candidate = NUTSTORE_BACKUP / relative
    return candidate if candidate.exists() else None


def main():
    total = {"imported": 0, "updated": 0, "unchanged": 0, "skipped_size": 0, "scanned": 0}
    print(f"Central DB: {CENTRAL_DB}")
    print()

    for agent_id, rel in SHRIMP_SOURCES:
        source = resolve_source(rel)
        if source is None:
            print(f"  [{agent_id:10s}] SKIP — source not found: {rel}")
            continue

        try:
            stats = import_directory(source, CENTRAL_DB, agent_id)
        except Exception as e:
            print(f"  [{agent_id:10s}] ERROR: {e}")
            continue

        for k in total:
            total[k] += stats.get(k, 0)

        print(
            f"  [{agent_id:10s}] {stats['scanned']:>4} scanned, "
            f"{stats['imported']:>3} new, "
            f"{stats['updated']:>3} updated, "
            f"{stats['unchanged']:>3} unchanged  "
            f"← {source}"
        )

    print()
    print("TOTAL:")
    for k, v in total.items():
        print(f"  {k:15s} {v}")


if __name__ == "__main__":
    main()
