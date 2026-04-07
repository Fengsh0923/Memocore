"""
Create a consistent, readable snapshot of a memocore.lite DB into another path.

Uses the SQLite Online Backup API (sqlite3 Connection.backup), which:
- Works even if the source DB is being actively written to
- Guarantees the target is a transactionally-consistent snapshot at the
  moment the backup finished (no WAL mid-state)
- Writes a single .db file (no .db-wal / .db-shm sidecar), so the target
  is safe to put on a file sync service like Nutstore.

Typical use:
    python -m memocore.lite.snapshot \
        --source ~/.memocore-lite/aoxia.db \
        --target "~/Nutstore Files/FS_KM/00_Agent_Sandbox/memocore_snapshots/aoxia.db"

Safety:
- Writes to <target>.tmp first, then atomically renames on success.
- If the source is missing or the backup fails, exits non-zero and leaves
  any existing target untouched.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from pathlib import Path


def snapshot(source: Path, target: Path) -> int:
    """Backup source DB to target path atomically. Returns number of pages copied."""
    if not source.exists():
        raise FileNotFoundError(f"source DB does not exist: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")

    # Clean up a stale .tmp from a previous aborted run
    if tmp.exists():
        tmp.unlink()

    src_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        tgt_conn = sqlite3.connect(str(tmp))
        try:
            # backup() copies pages in chunks; progress callback is optional
            src_conn.backup(tgt_conn, pages=0)  # pages=0 means all at once
            tgt_conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            # fetch page_count for reporting
            cur = tgt_conn.execute("PRAGMA page_count")
            page_count = cur.fetchone()[0]
        finally:
            tgt_conn.close()
    finally:
        src_conn.close()

    # Atomic rename — overwrites any previous snapshot
    os.replace(tmp, target)
    return page_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()

    source = args.source.expanduser()
    target = args.target.expanduser()

    try:
        pages = snapshot(source, target)
    except Exception as e:
        print(f"snapshot failed: {e}", file=sys.stderr)
        sys.exit(1)

    size = target.stat().st_size
    print(f"snapshot OK: {source} -> {target} ({pages} pages, {size} bytes)")


if __name__ == "__main__":
    main()
