"""
Bulk-import multiple agents' memory directories into a single central DB,
one agent_id per namespace. This is the "star topology" data path for
deployments where one central agent (e.g. an operations agent or a
governance aggregator) wants read access to many other agents' content.

Configuration is read from a JSON file (default:
~/.memocore-lite/sources.json) so the source paths and agent ids are never
hardcoded in the shipped code — this file is typically managed by the
deployment's provisioning layer, not checked into the module.

Config file format:

    {
      "central_db": "~/.memocore-lite/central.db",
      "sources": [
        {"agent_id": "alice",  "source": "~/memories/alice"},
        {"agent_id": "bob",    "source": "~/memories/bob"},
        {"agent_id": "carol",  "source": "~/shared/carol/latest/*"}
      ],
      "max_size": 1000000
    }

Paths in "source" may contain a trailing '*' to pick the most recently
updated subdirectory (useful when upstream agents write to date-stamped
snapshot directories like .../20260408). Otherwise they are used as-is.

Usage:

    python -m memocore.lite.import_multi_agent \
        [--config ~/.memocore-lite/sources.json]

Safety:
- Idempotent: re-running over unchanged source directories writes nothing.
- Never modifies source files.
- Each agent_id is fully isolated in the central DB via the existing
  (agent_id, page_path) primary key, so agents cannot clobber each other.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from memocore.lite.import_files import import_directory


DEFAULT_CONFIG_PATH = Path("~/.memocore-lite/sources.json").expanduser()


def resolve_source(raw: str) -> Path | None:
    """Resolve a config entry's source path to an actual directory.

    - Absolute or home-relative paths are returned as-is if they exist.
    - A path ending in '/*' picks the lexicographically highest subdirectory,
      which for date-stamped dirs (YYYYMMDD) is the most recent snapshot.
    - Missing paths return None (the caller reports and skips).
    """
    path = Path(raw).expanduser()

    # Glob tail: pick the latest subdirectory
    if raw.endswith("*") or raw.endswith("/*"):
        parent = Path(raw.rstrip("*").rstrip("/")).expanduser()
        if not parent.exists():
            return None
        subdirs = sorted(
            (d for d in parent.iterdir() if d.is_dir()),
            key=lambda d: d.name,
            reverse=True,
        )
        return subdirs[0] if subdirs else None

    return path if path.exists() else None


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"config not found: {config_path}\n\n"
            f"Create one with the following shape:\n"
            f"{{\n"
            f'  "central_db": "~/.memocore-lite/central.db",\n'
            f'  "sources": [\n'
            f'    {{"agent_id": "alice", "source": "~/memories/alice"}},\n'
            f'    {{"agent_id": "bob",   "source": "~/memories/bob"}}\n'
            f"  ]\n"
            f"}}"
        )
    return json.loads(config_path.read_text())


def run(config_path: Path) -> int:
    config = load_config(config_path)

    central_db = Path(config.get("central_db", "~/.memocore-lite/central.db")).expanduser()
    sources = config.get("sources", [])
    max_size = int(config.get("max_size", 1_000_000))

    if not sources:
        print("warn: no sources configured, nothing to import", file=sys.stderr)
        return 0

    print(f"Central DB: {central_db}")
    print(f"Config:     {config_path}")
    print()

    total = {"imported": 0, "updated": 0, "unchanged": 0, "skipped_size": 0, "scanned": 0}

    for entry in sources:
        agent_id = entry.get("agent_id")
        raw_source = entry.get("source")
        if not agent_id or not raw_source:
            print(f"  [invalid] skipping entry without agent_id/source: {entry}")
            continue

        source = resolve_source(raw_source)
        if source is None:
            print(f"  [{agent_id:12s}] SKIP — source not found: {raw_source}")
            continue

        try:
            stats = import_directory(source, central_db, agent_id, max_size=max_size)
        except Exception as e:
            print(f"  [{agent_id:12s}] ERROR: {e}")
            continue

        for k in total:
            total[k] += stats.get(k, 0)

        print(
            f"  [{agent_id:12s}] {stats['scanned']:>4} scanned, "
            f"{stats['imported']:>3} new, "
            f"{stats['updated']:>3} updated, "
            f"{stats['unchanged']:>3} unchanged  "
            f"← {source}"
        )

    print()
    print("TOTAL:")
    for k, v in total.items():
        print(f"  {k:15s} {v}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to sources config JSON (default: {DEFAULT_CONFIG_PATH})",
    )
    args = parser.parse_args()

    try:
        sys.exit(run(args.config.expanduser()))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
