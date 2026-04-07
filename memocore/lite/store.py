"""
SQLite-backed markdown page store with FTS5 full-text search.

Schema:
    pages(agent_id, page_path, content, updated_at, version) PRIMARY KEY (agent_id, page_path)
    pages_fts — FTS5 virtual table mirroring (agent_id, page_path, content)

Concurrency model:
- Multiple readers OK (SQLite WAL mode).
- Single writer per process. Cross-process writes are serialized by SQLite,
  but high-concurrency write patterns should split into per-agent files.

Why one DB per agent (recommended for 4000-agent deployment):
- No write contention between agents.
- Trivial sharding / per-agent backup.
- Aggregate queries via SQLite ATTACH DATABASE across files.

This file is intentionally small (one table, no migrations, no ORM) so it
stays auditable. If schema needs to evolve, add a migration() function and a
schema_version table — don't reach for SQLAlchemy.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS pages (
    agent_id   TEXT NOT NULL,
    page_path  TEXT NOT NULL,
    content    TEXT NOT NULL,
    updated_at REAL NOT NULL,
    version    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (agent_id, page_path)
);

CREATE INDEX IF NOT EXISTS idx_pages_agent_updated
    ON pages(agent_id, updated_at DESC);

-- trigram tokenizer (FTS5 built-in since SQLite 3.34) handles CJK correctly
-- by splitting text into 3-character n-grams. Default unicode61 tokenizer
-- treats Chinese as one token-per-whitespace-run, which makes recall on
-- mixed CN/EN queries fail almost completely.
CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    agent_id UNINDEXED,
    page_path,
    content,
    content='pages',
    content_rowid='rowid',
    tokenize='trigram'
);

CREATE TRIGGER IF NOT EXISTS pages_ai AFTER INSERT ON pages BEGIN
    INSERT INTO pages_fts(rowid, agent_id, page_path, content)
    VALUES (new.rowid, new.agent_id, new.page_path, new.content);
END;

CREATE TRIGGER IF NOT EXISTS pages_ad AFTER DELETE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, agent_id, page_path, content)
    VALUES ('delete', old.rowid, old.agent_id, old.page_path, old.content);
END;

CREATE TRIGGER IF NOT EXISTS pages_au AFTER UPDATE ON pages BEGIN
    INSERT INTO pages_fts(pages_fts, rowid, agent_id, page_path, content)
    VALUES ('delete', old.rowid, old.agent_id, old.page_path, old.content);
    INSERT INTO pages_fts(rowid, agent_id, page_path, content)
    VALUES (new.rowid, new.agent_id, new.page_path, new.content);
END;
"""


class MemoryStore:
    """File-like markdown page store backed by SQLite.

    Each instance is bound to one (db_path, agent_id) pair. To share a DB
    between multiple agents, instantiate multiple MemoryStore instances with
    different agent_id values pointing to the same db_path.
    """

    def __init__(self, db_path: str | Path, agent_id: str):
        self.db_path = str(db_path)
        self.agent_id = agent_id
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # ── file-like operations ────────────────────────────────────────

    def write_page(self, page_path: str, content: str) -> int:
        """Create or overwrite a page. Returns the new version number."""
        now = time.time()
        cur = self._conn.execute(
            "SELECT version FROM pages WHERE agent_id = ? AND page_path = ?",
            (self.agent_id, page_path),
        )
        row = cur.fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO pages(agent_id, page_path, content, updated_at, version) "
                "VALUES (?, ?, ?, ?, 1)",
                (self.agent_id, page_path, content, now),
            )
            return 1
        new_version = row[0] + 1
        self._conn.execute(
            "UPDATE pages SET content = ?, updated_at = ?, version = ? "
            "WHERE agent_id = ? AND page_path = ?",
            (content, now, new_version, self.agent_id, page_path),
        )
        return new_version

    def read_page(self, page_path: str) -> Optional[str]:
        """Return page content, or None if not found."""
        cur = self._conn.execute(
            "SELECT content FROM pages WHERE agent_id = ? AND page_path = ?",
            (self.agent_id, page_path),
        )
        row = cur.fetchone()
        return row[0] if row else None

    def delete_page(self, page_path: str) -> bool:
        """Delete a page. Returns True if a row was deleted."""
        cur = self._conn.execute(
            "DELETE FROM pages WHERE agent_id = ? AND page_path = ?",
            (self.agent_id, page_path),
        )
        return cur.rowcount > 0

    def list_pages(self, prefix: str = "") -> list[dict]:
        """List all pages for this agent, optionally filtered by path prefix.

        Returns list of {page_path, updated_at, version, size} dicts.
        """
        if prefix:
            cur = self._conn.execute(
                "SELECT page_path, updated_at, version, length(content) "
                "FROM pages WHERE agent_id = ? AND page_path LIKE ? "
                "ORDER BY page_path",
                (self.agent_id, prefix + "%"),
            )
        else:
            cur = self._conn.execute(
                "SELECT page_path, updated_at, version, length(content) "
                "FROM pages WHERE agent_id = ? ORDER BY page_path",
                (self.agent_id,),
            )
        return [
            {
                "page_path": r[0],
                "updated_at": r[1],
                "version": r[2],
                "size": r[3],
            }
            for r in cur.fetchall()
        ]

    def search_pages(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across this agent's pages.

        Uses SQLite FTS5 (trigram tokenizer) with bm25 ranking.
        Multi-word queries are joined with OR rather than AND/phrase, so a
        long query like "Karpathy LLM Wiki 是什么" returns pages matching
        ANY of the terms — better recall, ranked by bm25 relevance.

        Returns list of {page_path, snippet, rank} sorted by relevance.
        """
        # Split on whitespace, drop empty, wrap each token in double quotes
        # to handle special characters (FTS5 reserved chars like - and *).
        # Multi-token queries become "tok1" OR "tok2" OR "tok3".
        tokens = [t for t in query.split() if t.strip()]
        if not tokens:
            return []
        if len(tokens) == 1:
            fts_query = f'"{tokens[0]}"'
        else:
            fts_query = " OR ".join(f'"{t}"' for t in tokens)

        try:
            cur = self._conn.execute(
                """
                SELECT p.page_path,
                       snippet(pages_fts, 2, '[', ']', '...', 12) AS snippet,
                       bm25(pages_fts) AS rank
                FROM pages_fts
                JOIN pages p ON p.rowid = pages_fts.rowid
                WHERE pages_fts MATCH ? AND pages_fts.agent_id = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, self.agent_id, limit),
            )
            return [
                {"page_path": r[0], "snippet": r[1], "rank": r[2]}
                for r in cur.fetchall()
            ]
        except sqlite3.OperationalError:
            # Malformed FTS query (e.g. only punctuation) — degrade to empty.
            return []

    def recent_pages(
        self,
        prefix: str = "",
        exclude_prefix: Optional[str] = None,
        limit: int = 5,
    ) -> list[dict]:
        """List pages sorted by updated_at DESC (most recently updated first).

        Used by the "short-term context" injection path: find the sessions
        the agent most recently touched, not the ones whose path sorts high.

        prefix:         restrict to pages whose path starts with this.
        exclude_prefix: drop pages whose path starts with this (e.g. to
                        exclude the *current* session from context lookup).
        limit:          return at most N rows.
        """
        params: list = [self.agent_id]
        sql = (
            "SELECT page_path, updated_at, version, length(content) "
            "FROM pages WHERE agent_id = ?"
        )
        if prefix:
            sql += " AND page_path LIKE ?"
            params.append(prefix + "%")
        if exclude_prefix:
            sql += " AND page_path NOT LIKE ?"
            params.append(exclude_prefix + "%")
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)

        cur = self._conn.execute(sql, params)
        return [
            {
                "page_path": r[0],
                "updated_at": r[1],
                "version": r[2],
                "size": r[3],
            }
            for r in cur.fetchall()
        ]

    # ── housekeeping ────────────────────────────────────────────────

    def page_count(self) -> int:
        cur = self._conn.execute(
            "SELECT count(*) FROM pages WHERE agent_id = ?", (self.agent_id,)
        )
        return cur.fetchone()[0]

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
