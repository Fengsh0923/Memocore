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
        self._attached: dict[str, str] = {}  # alias -> path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)

    # ── cross-DB federation (star-topology read-only access) ────────
    #
    # We DO NOT use SQLite ATTACH DATABASE here, even though it looks like
    # the obvious solution. Reason: FTS5's virtual-table helper functions
    # (snippet, bm25) and the pages_fts MATCH clause reject a schema prefix,
    # so a query like `snippet(tianxia.pages_fts, ...)` raises "no such
    # column". SQLite simply does not allow FTS5 virtual tables to be
    # addressed through an attached database alias in the contexts we need.
    #
    # Workaround: each federated DB gets its own short-lived connection
    # opened read-only via URI. Queries run natively inside that connection
    # with full FTS5 support. The federation is just a path registry.

    def attach_readonly(self, alias: str, db_path: str | Path) -> bool:
        """Register another memocore.lite DB as a federated read-only peer.

        Returns True if the file exists and is registered, False otherwise.
        Missing files are tolerated (silent no-op) — expected during boot or
        before a peer shrimp has published its first snapshot.

        Note: this does NOT open the DB. A short-lived connection is opened
        per search_all() / read_page_in_scope() call. This keeps the main
        connection clean and avoids FTS5 attach-prefix limitations.
        """
        path = Path(db_path).expanduser()
        if not path.exists():
            return False
        safe_alias = "".join(c if c.isalnum() or c == "_" else "_" for c in alias)
        self._attached[safe_alias] = str(path)
        return True

    def list_attached(self) -> dict[str, str]:
        return dict(self._attached)

    def _federated_connect(self, db_path: str) -> Optional[sqlite3.Connection]:
        """Open a read-only connection to a federated peer DB. None on error."""
        try:
            # URI mode=ro prevents accidental writes even if we have a bug.
            # No `immutable` flag because snapshots can legitimately change
            # under us (nutstore sync writing new snapshot mid-read is rare
            # but not impossible — a failed query just returns empty).
            conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
            return conn
        except sqlite3.OperationalError:
            return None

    def search_all(self, query: str, limit_per_scope: int = 5) -> list[dict]:
        """Federated FTS5 search across main DB plus every registered peer.

        Each peer (alias) is searched in its own short-lived connection so
        FTS5 works without any ATTACH-prefix limitations. Each peer
        contributes at most `limit_per_scope` hits, so a chatty shrimp can't
        drown out a quiet one. Results are tagged with the source scope.

        Returns list of {scope, page_path, snippet, rank}.
        """
        tokens = [t for t in query.split() if t.strip()]
        if not tokens:
            return []
        fts_query = (
            f'"{tokens[0]}"'
            if len(tokens) == 1
            else " OR ".join(f'"{t}"' for t in tokens)
        )

        def query_one(conn: sqlite3.Connection, scope_name: str) -> list[dict]:
            try:
                cur = conn.execute(
                    """
                    SELECT p.page_path,
                           snippet(pages_fts, 2, '[', ']', '...', 12) AS snippet,
                           bm25(pages_fts) AS rank
                    FROM pages_fts
                    JOIN pages p ON p.rowid = pages_fts.rowid
                    WHERE pages_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit_per_scope),
                )
                return [
                    {
                        "scope": scope_name,
                        "page_path": r[0],
                        "snippet": r[1],
                        "rank": r[2],
                    }
                    for r in cur.fetchall()
                ]
            except sqlite3.OperationalError:
                return []

        all_results: list[dict] = []

        # Main DB first (uses the primary connection, no re-open)
        all_results.extend(query_one(self._conn, self.agent_id))

        # Each federated peer in its own short-lived connection
        for alias, path in self._attached.items():
            peer = self._federated_connect(path)
            if peer is None:
                continue
            try:
                all_results.extend(query_one(peer, alias))
            finally:
                peer.close()

        return all_results

    def read_page_in_scope(self, scope: str, page_path: str) -> Optional[str]:
        """Read a page from the main DB or any federated peer by scope name.

        For the main DB, pass the agent_id the store was opened with.
        For peers, pass the alias used in attach_readonly().
        """
        if scope == self.agent_id or scope == "main":
            return self.read_page(page_path)

        safe_alias = "".join(c if c.isalnum() or c == "_" else "_" for c in scope)
        path = self._attached.get(safe_alias)
        if path is None:
            return None

        peer = self._federated_connect(path)
        if peer is None:
            return None
        try:
            cur = peer.execute(
                "SELECT content FROM pages WHERE page_path = ?",
                (page_path,),
            )
            row = cur.fetchone()
            return row[0] if row else None
        finally:
            peer.close()

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

    def search_all_agents(self, query: str, limit: int = 20) -> list[dict]:
        """Full-text search across ALL agent_id values in this DB.

        The star-topology deployment keeps every Flying Shrimp's memory in
        a single DB, each shrimp under its own agent_id namespace. This
        method runs one FTS5 query over everything and tags each result
        with its source agent_id, so the caller can render "who said it"
        in the recall output.

        Differs from search_pages() in that it:
          - does NOT restrict results to self.agent_id
          - returns an extra 'agent_id' field per row

        Like search_pages, long queries are OR-joined at the token level so
        a user prompt can match pages via any of its words.
        """
        tokens = [t for t in query.split() if t.strip()]
        if not tokens:
            return []
        fts_query = (
            f'"{tokens[0]}"'
            if len(tokens) == 1
            else " OR ".join(f'"{t}"' for t in tokens)
        )

        try:
            cur = self._conn.execute(
                """
                SELECT p.agent_id,
                       p.page_path,
                       snippet(pages_fts, 2, '[', ']', '...', 12) AS snippet,
                       bm25(pages_fts) AS rank
                FROM pages_fts
                JOIN pages p ON p.rowid = pages_fts.rowid
                WHERE pages_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
            return [
                {
                    "agent_id": r[0],
                    "page_path": r[1],
                    "snippet": r[2],
                    "rank": r[3],
                }
                for r in cur.fetchall()
            ]
        except sqlite3.OperationalError:
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
