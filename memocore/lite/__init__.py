"""
Memocore Lite — KV-store of markdown blobs with file-like API.

Replaces the heavy Graphiti+Neo4j+vector recall path with a single SQLite
table holding markdown pages, plus FTS5 full-text search. Designed for the
Karpathy LLM Wiki pattern: the LLM reads markdown pages directly, no entity
extraction or vector recall in the middle.

Why a separate submodule (vs. modifying memocore/core/):
- The original Graphiti path remains importable for legacy users.
- This module has zero dependencies on neo4j/graphiti/openai — only stdlib sqlite3.
- Can be lifted out into its own package later if Memocore proper is deprecated.

Public API:
    from memocore.lite import MemoryStore
    store = MemoryStore("/path/to/agent.db", agent_id="my_agent")
    store.write_page("memory.md", "# Hello\n...")
    content = store.read_page("memory.md")
    pages = store.list_pages(prefix="project_")
    hits = store.search_pages("any query text")
"""

from memocore.lite.store import MemoryStore

__all__ = ["MemoryStore"]
