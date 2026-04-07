# memocore.lite

A **zero-dependency, SQLite-backed, file-like** memory layer for AI agents.
Inspired by Andrej Karpathy's
[LLM Knowledge Bases](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f)
gist: **the LLM writes and maintains the wiki; the human reads and asks
questions**.

`memocore.lite` is an alternative to the Graphiti + Neo4j path in the main
`memocore` package. It is smaller (~600 LOC), has no external dependencies
(stdlib `sqlite3` only), and is designed for deployments where you want
Karpathy's file-first philosophy but need enterprise-style governance
(auditable storage, cross-agent aggregation, heterogeneous hosts).

---

## Why lite instead of the full stack?

| | `memocore` (Graphiti + Neo4j) | `memocore.lite` |
|---|---|---|
| Storage | Graph DB + vector index | One SQLite file |
| External deps | Neo4j, Graphiti, embedder, rerank LLM | Python stdlib |
| Recall mechanism | Vector + graph traversal + LLM rerank | FTS5 trigram full-text search |
| LLM calls per write | Multiple (extraction + embedding + rerank) | Zero |
| Lines of code | ~3000+ | ~600 |
| Debuggability | Needs Neo4j Browser, embedding tools | `sqlite3 agent.db` at the CLI |
| Install size | ~500 MB container | ~5 MB Python module |
| Best at | Semantic similarity across rephrased concepts | Lexical recall over LLM-maintained markdown |
| Scales to | Cross-enterprise queries over millions of entities | Thousands of agents × thousands of pages each |

Pick lite when:
- You want a memory layer that a single engineer can fully understand
- Your agents read their own markdown pages directly (Karpathy style)
- You want raw auditability (every page is human-readable markdown)
- You run at "tens to thousands of agents" scale, not "millions"
- You want to eliminate LLM calls from the write path

Pick the full graph-backed path when:
- You need cross-entity semantic similarity across rephrased concepts
- You need to answer "which agents have discussed topic X" at scale where
  a federated FTS lookup is too slow (this is rare in practice)

---

## Quick start — single agent

```python
from memocore.lite import MemoryStore

store = MemoryStore("/path/to/agent.db", agent_id="my_agent")

# Write a page
store.write_page("memory.md", "# What I learned today\n\n...")

# Read it back
content = store.read_page("memory.md")

# List everything under a prefix
pages = store.list_pages(prefix="projects/")

# Full-text search (FTS5 trigram; handles CJK + Latin + mixed queries)
hits = store.search_pages("scaling laws in transformer training")
for h in hits:
    print(h["page_path"], h["snippet"])

store.close()
```

## Bulk-import a directory of markdown files

Already have a memory directory full of markdown? Mirror it into a DB:

```bash
python -m memocore.lite.import_files \
    --source /path/to/markdown/dir \
    --db /path/to/agent.db \
    --agent-id my_agent
```

Idempotent: re-running only updates pages whose content changed.

## Multi-agent star topology

For a "one central agent aggregates many peers" setup (central can be a
governance reader, a personal assistant, or an operations agent), create
a config file describing all sources and run `import_multi_agent`:

```json
// ~/.memocore-lite/sources.json
{
  "central_db": "~/.memocore-lite/central.db",
  "max_size": 1000000,
  "sources": [
    {"agent_id": "alice",  "source": "~/memories/alice"},
    {"agent_id": "bob",    "source": "~/memories/bob"},
    {"agent_id": "carol",  "source": "~/shared/carol/latest/*"}
  ]
}
```

```bash
python -m memocore.lite.import_multi_agent
# or pass --config /path/to/custom.json
```

Paths ending in `*` pick the lexicographically highest subdirectory,
which is useful for date-stamped snapshot directories (`YYYYMMDD`).

Then, from any agent holding the central DB, search across everyone:

```python
from memocore.lite import MemoryStore

store = MemoryStore("~/.memocore-lite/central.db", agent_id="central")
hits = store.search_all_agents("what do we know about topic X")
for h in hits:
    print(f"[{h['agent_id']}] {h['page_path']}: {h['snippet']}")
```

## Live session persistence (Claude Code)

`bridge_adapter` provides drop-in replacements for Claude Code hooks that
mirror the live session JSONL file into the SQLite store on every user
prompt. If the session crashes or disconnects before the Stop hook fires,
at most the last in-flight turn is lost.

Wire it into `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {"hooks": [
        {"type": "command",
         "command": "python3 -m memocore.lite.bridge_adapter cc_prompt"},
        {"type": "command",
         "command": "python3 -m memocore.lite.bridge_adapter cc_mid"}
      ]}
    ],
    "Stop": [
      {"hooks": [
        {"type": "command",
         "command": "python3 -m memocore.lite.bridge_adapter cc_stop"}
      ]}
    ]
  }
}
```

Modes:
- `cc_prompt` — search the store and inject hits as additional system
  prompt context (JSON envelope for Claude Code)
- `cc_mid` — mirror the current session JSONL into the store (incremental
  persistence, called on every user prompt)
- `cc_stop` — final session mirror at conversation end
- `write` — write the transcript of an IM bridge turn into `sessions/<date>/<sid>.md`
- (default) — plain-text recall for an IM bridge system prompt

## Environment variables

| Variable | Meaning | Default |
|---|---|---|
| `MEMOCORE_LITE_DB` | Full path to the SQLite file | `~/.memocore-lite/<agent_id>.db` |
| `MEMOCORE_AGENT_ID` | This agent's namespace | `default` |

---

## The module layout

| File | Purpose |
|---|---|
| `store.py` | `MemoryStore` class: CRUD + FTS5 search + cross-agent search + federated read-only attach |
| `bridge_adapter.py` | IM bridge read/write + Claude Code hook entry points (`cc_prompt`, `cc_mid`, `cc_stop`) |
| `import_files.py` | Bulk-import one markdown directory into a DB |
| `import_multi_agent.py` | Bulk-import many agents' directories into one central DB via a config file |
| `snapshot.py` | Consistent DB snapshot via the SQLite Online Backup API (for backup / sync to file services) |

## Design principles

1. **Files are source of truth, SQLite is an index.** The expected data
   flow is: human / agent writes markdown files → `import_*` mirrors
   them to SQLite → retrievals read from SQLite. Losing the SQLite DB
   is not a disaster — you can always rebuild from files.

2. **No LLM calls in the write path.** Entity extraction, embedding,
   and rerank are all absent. The LLM does its extracting at *read*
   time, by reading full markdown pages that were already compiled
   for it. This is the core Karpathy insight.

3. **Zero external dependencies.** No Neo4j, no embedder, no vector
   index. Python's `sqlite3` plus FTS5 (built into the SQLite shipped
   with modern Python) is enough at this scale.

4. **One SQLite file per deployment.** Star topology (one central DB
   holding many agent namespaces) is the default pattern. Federation
   across multiple DB files is supported for the alternative topology
   where each agent owns its own DB.

5. **Human-readable at every layer.** Pages are markdown. Everything
   you'd want to debug is visible via `sqlite3 agent.db ".schema"` and
   `.dump`.

## Schema

One table + one FTS5 virtual table:

```sql
CREATE TABLE pages (
    agent_id   TEXT NOT NULL,
    page_path  TEXT NOT NULL,
    content    TEXT NOT NULL,
    updated_at REAL NOT NULL,
    version    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (agent_id, page_path)
);

CREATE VIRTUAL TABLE pages_fts USING fts5(
    agent_id UNINDEXED,
    page_path,
    content,
    content='pages',
    content_rowid='rowid',
    tokenize='trigram'
);
```

Trigram tokenizer (built into SQLite 3.34+) handles CJK, Latin, and mixed
queries correctly. The default `unicode61` tokenizer does not split
Chinese word boundaries and gives near-zero recall on CN queries, so
don't use it here.

## FAQ

**Q: Why not Dropbox/iCloud/network share for the DB file?**
A: SQLite uses a `-wal` and `-shm` sidecar file during writes. File sync
services do not atomically sync all three together, which causes
corruption. Use `memocore.lite.snapshot` to produce a consistent single
`.db` snapshot and sync that instead.

**Q: Can I have multiple writers?**
A: Multiple processes on the same host can read and write one DB with
WAL mode on — that's how the hooks work today. For multiple machines,
give each its own DB and federate via `attach_readonly` + `search_all`,
or centralize on one host.

**Q: What about backup?**
A: Either `cp agent.db agent.db.bak` (sloppy but works when no writer is
active) or `python -m memocore.lite.snapshot --source agent.db --target
/backup/agent.db` (uses the SQLite Online Backup API, consistent even
with an active writer).

**Q: Deleting an agent?**
A: `sqlite3 central.db "DELETE FROM pages WHERE agent_id = 'deleted_agent'"`
or `rm deleted_agent.db` depending on topology.

**Q: Tested at what scale?**
A: Pilot: 207 pages across 6 agent namespaces, ~1 MB DB, sub-50 ms FTS
queries on commodity hardware. Designed to comfortably handle 4000
agents × ~50 pages each (about 200K pages, ~50 MB) on a single host.

---

## License

Same as the parent `memocore` package (MIT).
