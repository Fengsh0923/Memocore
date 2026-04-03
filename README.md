# MemOS — Agent Universal Memory Layer

> Give your AI agents a persistent, self-healing memory.  
> Built on [Graphiti](https://github.com/getzep/graphiti) + Neo4j, designed for [Claude Code](https://claude.ai/code) hooks.

---

## The Problem

Every time you start a new conversation with an AI agent, it forgets everything.  
You re-explain your preferences. You repeat project context. You remind it of decisions made last week.

This is the **missing memory layer** problem — and it's the biggest blocker to agents becoming truly useful coworkers.

## What MemOS Does

MemOS is a universal memory layer that plugs into AI agents via hooks. It:

- **Automatically extracts** structured knowledge from conversations after each session
- **Automatically recalls** relevant memories at the start of each new conversation
- **Self-heals** its knowledge graph via periodic Dream consolidation (dedup, conflict resolution, stale pruning)
- **Stays actionable** — every memory carries metadata on *when to save* and *how to apply*

The result: your agent wakes up each session already knowing your preferences, project status, past decisions, and recurring patterns — without you saying a word.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code Session                   │
│                                                         │
│  UserPromptSubmit ──► prompt_hook.py                    │
│         │                (recall → additionalContext)   │
│         ▼                                               │
│   [ Conversation ]  ◄── memories injected automatically │
│         │                                               │
│  Stop ──► stop_hook.py ──► extract_and_store()          │
│                (write)       + Dream consolidation       │
└─────────────────────────────────────────────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Graphiti Engine     │
              │  temporal KG + RAG    │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Neo4j Graph DB      │
              │  nodes + edges        │
              │  + vector embeddings  │
              └───────────────────────┘
```

**Three core loops:**

| Loop | Trigger | What happens |
|------|---------|-------------|
| **Write** | Session ends (Stop hook) | Conversation → LLM extraction → structured entities → Neo4j |
| **Recall** | Each user message (UserPromptSubmit hook) | Query → vector search → LLM rerank → inject as `additionalContext` |
| **Dream** | Every ~5 sessions (async, post-write) | Scan graph → find duplicates/conflicts/stale nodes → prune |

## Key Features

### Typed Entity Schema
Seven entity types — easily customizable for your own agent context:

| Entity | What it captures |
|--------|-----------------|
| `FrankPreference` | Behavioral rules, aesthetic standards, communication norms |
| `ProjectStatus` | Active projects, current phase, next steps, key decisions |
| `Judgment` | Confirmed conclusions, technical choices, strategic decisions |
| `AgentConfig` | Agent team configuration, schedules, known issues |
| `TaskRecord` | Task execution history and lessons learned |
| `Incident` | Bugs, incidents, root causes, prevention checklist |
| `ExternalResource` | Service ports, API endpoints, file paths |

Every entity carries two meta-fields that make memories actionable:
- `when_to_save` — what context triggers saving this memory
- `how_to_apply` — how to use this memory when it's recalled

### Two-Stage Recall
```
User prompt
    │
    ▼
Stage 1: Graphiti vector + graph search  →  top 20 candidates
    │
    ▼
Stage 2: gpt-4o-mini relevance rerank    →  top 5 injected
```
The LLM rerank step cuts noise significantly vs. a pure similarity threshold.  
Stage 2 can be disabled (`use_rerank=False`) when latency matters more than precision.

### Dream Consolidation (4 phases)

The knowledge graph degrades without maintenance — duplicate nodes accumulate, facts contradict each other, stale orphans pile up. Dream runs automatically every ~5 sessions:

```
Phase 1  Orient      — scan graph, identify problem areas
Phase 2  Gather      — bundle candidates into task packages
Phase 3  Consolidate — gpt-4o-mini: merge / keep_latest / delete / skip
Phase 4  Prune       — execute Neo4j changes, write action log
```

Supports `--dry-run` for safe inspection without modifying the graph.

---

## Getting Started

### Prerequisites

- Python ≥ 3.10
- Neo4j running locally or in the cloud — [Neo4j Desktop](https://neo4j.com/download/)
- OpenAI API key
- Claude Code (for hook integration)

### Install

```bash
git clone https://github.com/Frankshen923/MemOS.git
cd MemOS

python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### Configure

```bash
cp .env.example .env
# Fill in your credentials
```

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
OPENAI_API_KEY=sk-...
DEFAULT_AGENT_ID=my_agent
```

### Verify the connection

```bash
PYTHONPATH=. python3 examples/aoxia_demo/test_connection.py
```

### Core API usage

```python
import asyncio
from memos.core.extractor import extract_and_store
from memos.core.retriever import MemoryRetriever

async def main():
    # Write a conversation into memory
    result = await extract_and_store(
        conversation="""
        User: we decided to use Graphiti for the memory layer.
        Agent: confirmed, storing that decision.
        """,
        agent_id="my_agent",
    )
    print(result)
    # → {'success': True, 'entities_extracted': 3, 'episode_name': '...'}

    # Recall relevant memories
    retriever = MemoryRetriever()
    context = await retriever.retrieve(
        query="memory layer technology choice",
        agent_id="my_agent",
        top_k=5,
        use_rerank=True,
    )
    print(context)
    # → Markdown text, ready to inject into a system prompt
    await retriever.close()

asyncio.run(main())
```

### Run Dream consolidation

```bash
# Dry run — inspect what would change without touching the graph
PYTHONPATH=. python3 -m memos.core.dream --agent-id my_agent --dry-run

# Live run
PYTHONPATH=. python3 -m memos.core.dream --agent-id my_agent
```

---

## Claude Code Hook Integration

Add to your `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/venv/bin/python3 /path/to/MemOS/memos/adapters/claude_code/prompt_hook.py",
            "timeout": 25
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/venv/bin/python3 /path/to/MemOS/memos/adapters/claude_code/stop_hook.py",
            "timeout": 60
          }
        ]
      }
    ]
  }
}
```

After this:
- Every new conversation **starts** with relevant memories injected as context
- Every conversation that ends **writes** extracted knowledge back to the graph

The hooks are non-blocking — if MemOS is slow or unavailable, the conversation continues normally.

---

## Project Structure

```
MemOS/
├── memos/
│   ├── core/
│   │   ├── extractor.py       # Conversation → Graphiti extraction
│   │   ├── retriever.py       # Two-stage recall (vector + LLM rerank)
│   │   └── dream.py           # 4-phase memory consolidation
│   ├── agents/
│   │   └── aoxia/
│   │       └── schema.py      # Entity type definitions (customize this)
│   └── adapters/
│       └── claude_code/
│           ├── stop_hook.py   # Write hook (session end)
│           └── prompt_hook.py # Recall hook (each message)
├── examples/
│   └── aoxia_demo/
│       └── test_connection.py
├── .env.example
└── pyproject.toml
```

---

## Adapting to Your Agent

The schema in `memos/agents/aoxia/schema.py` is built for a personal AI assistant context. The pattern is generic — to adapt it:

1. Copy `memos/agents/aoxia/` → `memos/agents/your_agent/`
2. Redefine entity types in `schema.py` to match your domain
3. Update `extraction_instructions` in `extractor.py` to guide the LLM extractor
4. Set `agent_id` in your hooks to your agent's name

Each `agent_id` is an isolated namespace in Neo4j — multiple agents can share the same database without memory leaking between them.

---

## Roadmap

- [ ] **M2** — Pre-built schemas for coding assistant, research assistant, project manager
- [ ] **M2** — Real-time streaming write (don't wait until session end)
- [ ] **M3** — Multi-agent shared memory with controlled cross-agent recall
- [ ] **M3** — Memory versioning and rollback
- [ ] **M4** — Web UI for graph inspection and manual memory editing

---

## Acknowledgements

MemOS builds directly on the work of others — these are not just citations, they're the actual foundation:

### [Graphiti](https://github.com/getzep/graphiti) — Zep AI
The temporal knowledge graph engine at the heart of MemOS. Graphiti handles entity extraction, embedding, graph construction, temporal reasoning, and hybrid search. MemOS wraps it with agent-specific schemas, hook adapters, and the Dream consolidation layer.  
**If you find MemOS useful, go give Graphiti a star — it's doing the heavy lifting.**

### [Claude Code](https://claude.ai/code) — Anthropic
The `UserPromptSubmit` and `Stop` hook system that makes seamless, zero-friction integration possible. The Dream consolidation mechanism was directly inspired by Claude Code's internal memory architecture — their async 4-phase consolidation pattern (Orient → Gather → Consolidate → Prune) maps cleanly to how MemOS maintains graph health.

### [Neo4j](https://neo4j.com)
The graph + vector database. The combination of graph traversal and vector similarity search is what enables hybrid recall — you can't replicate this with a pure vector store. Neo4j's free Desktop edition is all you need to get started.

---

## License

MIT

---

*Built by [Frank Shen](https://github.com/Frankshen923) · Part of the Flying Shrimp (飞虾队) AI infrastructure stack.*
