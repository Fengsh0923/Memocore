"""
Memocore Memory Retriever (v3: CompiledPage + multi-scope)
Recall the most relevant historical memories based on current context.

Retrieval strategy:
  Stage 1: Graphiti vector + graph hybrid search, top_k=20 (coarse)
  Stage 2: LLM rerank on 20 results, pick top_5 (fine)

Usage:
    from memocore.core.retriever import MemoryRetriever
    retriever = MemoryRetriever()
    context = await retriever.retrieve("project requirements", agent_id="my_agent")
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from memocore.core.llm_adapter import rerank as llm_rerank, chat_complete
from memocore.core.config import get_neo4j_config
from memocore.core.graphiti_factory import build_graphiti
from memocore.agents.registry import get_profile
from memocore.core.locale import t

logger = logging.getLogger("memocore.retriever")


async def llm_rerank_titles(query: str, pages: list[dict], top_k: int) -> list[dict]:
    """Use LLM to select the most query-relevant pages from the CompiledPage title list"""
    import json as _json

    index = "\n".join(
        f"{i}. {p['title']}" for i, p in enumerate(pages)
    )
    prompt = t("retriever.rerank_titles_prompt", top_k=top_k, query=query, index=index)

    content = await chat_complete(prompt=prompt, max_tokens=100, temperature=0.0, json_mode=True)
    from memocore.core.llm_adapter import parse_llm_json
    indices = parse_llm_json(content)
    if not isinstance(indices, list):
        logger.warning(f"rerank returned non-list: {type(indices)}, falling back")
        return pages[:top_k]
    return [pages[i] for i in indices if 0 <= i < len(pages)]


def _filter_by_confidence(results: list, min_confidence: float = 0.2) -> list:
    """
    Filter out nodes with too-low confidence (memocore_confidence < min_confidence or status='stale').
    Graphiti edge objects typically do not carry these attributes directly; attempt to read them; if unreadable, retain by default.
    """
    filtered = []
    for r in results:
        confidence = getattr(r, 'memocore_confidence', None)
        status = getattr(r, 'memocore_status', None)
        # If node marked as stale with very low confidence, skip
        if status == 'stale' and confidence is not None and confidence < min_confidence:
            continue
        filtered.append(r)
    return filtered if filtered else results  # If all filtered out, return original list (fallback)


def _get_profile(agent_id: str) -> dict:
    return get_profile(agent_id)


def _scope_label(group_id: str) -> str:
    """Return localized scope label based on group_id prefix."""
    if group_id.startswith("team:"):
        return t("ui.scope_team")
    elif group_id.startswith("org:"):
        return t("ui.scope_org")
    return t("ui.scope_personal")


class MemoryRetriever:
    """
    Memory Retriever v3

    Recall priority:
    1. CompiledPage (structured knowledge compiled by Dream) -> direct Neo4j query, return on hit
    2. Graphiti semantic + graph traversal hybrid search -> fragment facts fallback path

    CompiledPage is an implementation of the Karpathy LLM Wiki concept:
    knowledge compiled once (Dream Phase 7), recall reads already-compiled pages directly,
    rather than re-deriving from fragment facts each time.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        top_k_stage1: int = 20,
        top_k_final: int = 5,
    ):
        cfg = get_neo4j_config()
        self.neo4j_uri = neo4j_uri or cfg["uri"]
        self.neo4j_user = neo4j_user or cfg["user"]
        self.neo4j_password = neo4j_password or cfg["password"]
        self.top_k_stage1 = top_k_stage1
        self.top_k_final = top_k_final
        self._graphiti: Optional[Graphiti] = None
        self._driver = None
        self._init_lock = asyncio.Lock()

    async def _get_neo4j_driver(self):
        """Get Neo4j driver (for direct CompiledPage queries, bypassing Graphiti)"""
        async with self._init_lock:
            if self._driver is None:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
                )
        return self._driver

    async def _recall_compiled_pages(
        self, query: str, group_ids: list[str], top_k: int = 5
    ) -> Optional[str]:
        """
        Retrieve relevant pages from CompiledPage nodes.
        Uses Neo4j full-text index or title fuzzy matching.
        Returns compiled knowledge text, or None (falls back to Graphiti when no CompiledPage exists).
        """
        driver = await self._get_neo4j_driver()

        all_pages = []
        async with driver.session() as session:
            for gid in group_ids:
                # Query all CompiledPages for this group
                q = """
                MATCH (p:CompiledPage {group_id: $gid})
                WHERE p.page_type = 'entity' AND p.stale = false
                RETURN p.title AS title, p.content AS content,
                       p.confidence AS confidence, p.source_count AS source_count,
                       p.compiled_at AS compiled_at
                ORDER BY p.source_count DESC
                """
                r = await session.run(q, gid=gid)
                pages = await r.data()
                for p in pages:
                    p["_scope_label"] = _scope_label(gid)
                all_pages.extend(pages)

        if not all_pages:
            return None  # No CompiledPage, falling back

        # Build index (title list), let LLM select relevant pages
        # Return all when pages <= top_k, no LLM selection needed
        if len(all_pages) <= top_k:
            relevant_pages = all_pages
        else:
            # Use LLM to select the most query-relevant pages from the index
            try:
                selection = await llm_rerank_titles(query, all_pages, top_k)
                relevant_pages = selection
            except Exception:
                # On LLM selection failure, take highest source_count
                relevant_pages = all_pages[:top_k]

        if not relevant_pages:
            return None

        # Format output
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        lines = [t("ui.compiled_memory_header", now=now)]
        for p in relevant_pages:
            scope = p.get("_scope_label", "")
            prefix = f"[{scope}] " if scope else ""
            lines.append(f"### {prefix}{p['title']}\n")
            lines.append(p.get("content", "") + "\n")
        return "\n".join(lines)

    async def _get_graphiti(self):
        async with self._init_lock:
            if self._graphiti is None:
                self._graphiti = await build_graphiti(
                    uri=self.neo4j_uri,
                    user=self.neo4j_user,
                    password=self.neo4j_password,
                )
        return self._graphiti

    async def _rerank_with_llm(
        self,
        query: str,
        candidates: list,
        top_k: int,
    ) -> list:
        """Stage 2: Fine-rank candidates via LLM adapter"""
        return await llm_rerank(query=query, candidates=candidates, top_k=top_k)

    async def _search_scoped(
        self,
        graphiti,
        query: str,
        group_ids: list[str],
        num_results: int,
        scope_label: str,
    ) -> list:
        """Execute search for a single scope and tag results with the _scope_label attribute"""
        try:
            results = await graphiti.search(
                query=query,
                group_ids=group_ids,
                num_results=num_results,
            )
            for r in results:
                r._scope_label = scope_label
            return results
        except Exception as e:
            logger.warning(f"Search failed for scope {scope_label}: {e}")
            return []

    async def retrieve(
        self,
        query: str,
        agent_id: str = "default",
        top_k: int = 5,
        use_rerank: bool = True,
        as_markdown: bool = True,
        team_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Recall historical memories most relevant to query, supports multi-scope merged retrieval

        Args:
            query: Current conversation context or keywords
            agent_id: Personal namespace
            top_k: Final number of results to return, default 5
            use_rerank: Enable LLM secondary fine-ranking
            as_markdown: True returns Markdown, False returns raw results
            team_id: Team ID, when provided also retrieves team memories
            tenant_id: Organization ID, when provided also retrieves organization knowledge

        Returns:
            Formatted memory text, ready for direct use in system prompt
        """
        import asyncio as _asyncio

        if not query.strip():
            return ""

        # ── Try CompiledPage first (compiled structured knowledge) ──
        group_ids = [agent_id]
        if team_id:
            group_ids.append(f"team:{team_id}")
        if tenant_id:
            group_ids.append(f"org:{tenant_id}")

        try:
            compiled = await self._recall_compiled_pages(query, group_ids, top_k)
            if compiled:
                return compiled  # Hit compiled knowledge, return directly
        except Exception as e:
            logger.warning(f"CompiledPage query failed, degrading to Graphiti: {e}")

        # ── Fallback: Graphiti fragment facts search ──
        try:
            graphiti = await self._get_graphiti()
            stage1_k = self.top_k_stage1 if use_rerank else top_k

            # Build multi-scope parallel query tasks
            search_tasks = [
                self._search_scoped(graphiti, query, [agent_id], stage1_k, t("ui.scope_personal"))
            ]
            if team_id:
                search_tasks.append(
                    self._search_scoped(graphiti, query, [f"team:{team_id}"], stage1_k // 2, t("ui.scope_team"))
                )
            if tenant_id:
                search_tasks.append(
                    self._search_scoped(graphiti, query, [f"org:{tenant_id}"], stage1_k // 2, t("ui.scope_org"))
                )

            scoped_results = await _asyncio.gather(*search_tasks)

            # Merge, deduplicate (by uuid)
            seen = set()
            all_results = []
            for batch in scoped_results:
                for r in batch:
                    uid = getattr(r, 'uuid', id(r))
                    if uid not in seen:
                        seen.add(uid)
                        all_results.append(r)

            if not all_results:
                return ""

            # Filter stale nodes with too-low confidence
            all_results = _filter_by_confidence(all_results)

            # Stage 2: LLM fine-ranking
            if use_rerank and len(all_results) > top_k:
                all_results = await self._rerank_with_llm(query, all_results, top_k)

            if not as_markdown:
                return str(all_results)

            return self._format_as_context(all_results, agent_id)

        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return ""

    async def retrieve_for_session_start(
        self,
        agent_id: str = "default",
        top_k: int = 15,
        team_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        Full relevant memory recall at session start, supports multiple scopes

        Args:
            agent_id: Personal namespace
            top_k: Total number of results to recall
            team_id: When provided, also recalls team memories
            tenant_id: When provided, also recalls organization knowledge
        """
        import asyncio as _asyncio

        # ── Try CompiledPage first (overview page + all entity pages) ──
        group_ids = [agent_id]
        if team_id:
            group_ids.append(f"team:{team_id}")
        if tenant_id:
            group_ids.append(f"org:{tenant_id}")

        try:
            driver = await self._get_neo4j_driver()
            pages_content = []
            async with driver.session() as session:
                for gid in group_ids:
                    # overview page
                    ov_q = """
                    MATCH (p:CompiledPage {group_id: $gid, title: '__overview__'})
                    RETURN p.content AS content
                    """
                    r = await session.run(ov_q, gid=gid)
                    rec = await r.single()
                    if rec and rec["content"]:
                        scope = _scope_label(gid)
                        pages_content.append(f"### [{scope}] {t('ui.overview_label')}\n\n{rec['content']}")

                    # All entity pages (top_k items)
                    ep_q = """
                    MATCH (p:CompiledPage {group_id: $gid})
                    WHERE p.page_type = 'entity' AND p.stale = false
                    RETURN p.title AS title, p.content AS content
                    ORDER BY p.source_count DESC
                    LIMIT $limit
                    """
                    r = await session.run(ep_q, gid=gid, limit=top_k // max(1, len(group_ids)))
                    entity_pages = await r.data()
                    for ep in entity_pages:
                        scope = _scope_label(gid)
                        pages_content.append(f"### [{scope}] {ep['title']}\n\n{ep.get('content', '')}")

            if pages_content:
                now = datetime.now().strftime('%Y-%m-%d %H:%M')
                return t("ui.session_start_header", now=now) + "\n\n".join(pages_content)
        except Exception as e:
            logger.warning(f"session_start CompiledPage query failed, degrading to Graphiti: {e}")

        # ── Fallback: Graphiti fragment facts search ──
        try:
            profile = _get_profile(agent_id)
            queries = profile.get("session_start_queries", ["recent decisions", "active projects"])

            graphiti = await self._get_graphiti()
            per_query_k = max(1, top_k // len(queries))

            tasks = []
            for q in queries:
                tasks.append(self._search_scoped(graphiti, q, [agent_id], per_query_k, t("ui.scope_personal")))
                if team_id:
                    tasks.append(self._search_scoped(graphiti, q, [f"team:{team_id}"], per_query_k // 2, t("ui.scope_team")))
                if tenant_id:
                    tasks.append(self._search_scoped(graphiti, q, [f"org:{tenant_id}"], per_query_k // 2, t("ui.scope_org")))

            all_batches = await _asyncio.gather(*tasks)

            # Merge and deduplicate
            seen = set()
            unique_results = []
            for batch in all_batches:
                for r in batch:
                    uid = getattr(r, 'uuid', id(r))
                    if uid not in seen:
                        seen.add(uid)
                        unique_results.append(r)

            if not unique_results:
                return ""

            return self._format_as_context(unique_results, agent_id, is_session_start=True)
        except Exception as e:
            logger.warning(f"session_start Graphiti fallback failed: {e}")
            return ""

    def _format_as_context(
        self,
        results: list,
        agent_id: str,
        is_session_start: bool = False,
    ) -> str:
        """Format recall results as Markdown usable in system prompt, annotate source for multi-scope results"""

        profile = _get_profile(agent_id)
        assistant_name = profile.get("assistant_display_name", "Memocore")
        if is_session_start:
            header = t("ui.recall_header_session", name=assistant_name)
        else:
            header = t("ui.recall_header_normal")
        lines = [header]
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        lines.append(t("ui.recall_source", agent_id=agent_id, now=now))

        for i, r in enumerate(results, 1):
            fact = getattr(r, 'fact', None)
            scope_label = getattr(r, '_scope_label', None)
            prefix = f"[{scope_label}] " if scope_label else ""

            if fact:
                lines.append(f"{i}. {prefix}{fact}")
            else:
                lines.append(f"{i}. {prefix}{str(r)[:200]}")

        return "\n".join(lines)

    async def close(self):
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None
        if self._driver:
            await self._driver.close()
            self._driver = None


# ─── Convenience functions ─────────────────────────────────────────────────────────────────

async def retrieve(
    query: str,
    agent_id: str = "default",
    top_k: int = 10,
) -> str:
    """Single retrieval convenience function, suitable for hooks."""
    retriever = MemoryRetriever()
    try:
        return await retriever.retrieve(query=query, agent_id=agent_id, top_k=top_k)
    finally:
        await retriever.close()


async def retrieve_for_session_start(agent_id: str = "default") -> str:
    """Session start retrieval — recall full background memory."""
    retriever = MemoryRetriever()
    try:
        return await retriever.retrieve_for_session_start(agent_id=agent_id)
    finally:
        await retriever.close()


if __name__ == "__main__":
    import asyncio

    async def run():
        print("=== Memory Retrieval Test ===\n")

        result = await retrieve("project requirements", agent_id="default")
        print("[Targeted recall] project requirements:")
        print(result or "(no results)")
        print()

        result2 = await retrieve_for_session_start(agent_id="default")
        print("[Session start] full background recall:")
        print(result2 or "(no results)")

    asyncio.run(run())
