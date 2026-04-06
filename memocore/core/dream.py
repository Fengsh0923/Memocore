"""
Memocore Dream — memory consolidation

Trigger: runs asynchronously after stop hook completion.
Phases:
  Phase 1 Orient      — scan graph, find candidate problem areas (duplicates, conflicts, stale)
  Phase 2 Gather      — aggregate related edges for candidate nodes
  Phase 3 Consolidate — LLM decides: merge / update / keep / delete
  Phase 4 Prune       — execute graph cleanup, write operation log
  Phase 5 Confidence  — update confidence scores
  Phase 6 Stale       — mark stale nodes
  Phase 7 Compile     — compile entity pages (Karpathy LLM Wiki)

Usage:
    from memocore.core.dream import run_dream
    await run_dream(agent_id="my_agent")

    # Or run as standalone script (for cron / stop hook):
    python -m memocore.core.dream --agent-id my_agent
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from memocore.core.llm_adapter import chat_complete, parse_llm_json
from memocore.core.config import get_dream_ttl_days, get_logs_dir, get_state_dir, cleanup_old_session_flags, get_neo4j_config, validate_agent_id, make_safe_agent_key
from memocore.core.locale import t

logger = logging.getLogger("memocore.dream")


def _configure_dream_logging():
    """Configure file logging for dream. Called from CLI entrypoint only."""
    log_file = get_logs_dir() / "dream.log"
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s [dream] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class DreamReport:
    """Dream run report."""
    agent_id: str
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    finished_at: Optional[datetime] = None

    # Phase 1 Orient
    total_nodes: int = 0
    total_edges: int = 0
    duplicate_groups: int = 0
    conflict_pairs: int = 0
    stale_nodes: int = 0

    # Phase 3 Consolidate
    merged: int = 0
    updated: int = 0
    pruned: int = 0
    kept: int = 0

    # TTL + confidence
    expired: int = 0
    confidence_lowered: int = 0

    # Phase 7 Compile
    compiled_pages: int = 0
    skipped_pages: int = 0

    # Phase 8 Lint
    lint_contradictions: int = 0
    lint_orphans: int = 0
    lint_missing: int = 0
    lint_stale_pages: int = 0
    lint_report_path: Optional[str] = None

    status: str = "running"
    error: Optional[str] = None

    def summary(self) -> str:
        elapsed = ""
        if self.finished_at:
            secs = (self.finished_at - self.started_at).total_seconds()
            elapsed = f" | {secs:.1f}s"
        return (
            f"[Dream {self.agent_id}] {self.status}{elapsed} | "
            f"nodes={self.total_nodes} edges={self.total_edges} | "
            f"duplicates={self.duplicate_groups} conflicts={self.conflict_pairs} "
            f"stale={self.stale_nodes} expired={self.expired} decayed={self.confidence_lowered} | "
            f"merged={self.merged} updated={self.updated} pruned={self.pruned} kept={self.kept} | "
            f"compiled={self.compiled_pages} lint_conflicts={self.lint_contradictions} "
            f"lint_orphans={self.lint_orphans} lint_missing={self.lint_missing}"
        )


# ─── Dream core class ────────────────────────────────────────────────────────

class DreamConsolidator:
    """
    Memocore Dream Consolidator
    Operates directly on Neo4j (via graphiti), independent of LLM add_episode path.
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        stale_days: int = 30,
        max_nodes_per_run: int = 200,
    ):
        cfg = get_neo4j_config()
        self.neo4j_uri = neo4j_uri or cfg["uri"]
        self.neo4j_user = neo4j_user or cfg["user"]
        self.neo4j_password = neo4j_password or cfg["password"]
        self.stale_days = stale_days
        self.ttl_days = get_dream_ttl_days()
        self.max_nodes_per_run = max_nodes_per_run
        self._driver = None

    async def _get_driver(self):
        """Get Neo4j driver (lazy init)."""
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )
            except ImportError:
                raise RuntimeError(
                    "Neo4j driver required: pip install neo4j"
                )
        return self._driver

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ── Phase 1: Orient ────────────────────────────────────────────────────────

    async def phase1_orient(self, agent_id: str, report: DreamReport) -> dict:
        """
        Scan the graph and return three categories of candidate issues:
        - duplicate_groups: groups of nodes with similar names (may need merging)
        - conflict_edges: edges with same subject/predicate but contradictory conclusions
        - stale_nodes: nodes older than stale_days with reference_count=0
        """
        logger.info(f"[orient] agent={agent_id} scanning graph...")
        driver = await self._get_driver()

        result = {
            "duplicate_groups": [],
            "conflict_edges": [],
            "stale_nodes": [],
        }

        async with driver.session() as session:
            # 1a. count totals
            count_q = """
            MATCH (n {group_id: $gid}) RETURN count(n) as cnt
            """
            r = await session.run(count_q, gid=agent_id)
            record = await r.single()
            report.total_nodes = record["cnt"] if record else 0

            edge_count_q = """
            MATCH ()-[e {group_id: $gid}]->() RETURN count(e) as cnt
            """
            r = await session.run(edge_count_q, gid=agent_id)
            record = await r.single()
            report.total_edges = record["cnt"] if record else 0

            logger.info(
                f"[orient] graph size: nodes={report.total_nodes} edges={report.total_edges}"
            )

            # 1b. find duplicate nodes (same name, different uuid, up to 50 groups)
            dup_q = """
            MATCH (n {group_id: $gid})
            WITH n.name AS name, collect(n.uuid) AS uuids, count(*) AS cnt
            WHERE cnt > 1 AND name IS NOT NULL
            RETURN name, uuids, cnt
            ORDER BY cnt DESC
            LIMIT 50
            """
            r = await session.run(dup_q, gid=agent_id)
            async for rec in r:
                result["duplicate_groups"].append({
                    "name": rec["name"],
                    "uuids": rec["uuids"],
                    "count": rec["cnt"],
                })

            report.duplicate_groups = len(result["duplicate_groups"])
            logger.info(f"[orient] duplicate node groups: {report.duplicate_groups}")

            # 1c. find conflicting edges (same source+target but contradictory fact keywords, up to 30 pairs)
            # strategy: find cases where the same pair of nodes has multiple RELATES_TO edges
            conflict_q = """
            MATCH (a {group_id: $gid})-[e1]->(b {group_id: $gid})
            MATCH (a)-[e2]->(b)
            WHERE e1.uuid < e2.uuid
              AND e1.fact IS NOT NULL AND e2.fact IS NOT NULL
            RETURN a.name AS src, b.name AS tgt,
                   e1.uuid AS uuid1, e1.fact AS fact1,
                   e2.uuid AS uuid2, e2.fact AS fact2
            LIMIT 30
            """
            r = await session.run(conflict_q, gid=agent_id)
            async for rec in r:
                result["conflict_edges"].append({
                    "src": rec["src"],
                    "tgt": rec["tgt"],
                    "edge1": {"uuid": rec["uuid1"], "fact": rec["fact1"]},
                    "edge2": {"uuid": rec["uuid2"], "fact": rec["fact2"]},
                })

            report.conflict_pairs = len(result["conflict_edges"])
            logger.info(f"[orient] conflicting edges: {report.conflict_pairs}")

            # 1d. find stale nodes (created_at older than stale_days, no edge references)
            stale_cutoff = datetime.now(timezone.utc) - timedelta(days=self.stale_days)
            stale_q = """
            MATCH (n {group_id: $gid})
            WHERE n.created_at < $cutoff
            AND NOT (n)--()
            RETURN n.uuid AS uuid, n.name AS name, n.created_at AS created_at
            LIMIT 100
            """
            r = await session.run(stale_q, gid=agent_id, cutoff=stale_cutoff.isoformat())
            async for rec in r:
                result["stale_nodes"].append({
                    "uuid": rec["uuid"],
                    "name": rec["name"],
                    "created_at": rec["created_at"],
                })

            report.stale_nodes = len(result["stale_nodes"])
            logger.info(f"[orient] stale orphan nodes: {report.stale_nodes}")

        return result

    # ── Phase 2: Gather ────────────────────────────────────────────────────────

    async def phase2_gather(
        self, agent_id: str, orient_result: dict
    ) -> list[dict]:
        """
        Aggregate candidate issues; each task bundle contains:
        - type: duplicate / conflict / stale
        - nodes/edges references
        - context text for LLM decision-making
        """
        tasks = []
        driver = await self._get_driver()

        async with driver.session() as session:
            # bundle duplicate node tasks
            for group in orient_result["duplicate_groups"][:20]:  # limit to 20 groups per run
                # fetch detailed attributes for each uuid
                details = []
                for uid in group["uuids"][:5]:  # up to 5 per group
                    detail_q = """
                    MATCH (n {uuid: $uid})
                    RETURN n.name AS name, n.summary AS summary,
                           n.created_at AS created_at, n.uuid AS uuid
                    """
                    r = await session.run(detail_q, uid=uid)
                    rec = await r.single()
                    if rec:
                        details.append(dict(rec))

                tasks.append({
                    "type": "duplicate",
                    "group_name": group["name"],
                    "nodes": details,
                    "context": f"Node \"{group['name']}\" has {group['count']} duplicate instances",
                })

            # bundle conflicting edge tasks
            for pair in orient_result["conflict_edges"][:15]:
                tasks.append({
                    "type": "conflict",
                    "src": pair["src"],
                    "tgt": pair["tgt"],
                    "edges": [pair["edge1"], pair["edge2"]],
                    "context": (
                        f"\"{pair['src']}\" -> \"{pair['tgt']}\" has two relationships:\n"
                        f"  1. {pair['edge1']['fact']}\n"
                        f"  2. {pair['edge2']['fact']}"
                    ),
                })

            # bundle stale node tasks (mark directly, no LLM decision needed)
            if orient_result["stale_nodes"]:
                tasks.append({
                    "type": "stale",
                    "nodes": orient_result["stale_nodes"],
                    "context": f"{len(orient_result['stale_nodes'])} orphan nodes older than {self.stale_days} days",
                })

        logger.info(f"[gather] generated {len(tasks)} task bundles")
        return tasks

    # ── Phase 3: Consolidate ───────────────────────────────────────────────────

    async def phase3_consolidate(
        self, tasks: list[dict], agent_id: str, report: DreamReport
    ) -> list[dict]:
        """
        Call LLM (gpt-4o-mini) to make a decision for each task bundle.
        Decision types:
        - merge: merge duplicate nodes (which uuid to keep)
        - keep_latest: keep the newest edge, delete the old one
        - keep_both: keep both edges (different meanings)
        - delete: delete stale nodes
        - skip: skip (insufficient data)
        """
        if not tasks:
            return []

        actions = []

        for task in tasks:
            try:
                if task["type"] == "stale":
                    # stale nodes are marked for deletion directly, no LLM call needed
                    for node in task["nodes"]:
                        actions.append({
                            "action": "delete_node",
                            "uuid": node["uuid"],
                            "reason": t("dream.stale_reason", days=self.stale_days),
                        })
                        report.pruned += 1
                    continue

                # build LLM prompt
                if task["type"] == "duplicate":
                    prompt = t(
                        "dream.consolidate_duplicate",
                        context=task['context'],
                        details=json.dumps(task['nodes'], ensure_ascii=False, indent=2),
                    )

                elif task["type"] == "conflict":
                    prompt = t(
                        "dream.consolidate_conflict",
                        context=task['context'],
                    )

                else:
                    continue

                try:
                    content = await asyncio.wait_for(
                        chat_complete(
                            prompt=prompt,
                            max_tokens=200,
                            temperature=0.0,
                            json_mode=True,
                        ),
                        timeout=60,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        f"[consolidate] LLM call timed out for {task['type']} task"
                    )
                    continue

                decision = parse_llm_json(content)
                decision["task_type"] = task["type"]
                actions.append(decision)

                # update report counts
                action = decision.get("action", "skip")
                if action == "merge":
                    report.merged += 1
                elif action == "keep_latest":
                    report.updated += 1
                elif action == "keep_both":
                    report.kept += 1
                else:
                    report.kept += 1

                logger.info(
                    f"[consolidate] {task['type']} → {action}: {decision.get('reason', '')[:80]}"
                )

            except Exception as e:
                logger.warning(f"[consolidate] task processing failed: {e}")
                actions.append({
                    "action": "skip",
                    "task_type": task.get("type"),
                    "reason": str(e),
                })

        return actions

    # ── Phase 4: Prune ─────────────────────────────────────────────────────────

    async def phase4_prune(
        self, actions: list[dict], agent_id: str, report: DreamReport
    ):
        """
        Execute graph cleanup operations:
        - merge: transfer all relationships from non-primary nodes to the primary node, delete duplicates
        - keep_latest: delete the specified edge
        - delete_node: delete orphan nodes
        """
        if not actions:
            logger.info("[prune] no cleanup needed")
            return

        driver = await self._get_driver()
        executed = 0

        async with driver.session() as session:
            for action in actions:
                act = action.get("action", "skip")
                try:
                    if act == "skip" or act == "keep_both":
                        continue

                    elif act == "merge":
                        keep_uuid = action.get("keep_uuid")
                        if not keep_uuid:
                            continue
                        # find other nodes with the same name, transfer their relationships to the primary node, then delete
                        # conservative approach: only delete duplicate nodes that have no relationships
                        merge_q = """
                        MATCH (n {uuid: $uuid})
                        WITH n
                        MATCH (dup {name: n.name, group_id: $gid})
                        WHERE dup.uuid <> $uuid AND NOT (dup)--()
                        DELETE dup
                        RETURN count(dup) AS deleted
                        """
                        r = await session.run(merge_q, uuid=keep_uuid, gid=agent_id)
                        rec = await r.single()
                        deleted = rec["deleted"] if rec else 0
                        logger.info(f"[prune] merge: keep {keep_uuid}, deleted {deleted} isolated duplicate nodes")
                        executed += deleted

                    elif act == "keep_latest":
                        delete_uuid = action.get("delete_uuid")
                        if not delete_uuid:
                            continue
                        del_q = """
                        MATCH ()-[e {uuid: $uuid, group_id: $gid}]->()
                        DELETE e
                        """
                        await session.run(del_q, uuid=delete_uuid, gid=agent_id)
                        logger.info(f"[prune] keep_latest: deleted old edge {delete_uuid}")
                        executed += 1

                    elif act == "delete_node":
                        node_uuid = action.get("uuid")
                        if not node_uuid:
                            continue
                        del_node_q = """
                        MATCH (n {uuid: $uuid, group_id: $gid})
                        WHERE NOT (n)--()
                        DELETE n
                        """
                        await session.run(del_node_q, uuid=node_uuid, gid=agent_id)
                        logger.info(f"[prune] delete_node: deleted stale node {node_uuid}")
                        executed += 1

                except Exception as e:
                    logger.warning(f"[prune] executing {act} failed: {e}")

        logger.info(f"[prune] executed {executed} graph changes total")

    # ── Phase 5: TTL Expiry Deletion ─────────────────────────────────────────────

    async def phase5_ttl_expire(self, agent_id: str, report: DreamReport):
        """
        Delete nodes that have exceeded TTL and have memocore_confidence < 0.3.
        High-confidence nodes (>= 0.3) are kept even if TTL is exceeded (important memories are not force-deleted).
        """
        ttl_cutoff = datetime.now(timezone.utc) - timedelta(days=self.ttl_days)
        driver = await self._get_driver()

        async with driver.session() as session:
            expire_q = """
            MATCH (n {group_id: $gid})
            WHERE n.created_at < $cutoff
              AND (n.memocore_confidence IS NULL OR n.memocore_confidence < 0.3)
              AND NOT (n)--()
            WITH n LIMIT 200
            DELETE n
            RETURN count(n) AS deleted
            """
            r = await session.run(expire_q, gid=agent_id, cutoff=ttl_cutoff.isoformat())
            rec = await r.single()
            deleted = rec["deleted"] if rec else 0
            report.expired = deleted
            logger.info(f"[ttl] TTL expired, deleted {deleted} nodes (ttl={self.ttl_days}d)")

    # ── Phase 6: Confidence Decay ────────────────────────────────────────────────

    async def phase6_decay_confidence(self, agent_id: str, report: DreamReport):
        """
        Lower confidence scores for nodes that have not been referenced for a long time.

        Rules:
          - Node not referenced by any edge in 30 days → confidence reduced by 0.1 (minimum 0.1)
          - New nodes default memocore_confidence = 1.0 (treated as 1.0 if absent)
          - Confidence >= 0.7: marked as confirmed (priority recall)
          - Confidence 0.3-0.7: marked as tentative (normal recall)
          - Confidence < 0.3: marked as stale (decay on recall)
        """
        idle_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        driver = await self._get_driver()

        async with driver.session() as session:
            # find nodes not referenced by any edge in the last 30 days
            decay_q = """
            MATCH (n {group_id: $gid})
            WHERE n.created_at < $cutoff AND NOT (n)--()
            WITH n,
              CASE
                WHEN n.memocore_confidence IS NULL THEN 0.9
                WHEN n.memocore_confidence - 0.1 < 0.1 THEN 0.1
                ELSE n.memocore_confidence - 0.1
              END AS new_conf
            LIMIT 500
            SET n.memocore_confidence = new_conf,
                n.memocore_status = CASE
                  WHEN new_conf >= 0.7 THEN 'confirmed'
                  WHEN new_conf >= 0.3 THEN 'tentative'
                  ELSE 'stale'
                END
            RETURN count(n) AS updated
            """
            r = await session.run(decay_q, gid=agent_id, cutoff=idle_cutoff.isoformat())
            rec = await r.single()
            updated = rec["updated"] if rec else 0
            report.confidence_lowered = updated
            logger.info(f"[confidence] decayed {updated} nodes")

            # restore confidence for nodes with associated edges (referenced = active = valuable)
            restore_q = """
            MATCH (n {group_id: $gid})
            WHERE (n)--()
              AND (n.memocore_confidence IS NULL OR n.memocore_confidence < 1.0)
            WITH n LIMIT 500
            SET n.memocore_confidence = CASE
                WHEN n.memocore_confidence IS NULL THEN 1.0
                WHEN n.memocore_confidence + 0.1 > 1.0 THEN 1.0
                ELSE n.memocore_confidence + 0.1
            END,
            n.memocore_status = 'confirmed'
            RETURN count(n) AS restored
            """
            r = await session.run(restore_q, gid=agent_id)
            rec = await r.single()
            restored = rec["restored"] if rec else 0
            logger.info(f"[confidence] restored confidence for {restored} active nodes")

    # ── Phase 7: Compile — Knowledge Compilation ────────────────────────────────

    async def phase7_compile(self, agent_id: str, report: DreamReport):
        """
        Compile fragmented facts from the Graph into structured CompiledPage nodes.

        Karpathy philosophy: knowledge should be compiled once, not re-derived from
        fragments on every query. CompiledPage is stored in Neo4j (page_type=entity|concept|overview);
        recall preferentially reads CompiledPage rather than retrieving fragmented edges.

        Process:
        1. Find all entity nodes with edges
        2. For each entity, collect all associated facts
        3. Check for existing CompiledPage with no new information → skip
        4. Call LLM to compile fragmented facts into structured Markdown
        5. MERGE write CompiledPage node, create COMPILED_FROM edge
        6. Compile an overview (global summary page)
        """
        logger.info(f"[compile] starting knowledge compilation agent={agent_id}")
        driver = await self._get_driver()

        async with driver.session() as session:
            # 7a. find all entities with associated edges
            entity_q = """
            MATCH (n {group_id: $gid})-[e]->()
            WHERE n.name IS NOT NULL AND NOT n:CompiledPage
            WITH n, count(e) AS edge_cnt, max(e.created_at) AS latest_edge
            RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                   n.entity_type AS entity_type,
                   n.memocore_confidence AS confidence,
                   edge_cnt, latest_edge
            ORDER BY edge_cnt DESC
            LIMIT $max_nodes
            """
            r = await session.run(entity_q, gid=agent_id, max_nodes=self.max_nodes_per_run)
            entities = await r.data()
            logger.info(f"[compile] found {len(entities)} active entities")

            if not entities:
                return

            compiled_count = 0
            skipped_count = 0

            # parallel compile configuration (enterprise: 200 entities from ~200s → ~20s)
            compile_concurrency = int(os.getenv("MEMOCORE_COMPILE_CONCURRENCY", "10"))
            sem = asyncio.Semaphore(compile_concurrency)

            async def _compile_one(entity: dict) -> tuple[str, bool]:
                """Compile a single entity, returns (entity_name, was_compiled). Each coroutine opens its own Neo4j session."""
                entity_name = entity["name"]
                entity_uuid = entity["uuid"]

                async with sem:
                    driver = await self._get_driver()
                    async with driver.session() as sess:
                        # 7b. check whether recompilation is needed
                        existing_q = """
                        MATCH (p:CompiledPage {group_id: $gid, title: $title})
                        RETURN p.compiled_at AS compiled_at
                        """
                        r = await sess.run(existing_q, gid=agent_id, title=entity_name)
                        existing = await r.single()

                        latest_edge = entity.get("latest_edge")
                        if existing and latest_edge and existing["compiled_at"]:
                            try:
                                compiled_str = str(existing["compiled_at"])[:19]
                                edge_str = str(latest_edge)[:19]
                                if compiled_str >= edge_str:
                                    return entity_name, False  # no new information
                            except Exception:
                                pass

                        # 7c. collect all facts for this entity
                        facts_q = """
                        MATCH (a {group_id: $gid})-[e]->(b {group_id: $gid})
                        WHERE a.uuid = $uuid OR b.uuid = $uuid
                        RETURN a.name AS src, b.name AS tgt, e.fact AS fact,
                               e.created_at AS created_at
                        ORDER BY e.created_at DESC
                        LIMIT 50
                        """
                        r = await sess.run(facts_q, gid=agent_id, uuid=entity_uuid)
                        facts = await r.data()

                        if not facts:
                            return entity_name, False

                        # 7d. LLM compile
                        facts_text = "\n".join(
                            f"- [{f.get('src', '?')} → {f.get('tgt', '?')}] {f.get('fact', '')}"
                            for f in facts
                        )
                        confidence = entity.get("confidence") or 1.0
                        conf_label = "high" if confidence >= 0.7 else ("medium" if confidence >= 0.3 else "low")

                        compile_prompt = t(
                            "dream.compile_entity",
                            entity_name=entity_name,
                            entity_type=entity.get('entity_type', 'unknown'),
                            confidence_label=conf_label,
                            fact_count=len(facts),
                            facts_text=facts_text,
                        )

                        try:
                            compiled_content = await asyncio.wait_for(
                                chat_complete(
                                    prompt=compile_prompt,
                                    max_tokens=800,
                                    temperature=0.0,
                                ),
                                timeout=60,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                f"[compile] LLM call timed out for \"{entity_name}\""
                            )
                            return entity_name, False
                        except Exception as e:
                            logger.warning(f"[compile] LLM compilation of \"{entity_name}\" failed: {e}")
                            return entity_name, False

                        # 7e. MERGE write CompiledPage node
                        now_str = datetime.now(timezone.utc).isoformat()
                        upsert_q = """
                        MERGE (p:CompiledPage {group_id: $gid, title: $title})
                        SET p.content = $content,
                            p.page_type = $page_type,
                            p.confidence = $confidence,
                            p.source_count = $source_count,
                            p.compiled_at = $compiled_at,
                            p.stale = false
                        """
                        await sess.run(upsert_q,
                            gid=agent_id,
                            title=entity_name,
                            content=compiled_content,
                            page_type="entity",
                            confidence=confidence,
                            source_count=len(facts),
                            compiled_at=now_str,
                        )

                        # create COMPILED_FROM edge
                        link_q = """
                        MATCH (p:CompiledPage {group_id: $gid, title: $title})
                        MATCH (n {group_id: $gid, uuid: $uuid})
                        MERGE (p)-[:COMPILED_FROM]->(n)
                        """
                        await sess.run(link_q, gid=agent_id, title=entity_name, uuid=entity_uuid)

                        logger.info(f"[compile] compiled \"{entity_name}\" ({len(facts)} facts)")
                        return entity_name, True

            # execute all entity compilations in parallel
            results = await asyncio.gather(
                *[_compile_one(e) for e in entities],
                return_exceptions=True,
            )
            for result in results:
                if isinstance(result, Exception):
                    logger.warning(f"[compile] entity compilation exception: {result}")
                elif result[1]:
                    compiled_count += 1
                else:
                    skipped_count += 1

            # 7f. compile global overview (if any pages have been compiled)
            overview_q = """
            MATCH (p:CompiledPage {group_id: $gid})
            WHERE p.page_type = 'entity'
            RETURN p.title AS title, p.confidence AS confidence, p.source_count AS source_count
            ORDER BY p.source_count DESC
            """
            r = await session.run(overview_q, gid=agent_id)
            all_pages = await r.data()

            if all_pages:
                page_index = "\n".join(
                    f"- **{p['title']}** (facts: {p.get('source_count', 0)}, "
                    f"confidence: {'high' if (p.get('confidence') or 1) >= 0.7 else 'medium' if (p.get('confidence') or 1) >= 0.3 else 'low'})"
                    for p in all_pages
                )

                overview_prompt = t(
                    "dream.compile_overview",
                    page_count=len(all_pages),
                    page_index=page_index,
                )

                try:
                    overview_content = await asyncio.wait_for(
                        chat_complete(
                            prompt=overview_prompt, max_tokens=400, temperature=0.0
                        ),
                        timeout=60,
                    )
                    now_str = datetime.now(timezone.utc).isoformat()
                    await session.run("""
                        MERGE (p:CompiledPage {group_id: $gid, title: '__overview__'})
                        SET p.content = $content, p.page_type = 'overview',
                            p.compiled_at = $compiled_at, p.source_count = $cnt,
                            p.stale = false
                    """, gid=agent_id, content=overview_content,
                        compiled_at=now_str, cnt=len(all_pages))
                except asyncio.TimeoutError:
                    logger.warning("[compile] LLM call timed out for overview compilation")
                except Exception as e:
                    logger.warning(f"[compile] overview compilation failed: {e}")

            report.compiled_pages = compiled_count
            report.skipped_pages = skipped_count
            logger.info(
                f"[compile] done: compiled={compiled_count} skipped={skipped_count} total_pages={len(all_pages)}"
            )

    # ── Phase 8: Lint — Knowledge Health Check ───────────────────────────────────

    async def phase8_lint(self, agent_id: str, report: DreamReport):
        """
        Check for issues among CompiledPages and output a Karpathy-style health report.

        Checks:
        1. Contradictions — different pages describe the same fact in conflicting ways
        2. Orphan pages — entity pages with no inbound COMPILED_FROM edge (source data cleaned up)
        3. Missing pages — frequently referenced in facts but no CompiledPage exists
        4. Stale pages — compiled more than 14 days ago and source entity has new data

        Output written to get_state_dir()/reports/{agent_id}/ directory
        """
        logger.info(f"[lint] starting health check agent={agent_id}")
        driver = await self._get_driver()
        sections = []

        async with driver.session() as session:
            # 8a. compilation status stats
            stats_q = """
            MATCH (p:CompiledPage {group_id: $gid})
            WHERE p.page_type = 'entity'
            RETURN count(p) AS page_count
            """
            r = await session.run(stats_q, gid=agent_id)
            rec = await r.single()
            page_count = rec["page_count"] if rec else 0

            total_facts_q = """
            MATCH ()-[e {group_id: $gid}]->()
            WHERE e.fact IS NOT NULL
            RETURN count(e) AS cnt
            """
            r = await session.run(total_facts_q, gid=agent_id)
            rec = await r.single()
            total_facts = rec["cnt"] if rec else 0

            sections.append(f"### Compilation Status\n- compiled entity pages: {page_count}\n- total facts: {total_facts}")

            # 8b. contradiction detection — multiple different facts between the same pair of entities
            contradiction_q = """
            MATCH (a {group_id: $gid})-[e1]->(b {group_id: $gid})
            MATCH (a)-[e2]->(b)
            WHERE e1.uuid < e2.uuid
              AND e1.fact IS NOT NULL AND e2.fact IS NOT NULL
              AND e1.fact <> e2.fact
            RETURN a.name AS src, b.name AS tgt,
                   e1.fact AS fact1, e2.fact AS fact2
            LIMIT 20
            """
            r = await session.run(contradiction_q, gid=agent_id)
            contradictions = await r.data()
            report.lint_contradictions = len(contradictions)

            if contradictions:
                lines = [f"### Potential Contradictions ({len(contradictions)})"]
                for c in contradictions:
                    lines.append(
                        f"- **{c['src']}** -> **{c['tgt']}**\n"
                        f"  - Fact A: {c['fact1'][:120]}\n"
                        f"  - Fact B: {c['fact2'][:120]}"
                    )
                sections.append("\n".join(lines))

            # 8c. orphan CompiledPage — no COMPILED_FROM edge
            orphan_q = """
            MATCH (p:CompiledPage {group_id: $gid})
            WHERE p.page_type = 'entity'
              AND NOT (p)-[:COMPILED_FROM]->()
            RETURN p.title AS title
            """
            r = await session.run(orphan_q, gid=agent_id)
            orphans = await r.data()
            report.lint_orphans = len(orphans)

            if orphans:
                lines = [f"### Orphan Pages ({len(orphans)})"]
                for o in orphans:
                    lines.append(f"- `{o['title']}` — source entity cleaned up, recommend deletion or recompilation")
                sections.append("\n".join(lines))

            # 8d. missing pages — entity names frequently referenced but lacking a CompiledPage
            missing_q = """
            MATCH (n {group_id: $gid})
            WHERE n.name IS NOT NULL
              AND NOT n:CompiledPage
              AND (n)--()
            WITH n.name AS name, count{(n)--() } AS ref_count
            WHERE ref_count >= 2
            AND NOT EXISTS {
                MATCH (p:CompiledPage {group_id: $gid, title: name})
            }
            RETURN name, ref_count
            ORDER BY ref_count DESC
            LIMIT 15
            """
            try:
                r = await session.run(missing_q, gid=agent_id)
                missing = await r.data()
            except Exception:
                # older Neo4j does not support count{} syntax, fall back to compatible query
                missing_q_compat = """
                MATCH (n {group_id: $gid})-[e]-()
                WHERE n.name IS NOT NULL AND NOT n:CompiledPage
                WITH n.name AS name, count(e) AS ref_count
                WHERE ref_count >= 2
                RETURN name, ref_count
                ORDER BY ref_count DESC
                LIMIT 15
                """
                r = await session.run(missing_q_compat, gid=agent_id)
                candidates = await r.data()
                # manually exclude entities that already have a CompiledPage
                existing_pages_q = """
                MATCH (p:CompiledPage {group_id: $gid})
                RETURN p.title AS title
                """
                r2 = await session.run(existing_pages_q, gid=agent_id)
                existing_titles = {rec["title"] for rec in await r2.data()}
                missing = [c for c in candidates if c["name"] not in existing_titles]

            report.lint_missing = len(missing)

            if missing:
                lines = [f"### Missing Pages ({len(missing)})"]
                for m in missing:
                    lines.append(f"- **{m['name']}** — referenced {m['ref_count']} times, recommend compilation")
                sections.append("\n".join(lines))

            # 8e. stale pages — compiled more than 14 days ago
            stale_q = """
            MATCH (p:CompiledPage {group_id: $gid})
            WHERE p.page_type = 'entity'
              AND p.compiled_at < $cutoff
            RETURN p.title AS title, p.compiled_at AS compiled_at
            """
            cutoff_14d = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()
            r = await session.run(stale_q, gid=agent_id, cutoff=cutoff_14d)
            stale_pages = await r.data()
            report.lint_stale_pages = len(stale_pages)

            if stale_pages:
                lines = [f"### Stale Pages ({len(stale_pages)})"]
                for s in stale_pages:
                    lines.append(f"- `{s['title']}` — compiled at {str(s['compiled_at'])[:10]}, recommend recompilation")
                sections.append("\n".join(lines))

        # 8f. write report file
        report_dir = get_state_dir() / "reports" / make_safe_agent_key(agent_id)
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        report_md = (
            f"## Memory Health Report — {agent_id} | {now_str}\n\n"
            + "\n\n".join(sections)
        )

        if not (report.lint_contradictions or report.lint_orphans
                or report.lint_missing or report.lint_stale_pages):
            report_md += "\n\n*All clear — no issues found.*"

        report_file.write_text(report_md, encoding="utf-8")
        report.lint_report_path = str(report_file)
        logger.info(f"[lint] report written to {report_file}")


# ─── Main entrypoint ─────────────────────────────────────────────────────────

async def run_dream(
    agent_id: str = "default",
    dry_run: bool = False,
) -> DreamReport:
    """
    Run the complete Dream consolidation pipeline.

    Args:
        agent_id: Agent namespace
        dry_run: When True, scan only without executing cleanup (for debugging)

    Returns:
        DreamReport — run report
    """
    agent_id = validate_agent_id(agent_id)
    report = DreamReport(agent_id=agent_id)
    consolidator = DreamConsolidator()

    # clean up stale session flag files (prevent accumulation of millions of files)
    try:
        cleaned = cleanup_old_session_flags()
        if cleaned:
            logger.info(f"[cleanup] cleaned up {cleaned} stale session flag files")
    except Exception as e:
        logger.warning(f"[cleanup] session flag cleanup failed: {e}")

    logger.info(f"=== Dream starting | agent={agent_id} dry_run={dry_run} ===")

    try:
        # Phase 1: Orient
        logger.info("[phase1] Orient — scanning graph")
        orient_result = await consolidator.phase1_orient(agent_id, report)

        # Phase 2: Gather
        logger.info("[phase2] Gather — aggregating candidate issues")
        tasks = await consolidator.phase2_gather(agent_id, orient_result)

        # Phase 3: Consolidate
        logger.info("[phase3] Consolidate — LLM decision")
        actions = await consolidator.phase3_consolidate(tasks, agent_id, report)

        # Phase 4: Prune
        if dry_run:
            logger.info(f"[phase4] Prune — dry_run=True, skipping execution, planned operations={len(actions)}")
        else:
            logger.info("[phase4] Prune — executing graph cleanup")
            await consolidator.phase4_prune(actions, agent_id, report)

        # Phase 5: TTL expiry deletion
        if not dry_run:
            logger.info("[phase5] TTL — deleting expired low-confidence nodes")
            await consolidator.phase5_ttl_expire(agent_id, report)

        # Phase 6: Confidence decay and restore
        if not dry_run:
            logger.info("[phase6] Confidence — updating node confidence scores")
            await consolidator.phase6_decay_confidence(agent_id, report)

        # Phase 7: Compile — knowledge compilation (fragmented facts → structured CompiledPage)
        if not dry_run:
            logger.info("[phase7] Compile — compiling structured knowledge pages")
            await consolidator.phase7_compile(agent_id, report)

        # Phase 8: Lint — knowledge health check
        logger.info("[phase8] Lint — health check")
        await consolidator.phase8_lint(agent_id, report)

        report.status = "done"
        report.finished_at = datetime.now(timezone.utc)
        logger.info(f"=== Dream complete === {report.summary()}")

    except Exception as e:
        report.status = "failed"
        report.error = str(e)
        report.finished_at = datetime.now(timezone.utc)
        logger.error(f"=== Dream failed: {e} ===")

    finally:
        await consolidator.close()

    return report


def main():
    """CLI entrypoint for running Dream consolidation."""
    import argparse

    _configure_dream_logging()

    parser = argparse.ArgumentParser(description="Memocore Dream — memory consolidation")
    parser.add_argument("--agent-id", default="default", help="Agent namespace")
    parser.add_argument("--dry-run", action="store_true", help="Scan only, do not execute cleanup")
    args = parser.parse_args()

    report = asyncio.run(run_dream(agent_id=args.agent_id, dry_run=args.dry_run))
    print(report.summary())


if __name__ == "__main__":
    main()
