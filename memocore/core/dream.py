"""
MemOS Dream 机制 — 记忆巩固（借鉴 Claude Code Dream 4阶段设计）

触发时机：Stop hook 完成后，异步后台运行
四阶段：
  Phase 1 Orient  — 扫描图谱，找出候选问题区域（重复、矛盾、过期）
  Phase 2 Gather  — 聚合同类节点的关联 edges
  Phase 3 Consolidate — LLM 决策：合并 / 更新 / 保留 / 删除
  Phase 4 Prune   — 执行图谱清理，写入操作日志

设计目标：
- 保持图谱整洁，避免 Graphiti 里节点爆炸
- 矛盾事实自动解决（取最新版本）
- 过期信息降权或删除（> 30 天且未被引用）

用法：
    from memocore.core.dream import run_dream
    await run_dream(agent_id="aoxia")

    # 或作为独立脚本运行（供 cron / stop hook 调用）
    python -m memos.core.dream
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from memocore.core.llm_adapter import chat_complete
from memocore.core.config import get_dream_ttl_days

_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

LOG_FILE = Path.home() / ".private" / "memos_dream.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [dream] %(levelname)s %(message)s",
)
logger = logging.getLogger("memos.dream")


# ─── 数据结构 ──────────────────────────────────────────────────────────────────

@dataclass
class DreamReport:
    """Dream 运行报告"""
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

    # TTL + 置信度
    expired: int = 0          # 超过 TTL 被删除的节点
    confidence_lowered: int = 0  # 置信度被降低的节点

    # Phase 7 Compile
    compiled_pages: int = 0       # 本次编译/更新的 CompiledPage 数
    skipped_pages: int = 0        # 无新信息跳过的 CompiledPage 数

    # Phase 8 Lint
    lint_contradictions: int = 0
    lint_orphans: int = 0
    lint_missing: int = 0
    lint_stale_pages: int = 0
    lint_report_path: Optional[str] = None

    # 状态
    status: str = "running"   # running / done / failed
    error: Optional[str] = None

    def summary(self) -> str:
        elapsed = ""
        if self.finished_at:
            secs = (self.finished_at - self.started_at).total_seconds()
            elapsed = f" | 耗时 {secs:.1f}s"
        return (
            f"[Dream {self.agent_id}] {self.status}{elapsed} | "
            f"节点={self.total_nodes} edges={self.total_edges} | "
            f"重复={self.duplicate_groups} 矛盾={self.conflict_pairs} "
            f"过期={self.stale_nodes} 到期删除={self.expired} 降权={self.confidence_lowered} | "
            f"合并={self.merged} 更新={self.updated} 剪枝={self.pruned} 保留={self.kept} | "
            f"编译={self.compiled_pages} lint矛盾={self.lint_contradictions} "
            f"lint孤立={self.lint_orphans} lint缺页={self.lint_missing}"
        )


# ─── Dream 核心类 ──────────────────────────────────────────────────────────────

class DreamConsolidator:
    """
    MemOS Dream 机制主控
    直接操作 Neo4j（通过 graphiti 封装），不依赖 LLM 的 add_episode 路径
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        stale_days: int = 30,
        max_nodes_per_run: int = 200,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.stale_days = stale_days
        self.ttl_days = get_dream_ttl_days()
        self.max_nodes_per_run = max_nodes_per_run
        self._driver = None

    async def _get_driver(self):
        """获取 Neo4j 驱动（lazy init）"""
        if self._driver is None:
            try:
                from neo4j import AsyncGraphDatabase
                self._driver = AsyncGraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )
            except ImportError:
                raise RuntimeError(
                    "需要安装 neo4j 驱动: pip install neo4j"
                )
        return self._driver

    async def close(self):
        if self._driver:
            await self._driver.close()
            self._driver = None

    # ── Phase 1: Orient ────────────────────────────────────────────────────────

    async def phase1_orient(self, agent_id: str, report: DreamReport) -> dict:
        """
        扫描图谱，返回三类候选问题：
        - duplicate_groups: 名称相似的节点组（可能需要合并）
        - conflict_edges: 同主语/同谓语但结论相反的 edges
        - stale_nodes: 超过 stale_days 天且 reference_count=0 的节点
        """
        logger.info(f"[orient] agent={agent_id} 开始扫描图谱...")
        driver = await self._get_driver()

        result = {
            "duplicate_groups": [],
            "conflict_edges": [],
            "stale_nodes": [],
        }

        async with driver.session() as session:
            # 1a. 统计总量
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
                f"[orient] 图谱规模: nodes={report.total_nodes} edges={report.total_edges}"
            )

            # 1b. 找重复节点（同 name 不同 uuid，取最多 50 组）
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
            logger.info(f"[orient] 重复节点组: {report.duplicate_groups}")

            # 1c. 找矛盾 edges（同 source+target 但 fact 关键词相反，取最多 30 对）
            # 策略：找同一对节点之间有多条 RELATES_TO edge 的情况
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
            logger.info(f"[orient] 矛盾 edges: {report.conflict_pairs}")

            # 1d. 找过期节点（created_at 超过 stale_days，无任何 edge 引用）
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
            logger.info(f"[orient] 过期孤立节点: {report.stale_nodes}")

        return result

    # ── Phase 2: Gather ────────────────────────────────────────────────────────

    async def phase2_gather(
        self, agent_id: str, orient_result: dict
    ) -> list[dict]:
        """
        聚合候选问题，每个"任务包"包含：
        - type: duplicate / conflict / stale
        - nodes/edges 引用
        - 供 LLM 判断用的上下文文本
        """
        tasks = []
        driver = await self._get_driver()

        async with driver.session() as session:
            # 打包重复节点任务
            for group in orient_result["duplicate_groups"][:20]:  # 限制每次最多20组
                # 拉取每个 uuid 的详细属性
                details = []
                for uid in group["uuids"][:5]:  # 每组最多5个
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
                    "context": f"节点名称「{group['name']}」有 {group['count']} 个重复实例",
                })

            # 打包矛盾 edges 任务
            for pair in orient_result["conflict_edges"][:15]:
                tasks.append({
                    "type": "conflict",
                    "src": pair["src"],
                    "tgt": pair["tgt"],
                    "edges": [pair["edge1"], pair["edge2"]],
                    "context": (
                        f"「{pair['src']}」→「{pair['tgt']}」存在两条关系：\n"
                        f"  1. {pair['edge1']['fact']}\n"
                        f"  2. {pair['edge2']['fact']}"
                    ),
                })

            # 打包过期节点（直接标记，不需要LLM判断）
            if orient_result["stale_nodes"]:
                tasks.append({
                    "type": "stale",
                    "nodes": orient_result["stale_nodes"],
                    "context": f"共 {len(orient_result['stale_nodes'])} 个超过 {self.stale_days} 天的孤立节点",
                })

        logger.info(f"[gather] 生成 {len(tasks)} 个任务包")
        return tasks

    # ── Phase 3: Consolidate ───────────────────────────────────────────────────

    async def phase3_consolidate(
        self, tasks: list[dict], agent_id: str, report: DreamReport
    ) -> list[dict]:
        """
        调用 LLM（gpt-4o-mini）对每个任务包作出决策
        决策类型：
        - merge: 合并重复节点（保留哪个 uuid）
        - keep_latest: 保留最新 edge，删除旧的
        - keep_both: 两条 edge 都保留（含义不同）
        - delete: 删除过期节点
        - skip: 跳过（数据不够充分）
        """
        if not tasks:
            return []

        actions = []

        for task in tasks:
            try:
                if task["type"] == "stale":
                    # 过期节点直接标记删除，不调 LLM
                    for node in task["nodes"]:
                        actions.append({
                            "action": "delete_node",
                            "uuid": node["uuid"],
                            "reason": f"孤立节点超过 {self.stale_days} 天",
                        })
                        report.pruned += 1
                    continue

                # 构建 LLM prompt
                if task["type"] == "duplicate":
                    prompt = f"""你是记忆图谱清洁工。以下是同名节点的详情：

{task['context']}

节点详情：
{json.dumps(task['nodes'], ensure_ascii=False, indent=2)}

请决定：
1. merge — 合并为一个，指定保留的 uuid（最完整的那个）
2. skip — 数据不足，跳过

仅返回 JSON：{{"action": "merge或skip", "keep_uuid": "...", "reason": "..."}}"""

                elif task["type"] == "conflict":
                    prompt = f"""你是记忆图谱清洁工。以下是两条可能矛盾的关系：

{task['context']}

请决定：
1. keep_latest — 保留时间更新的那条（通常是正确的），删除旧的
2. keep_both — 两条含义不同，都保留
3. skip — 数据不足

仅返回 JSON：{{"action": "keep_latest或keep_both或skip", "delete_uuid": "...", "reason": "..."}}"""

                else:
                    continue

                content = await chat_complete(
                    prompt=prompt,
                    max_tokens=200,
                    temperature=0.0,
                    json_mode=True,
                )

                decision = json.loads(content)
                decision["task_type"] = task["type"]
                actions.append(decision)

                # 更新报告计数
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
                logger.warning(f"[consolidate] 任务处理失败: {e}")
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
        执行图谱清理操作
        - merge: 把非主节点的所有关系转移到主节点，删除副本
        - keep_latest: 删除指定 edge
        - delete_node: 删除孤立节点
        """
        if not actions:
            logger.info("[prune] 无需执行清理")
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
                        # 找出同名的其他节点，把它们的关系接到主节点，再删除
                        # 这里做保守操作：只删除没有任何关系的重复节点
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
                        logger.info(f"[prune] merge: 保留 {keep_uuid}, 删除 {deleted} 个孤立重复节点")
                        executed += deleted

                    elif act == "keep_latest":
                        delete_uuid = action.get("delete_uuid")
                        if not delete_uuid:
                            continue
                        del_q = """
                        MATCH ()-[e {uuid: $uuid}]->()
                        DELETE e
                        """
                        await session.run(del_q, uuid=delete_uuid)
                        logger.info(f"[prune] keep_latest: 删除旧 edge {delete_uuid}")
                        executed += 1

                    elif act == "delete_node":
                        node_uuid = action.get("uuid")
                        if not node_uuid:
                            continue
                        del_node_q = """
                        MATCH (n {uuid: $uuid})
                        WHERE NOT (n)--()
                        DELETE n
                        """
                        await session.run(del_node_q, uuid=node_uuid)
                        logger.info(f"[prune] delete_node: 删除过期节点 {node_uuid}")
                        executed += 1

                except Exception as e:
                    logger.warning(f"[prune] 执行 {act} 失败: {e}")

        logger.info(f"[prune] 共执行 {executed} 次图谱变更")

    # ── Phase 5: TTL 过期删除 ────────────────────────────────────────────────────

    async def phase5_ttl_expire(self, agent_id: str, report: DreamReport):
        """
        删除超过 TTL 且 memocore_confidence < 0.3 的节点。
        高置信度节点（>= 0.3）即使超过 TTL 也保留（重要记忆不强制删除）。
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
            logger.info(f"[ttl] TTL 过期删除 {deleted} 个节点 (ttl={self.ttl_days}d)")

    # ── Phase 6: 置信度衰减 ──────────────────────────────────────────────────────

    async def phase6_decay_confidence(self, agent_id: str, report: DreamReport):
        """
        对长期未被引用的节点降低置信度评分。

        规则：
          - 节点 30 天内未被任何 edge 引用 → confidence 降低 0.1（最低 0.1）
          - 新建节点默认 memocore_confidence = 1.0（不存在时视为 1.0）
          - 置信度 >= 0.7：标记为 confirmed（优先召回）
          - 置信度 0.3-0.7：标记为 tentative（正常召回）
          - 置信度 < 0.3：标记为 stale（召回时降权）
        """
        idle_cutoff = datetime.now(timezone.utc) - timedelta(days=30)
        driver = await self._get_driver()

        async with driver.session() as session:
            # 找出 30 天未被 edge 引用的节点
            decay_q = """
            MATCH (n {group_id: $gid})
            WHERE n.created_at < $cutoff AND NOT (n)--()
            SET n.memocore_confidence = CASE
                WHEN n.memocore_confidence IS NULL THEN 0.9
                WHEN n.memocore_confidence - 0.1 < 0.1 THEN 0.1
                ELSE n.memocore_confidence - 0.1
            END,
            n.memocore_status = CASE
                WHEN (CASE WHEN n.memocore_confidence IS NULL THEN 0.9
                      ELSE n.memocore_confidence END) - 0.1 >= 0.7 THEN 'confirmed'
                WHEN (CASE WHEN n.memocore_confidence IS NULL THEN 0.9
                      ELSE n.memocore_confidence END) - 0.1 >= 0.3 THEN 'tentative'
                ELSE 'stale'
            END
            WITH n LIMIT 500
            RETURN count(n) AS updated
            """
            r = await session.run(decay_q, gid=agent_id, cutoff=idle_cutoff.isoformat())
            rec = await r.single()
            updated = rec["updated"] if rec else 0
            report.confidence_lowered = updated
            logger.info(f"[confidence] 降权 {updated} 个节点")

            # 对有关联 edge 的节点恢复置信度（被引用 = 活跃 = 有价值）
            restore_q = """
            MATCH (n {group_id: $gid})
            WHERE (n)--()
              AND (n.memocore_confidence IS NULL OR n.memocore_confidence < 1.0)
            SET n.memocore_confidence = CASE
                WHEN n.memocore_confidence IS NULL THEN 1.0
                WHEN n.memocore_confidence + 0.1 > 1.0 THEN 1.0
                ELSE n.memocore_confidence + 0.1
            END,
            n.memocore_status = 'confirmed'
            WITH n LIMIT 500
            RETURN count(n) AS restored
            """
            r = await session.run(restore_q, gid=agent_id)
            rec = await r.single()
            restored = rec["restored"] if rec else 0
            logger.info(f"[confidence] 恢复 {restored} 个活跃节点置信度")

    # ── Phase 7: Compile — 知识编译 ─────────────────────────────────────────────

    async def phase7_compile(self, agent_id: str, report: DreamReport):
        """
        将 Graph 中的碎片 facts 编译成结构化 CompiledPage 节点。

        Karpathy 理念：知识应被编译一次，而非每次查询都从碎片重新推导。
        CompiledPage 存储在 Neo4j 中（page_type=entity|concept|overview），
        recall 时优先读取 CompiledPage 而非检索碎片 edges。

        流程：
        1. 找出所有有 edges 的实体节点
        2. 对每个实体，收集其关联的所有 facts
        3. 检查是否有已存在的 CompiledPage 且无新信息 → 跳过
        4. 调用 LLM 将碎片 facts 编译成结构化 Markdown
        5. MERGE 写入 CompiledPage 节点，建 COMPILED_FROM 边
        6. 编译一个 overview（全局概览页）
        """
        logger.info(f"[compile] 开始知识编译 agent={agent_id}")
        driver = await self._get_driver()

        async with driver.session() as session:
            # 7a. 找出所有有 edge 关联的实体
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
            logger.info(f"[compile] 找到 {len(entities)} 个活跃实体")

            if not entities:
                return

            compiled_count = 0
            skipped_count = 0

            for entity in entities:
                entity_name = entity["name"]
                entity_uuid = entity["uuid"]

                # 7b. 检查是否需要重编译
                existing_q = """
                MATCH (p:CompiledPage {group_id: $gid, title: $title})
                RETURN p.compiled_at AS compiled_at
                """
                r = await session.run(existing_q, gid=agent_id, title=entity_name)
                existing = await r.single()

                latest_edge = entity.get("latest_edge")
                if existing and latest_edge and existing["compiled_at"]:
                    try:
                        compiled_str = str(existing["compiled_at"])[:19]
                        edge_str = str(latest_edge)[:19]
                        if compiled_str >= edge_str:
                            skipped_count += 1
                            continue  # 无新信息
                    except Exception:
                        pass  # 时间比较失败时总是重编译

                # 7c. 收集该实体的所有 facts
                facts_q = """
                MATCH (a {group_id: $gid})-[e]->(b {group_id: $gid})
                WHERE a.uuid = $uuid OR b.uuid = $uuid
                RETURN a.name AS src, b.name AS tgt, e.fact AS fact,
                       e.created_at AS created_at
                ORDER BY e.created_at DESC
                LIMIT 50
                """
                r = await session.run(facts_q, gid=agent_id, uuid=entity_uuid)
                facts = await r.data()

                if not facts:
                    skipped_count += 1
                    continue

                # 7d. LLM 编译碎片 → 结构化知识
                facts_text = "\n".join(
                    f"- [{f.get('src', '?')} → {f.get('tgt', '?')}] {f.get('fact', '')}"
                    for f in facts
                )

                confidence = entity.get("confidence") or 1.0
                conf_label = "high" if confidence >= 0.7 else ("medium" if confidence >= 0.3 else "low")

                compile_prompt = f"""你是知识编译器。将以下关于「{entity_name}」的碎片信息编译成一个结构化的知识页面。

实体类型: {entity.get('entity_type', '未知')}
当前置信度: {conf_label}

碎片信息（{len(facts)} 条）:
{facts_text}

要求:
1. 合并重复信息，去除冗余
2. 如有矛盾，以最新的为准，旧的标注为「存疑」
3. 按主题分段（如：概述、偏好、关系、近期动态）
4. 用简洁的中文 Markdown 格式
5. 不要编造碎片中没有的信息
6. 总长度控制在 500 字以内"""

                try:
                    compiled_content = await chat_complete(
                        prompt=compile_prompt,
                        max_tokens=800,
                        temperature=0.0,
                    )
                except Exception as e:
                    logger.warning(f"[compile] LLM 编译「{entity_name}」失败: {e}")
                    continue

                # 7e. MERGE 写入 CompiledPage 节点
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
                await session.run(upsert_q,
                    gid=agent_id,
                    title=entity_name,
                    content=compiled_content,
                    page_type="entity",
                    confidence=confidence,
                    source_count=len(facts),
                    compiled_at=now_str,
                )

                # 建 COMPILED_FROM 边（链接到源实体）
                link_q = """
                MATCH (p:CompiledPage {group_id: $gid, title: $title})
                MATCH (n {group_id: $gid, uuid: $uuid})
                MERGE (p)-[:COMPILED_FROM]->(n)
                """
                await session.run(link_q, gid=agent_id, title=entity_name, uuid=entity_uuid)

                compiled_count += 1
                logger.info(f"[compile] 编译「{entity_name}」完成 ({len(facts)} facts)")

            # 7f. 编译全局 overview（如果有编译过的页面）
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

                overview_prompt = f"""你是知识编译器。根据以下已编译的实体页面列表，写一段 200 字以内的总体概述，描述这个 Agent 记忆中的核心人物、关注领域和重要决策。

已编译页面（{len(all_pages)} 个）:
{page_index}

要求: 简洁，不要罗列，提炼模式和核心信息。"""

                try:
                    overview_content = await chat_complete(
                        prompt=overview_prompt, max_tokens=400, temperature=0.0
                    )
                    now_str = datetime.now(timezone.utc).isoformat()
                    await session.run("""
                        MERGE (p:CompiledPage {group_id: $gid, title: '__overview__'})
                        SET p.content = $content, p.page_type = 'overview',
                            p.compiled_at = $compiled_at, p.source_count = $cnt,
                            p.stale = false
                    """, gid=agent_id, content=overview_content,
                        compiled_at=now_str, cnt=len(all_pages))
                except Exception as e:
                    logger.warning(f"[compile] overview 编译失败: {e}")

            report.compiled_pages = compiled_count
            report.skipped_pages = skipped_count
            logger.info(
                f"[compile] 完成: 编译={compiled_count} 跳过={skipped_count} 总页面={len(all_pages)}"
            )

    # ── Phase 8: Lint — 知识健康检查 ─────────────────────────────────────────────

    async def phase8_lint(self, agent_id: str, report: DreamReport):
        """
        检查 CompiledPage 之间的问题，输出 Karpathy 风格的健康报告。

        检查项:
        1. 矛盾 — 不同页面对同一事实描述冲突
        2. 孤立页 — 无入站 COMPILED_FROM 边的实体页（源数据已被清理）
        3. 缺页 — 在 facts 中被频繁提及但无 CompiledPage
        4. 过期页 — 编译时间超过 14 天且源实体有新数据

        输出写入 ~/.memocore/reports/{agent_id}/ 目录
        """
        logger.info(f"[lint] 开始健康检查 agent={agent_id}")
        driver = await self._get_driver()
        sections = []

        async with driver.session() as session:
            # 8a. 编译状态统计
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

            sections.append(f"### 编译状态\n- 已编译实体页: {page_count}\n- 总 facts: {total_facts}")

            # 8b. 矛盾检测 — 同一对实体之间有多条不同 facts
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
                lines = [f"### 潜在矛盾 ({len(contradictions)})"]
                for c in contradictions:
                    lines.append(
                        f"- **{c['src']}** -> **{c['tgt']}**\n"
                        f"  - Fact A: {c['fact1'][:120]}\n"
                        f"  - Fact B: {c['fact2'][:120]}"
                    )
                sections.append("\n".join(lines))

            # 8c. 孤立 CompiledPage — 没有 COMPILED_FROM 边
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
                lines = [f"### 孤立页面 ({len(orphans)})"]
                for o in orphans:
                    lines.append(f"- `{o['title']}` — 源实体已被清理，建议删除或重编译")
                sections.append("\n".join(lines))

            # 8d. 缺页 — 被频繁提及但无 CompiledPage 的实体名
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
                # 旧版 Neo4j 不支持 count{} 语法，降级查询
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
                # 手动排除已有 CompiledPage 的
                existing_pages_q = """
                MATCH (p:CompiledPage {group_id: $gid})
                RETURN p.title AS title
                """
                r2 = await session.run(existing_pages_q, gid=agent_id)
                existing_titles = {rec["title"] for rec in await r2.data()}
                missing = [c for c in candidates if c["name"] not in existing_titles]

            report.lint_missing = len(missing)

            if missing:
                lines = [f"### 缺失页面 ({len(missing)})"]
                for m in missing:
                    lines.append(f"- **{m['name']}** — 被引用 {m['ref_count']} 次，建议编译")
                sections.append("\n".join(lines))

            # 8e. 过期页 — 编译时间超过 14 天
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
                lines = [f"### 过期页面 ({len(stale_pages)})"]
                for s in stale_pages:
                    lines.append(f"- `{s['title']}` — 编译于 {str(s['compiled_at'])[:10]}，建议重编译")
                sections.append("\n".join(lines))

        # 8f. 写入报告文件
        report_dir = Path.home() / ".memocore" / "reports" / agent_id
        report_dir.mkdir(parents=True, exist_ok=True)
        report_file = report_dir / f"{datetime.now().strftime('%Y-%m-%d')}.md"

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
        report_md = (
            f"## Memory Health Report — {agent_id} | {now_str}\n\n"
            + "\n\n".join(sections)
        )

        if not (report.lint_contradictions or report.lint_orphans
                or report.lint_missing or report.lint_stale_pages):
            report_md += "\n\n*All clear — 未发现问题。*"

        report_file.write_text(report_md, encoding="utf-8")
        report.lint_report_path = str(report_file)
        logger.info(f"[lint] 报告已写入 {report_file}")


# ─── 主入口 ────────────────────────────────────────────────────────────────────

async def run_dream(
    agent_id: str = "aoxia",
    dry_run: bool = False,
) -> DreamReport:
    """
    运行完整 Dream 巩固流程

    Args:
        agent_id: Agent namespace
        dry_run: True 时只扫描不执行清理（用于调试）

    Returns:
        DreamReport — 运行报告
    """
    report = DreamReport(agent_id=agent_id)
    consolidator = DreamConsolidator()

    logger.info(f"=== Dream 开始 | agent={agent_id} dry_run={dry_run} ===")

    try:
        # Phase 1: Orient
        logger.info("[phase1] Orient — 扫描图谱")
        orient_result = await consolidator.phase1_orient(agent_id, report)

        # Phase 2: Gather
        logger.info("[phase2] Gather — 聚合候选问题")
        tasks = await consolidator.phase2_gather(agent_id, orient_result)

        # Phase 3: Consolidate
        logger.info("[phase3] Consolidate — LLM 决策")
        actions = await consolidator.phase3_consolidate(tasks, agent_id, report)

        # Phase 4: Prune
        if dry_run:
            logger.info(f"[phase4] Prune — dry_run=True，跳过执行，计划操作数={len(actions)}")
        else:
            logger.info("[phase4] Prune — 执行图谱清理")
            await consolidator.phase4_prune(actions, agent_id, report)

        # Phase 5: TTL 过期删除
        if not dry_run:
            logger.info("[phase5] TTL — 删除超期低置信度节点")
            await consolidator.phase5_ttl_expire(agent_id, report)

        # Phase 6: 置信度衰减与恢复
        if not dry_run:
            logger.info("[phase6] Confidence — 更新节点置信度评分")
            await consolidator.phase6_decay_confidence(agent_id, report)

        # Phase 7: Compile — 知识编译（碎片 facts → 结构化 CompiledPage）
        if not dry_run:
            logger.info("[phase7] Compile — 编译结构化知识页面")
            await consolidator.phase7_compile(agent_id, report)

        # Phase 8: Lint — 知识健康检查
        logger.info("[phase8] Lint — 健康检查")
        await consolidator.phase8_lint(agent_id, report)

        report.status = "done"
        report.finished_at = datetime.now(timezone.utc)
        logger.info(f"=== Dream 完成 === {report.summary()}")

    except Exception as e:
        report.status = "failed"
        report.error = str(e)
        report.finished_at = datetime.now(timezone.utc)
        logger.error(f"=== Dream 失败: {e} ===")

    finally:
        await consolidator.close()

    return report


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="MemOS Dream 记忆巩固")
    parser.add_argument("--agent-id", default="aoxia", help="Agent namespace")
    parser.add_argument("--dry-run", action="store_true", help="只扫描，不执行清理")
    args = parser.parse_args()

    report = asyncio.run(run_dream(agent_id=args.agent_id, dry_run=args.dry_run))
    print(report.summary())


if __name__ == "__main__":
    main()
