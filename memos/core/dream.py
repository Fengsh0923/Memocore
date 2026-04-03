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
    from memos.core.dream import run_dream
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
            f"重复组={self.duplicate_groups} 矛盾对={self.conflict_pairs} 过期={self.stale_nodes} | "
            f"合并={self.merged} 更新={self.updated} 剪枝={self.pruned} 保留={self.kept}"
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
        openai_api_key: Optional[str] = None,
        stale_days: int = 30,
        max_nodes_per_run: int = 200,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.stale_days = stale_days
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

        import openai
        client = openai.AsyncOpenAI(api_key=self.openai_api_key)

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

仅返回 JSON：{{"action": "merge"|"skip", "keep_uuid": "...", "reason": "..."}}"""

                elif task["type"] == "conflict":
                    prompt = f"""你是记忆图谱清洁工。以下是两条可能矛盾的关系：

{task['context']}

请决定：
1. keep_latest — 保留时间更新的那条（通常是正确的），删除旧的
2. keep_both — 两条含义不同，都保留
3. skip — 数据不足

仅返回 JSON：{{"action": "keep_latest"|"keep_both"|"skip", "delete_uuid": "...", "reason": "..."}}"""

                else:
                    continue

                response = await client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0,
                    max_tokens=200,
                    response_format={"type": "json_object"},
                )

                decision = json.loads(response.choices[0].message.content)
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
