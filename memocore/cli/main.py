"""
memocore CLI — 记忆管理命令行工具

命令：
  memocore init                         交互式初始化配置（API key、Neo4j 等）
  memocore list   [--agent ID] [-n N]   列出最近记忆
  memocore search QUERY [--agent ID]    语义搜索记忆
  memocore delete UUID [--force]        删除指定节点
  memocore stats  [--agent ID]          图谱统计
  memocore export [--agent ID] [--format json|md] [-o FILE]  导出记忆
  memocore import FILE [--agent ID]     导入记忆
  memocore privacy-scan TEXT            预览隐私过滤结果

用法示例：
  memocore init
  memocore list -n 20
  memocore search "飞书通知规则"
  memocore export --format md -o memories.md
  memocore import backup.json
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# 确保 memocore 可 import
_project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_project_root))

try:
    import click
except ImportError:
    print("CLI 依赖未安装，请运行：pip install click", file=sys.stderr)
    sys.exit(1)

from dotenv import load_dotenv
_env = _project_root / ".env"
if _env.exists():
    load_dotenv(_env)

from memocore.core.config import (
    get_agent_id, get_global_config_path, get_state_dir,
    get_neo4j_config, write_global_config,
)


# ── 辅助函数 ────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


async def _get_neo4j_driver():
    from neo4j import AsyncGraphDatabase
    cfg = get_neo4j_config()
    return AsyncGraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


# ── CLI 主组 ────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option("0.1.0", prog_name="memocore")
def cli():
    """Memocore — AI Agent 持久化记忆管理工具"""
    pass


# ── init ────────────────────────────────────────────────────────────────────────

@cli.command()
def init():
    """交互式初始化 Memocore 配置（写入 ~/.memocore/config.env）"""
    click.echo("\nMemocore 初始化向导")
    click.echo("=" * 40)
    click.echo(f"配置文件将写入: {get_global_config_path()}\n")

    values = {}

    # Agent ID
    agent_id = click.prompt("Agent ID（你的命名空间）", default=get_agent_id())
    values["MEMOCORE_AGENT_ID"] = agent_id

    # LLM Provider
    click.echo("\n--- LLM 配置 ---")
    provider = click.prompt(
        "LLM Provider",
        type=click.Choice(["anthropic", "openai", "auto"], case_sensitive=False),
        default="auto",
    )
    values["MEMOCORE_LLM_PROVIDER"] = provider

    if provider in ("anthropic", "auto"):
        key = click.prompt("Anthropic API Key（留空跳过）", default="", hide_input=True)
        if key:
            values["ANTHROPIC_API_KEY"] = key

    if provider in ("openai", "auto"):
        key = click.prompt("OpenAI API Key（留空跳过）", default="", hide_input=True)
        if key:
            values["OPENAI_API_KEY"] = key

    # Embedding
    click.echo("\n--- Embedding 配置 ---")
    embed = click.prompt(
        "Embedding Provider（local 无需 API key）",
        type=click.Choice(["auto", "openai", "local"], case_sensitive=False),
        default="auto",
    )
    values["MEMOCORE_EMBED_PROVIDER"] = embed

    # Neo4j
    click.echo("\n--- Neo4j 配置 ---")
    cfg = get_neo4j_config()
    neo4j_uri = click.prompt("Neo4j URI", default=cfg["uri"])
    neo4j_user = click.prompt("Neo4j 用户名", default=cfg["user"])
    neo4j_password = click.prompt("Neo4j 密码", default=cfg["password"], hide_input=True)
    values["NEO4J_URI"] = neo4j_uri
    values["NEO4J_USER"] = neo4j_user
    values["NEO4J_PASSWORD"] = neo4j_password

    # 高级配置
    click.echo("\n--- 高级配置（回车使用默认值）---")
    dream_interval = click.prompt("Dream 触发间隔（会话次数）", default="5")
    values["MEMOCORE_DREAM_INTERVAL"] = dream_interval

    ttl_days = click.prompt("记忆 TTL（天，超期且低置信度节点将被清理）", default="90")
    values["MEMOCORE_DREAM_TTL_DAYS"] = ttl_days

    privacy = click.confirm("启用隐私过滤（自动 redact API key 等敏感信息）", default=True)
    values["MEMOCORE_PRIVACY_ENABLED"] = "true" if privacy else "false"

    # 写入
    config_path = write_global_config(values)
    click.echo(f"\n配置已写入: {config_path}")
    click.echo("\n使用 `memocore stats` 验证连接。")


# ── list ────────────────────────────────────────────────────────────────────────

@cli.command(name="list")
@click.option("--agent", default=None, help="Agent ID（默认读取 MEMOCORE_AGENT_ID）")
@click.option("-n", "--limit", default=20, show_default=True, help="返回条数")
@click.option("--type", "entity_type", default=None, help="实体类型过滤（如 Judgment）")
def list_memories(agent, limit, entity_type):
    """列出最近写入的记忆节点"""
    agent_id = agent or get_agent_id()

    async def _run_list():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                type_filter = f"AND n.entity_type = '{entity_type}'" if entity_type else ""
                q = f"""
                MATCH (n {{group_id: $gid}})
                WHERE n.name IS NOT NULL {type_filter}
                RETURN n.uuid AS uuid, n.name AS name,
                       n.summary AS summary,
                       n.entity_type AS type,
                       n.created_at AS created_at,
                       n.memocore_confidence AS confidence,
                       n.memocore_status AS status
                ORDER BY n.created_at DESC
                LIMIT $limit
                """
                result = await session.run(q, gid=agent_id, limit=limit)
                records = await result.data()
                return records
        finally:
            await driver.close()

    records = _run(_run_list())

    if not records:
        click.echo(f"[{agent_id}] 暂无记忆（图谱为空或 agent_id 不匹配）")
        return

    click.echo(f"\n[{agent_id}] 最近 {len(records)} 条记忆：\n")
    for r in records:
        conf = f"{r['confidence']:.1f}" if r.get('confidence') else "—"
        status = r.get('status') or 'confirmed'
        created = (r.get('created_at') or '')[:10]
        click.echo(f"  [{status:9s}|conf={conf}] {created}  {r.get('type', '?'):18s}  {r['name']}")
        if r.get('summary'):
            click.echo(f"    {r['summary'][:80]}")


# ── search ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("query")
@click.option("--agent", default=None, help="Agent ID")
@click.option("-k", "--top-k", default=10, show_default=True, help="返回条数")
@click.option("--no-rerank", is_flag=True, help="跳过 LLM rerank（更快）")
def search(query, agent, top_k, no_rerank):
    """语义搜索记忆图谱"""
    agent_id = agent or get_agent_id()

    async def _run_search():
        from memocore.core.retriever import MemoryRetriever
        retriever = MemoryRetriever()
        try:
            result = await retriever.retrieve(
                query=query,
                agent_id=agent_id,
                top_k=top_k,
                use_rerank=not no_rerank,
            )
            return result
        finally:
            await retriever.close()

    result = _run(_run_search())
    if result and result.strip():
        click.echo(result)
    else:
        click.echo(f"未找到与「{query}」相关的记忆")


# ── delete ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("uuid")
@click.option("--force", is_flag=True, help="跳过确认直接删除")
@click.option("--agent", default=None, help="Agent ID（用于验证所属）")
def delete(uuid, force, agent):
    """删除指定 UUID 的记忆节点"""
    agent_id = agent or get_agent_id()

    async def _preview():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                r = await session.run(
                    "MATCH (n {uuid: $uuid, group_id: $gid}) RETURN n.name AS name, n.entity_type AS type",
                    uuid=uuid, gid=agent_id
                )
                return await r.single()
        finally:
            await driver.close()

    rec = _run(_preview())
    if not rec:
        click.echo(f"未找到节点 {uuid}（可能已删除，或 agent_id 不匹配）")
        return

    click.echo(f"目标节点: [{rec['type']}] {rec['name']} ({uuid})")

    if not force:
        if not click.confirm("确认删除？此操作不可恢复", default=False):
            click.echo("已取消")
            return

    async def _do_delete():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                await session.run(
                    "MATCH (n {uuid: $uuid, group_id: $gid}) DETACH DELETE n",
                    uuid=uuid, gid=agent_id
                )
        finally:
            await driver.close()

    _run(_do_delete())
    click.echo(f"已删除: {uuid}")


# ── stats ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--agent", default=None, help="Agent ID（None 则显示全部）")
def stats(agent):
    """显示记忆图谱统计信息"""
    agent_id = agent or get_agent_id()

    async def _run_stats():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                node_q = "MATCH (n {group_id: $gid}) RETURN count(n) AS cnt"
                edge_q = "MATCH ()-[e {group_id: $gid}]->() RETURN count(e) AS cnt"
                type_q = """
                    MATCH (n {group_id: $gid}) WHERE n.entity_type IS NOT NULL
                    RETURN n.entity_type AS type, count(n) AS cnt ORDER BY cnt DESC
                """
                conf_q = """
                    MATCH (n {group_id: $gid})
                    RETURN n.memocore_status AS status, count(n) AS cnt
                """
                recent_q = """
                    MATCH (n {group_id: $gid}) WHERE n.created_at IS NOT NULL
                    RETURN max(n.created_at) AS latest
                """
                nodes = (await (await session.run(node_q, gid=agent_id)).single() or {}).get("cnt", 0)
                edges = (await (await session.run(edge_q, gid=agent_id)).single() or {}).get("cnt", 0)
                types = await (await session.run(type_q, gid=agent_id)).data()
                confs = await (await session.run(conf_q, gid=agent_id)).data()
                latest = (await (await session.run(recent_q, gid=agent_id)).single() or {}).get("latest", "—")
                return nodes, edges, types, confs, latest
        finally:
            await driver.close()

    nodes, edges, types, confs, latest = _run(_run_stats())
    latest_str = str(latest)[:10] if latest else "—"

    click.echo(f"\nMemocore 图谱统计 — agent: {agent_id}")
    click.echo("=" * 40)
    click.echo(f"  节点总数:     {nodes}")
    click.echo(f"  关系总数:     {edges}")
    click.echo(f"  最近更新:     {latest_str}")

    if types:
        click.echo("\n  实体类型分布:")
        for t in types:
            click.echo(f"    {t['type']:20s} {t['cnt']:>5}")

    if any(r.get('status') for r in confs):
        click.echo("\n  置信度分布:")
        for c in confs:
            s = c.get('status') or 'confirmed'
            click.echo(f"    {s:12s} {c['cnt']:>5}")

    config_path = get_global_config_path()
    click.echo(f"\n  配置文件:     {config_path}")
    click.echo(f"  状态目录:     {get_state_dir()}")


# ── export ───────────────────────────────────────────────────────────────────────

@cli.command(name="export")
@click.option("--agent", default=None, help="Agent ID")
@click.option("--format", "fmt", type=click.Choice(["json", "md"]), default="json", show_default=True)
@click.option("-o", "--output", default=None, help="输出文件路径（默认 stdout）")
@click.option("--limit", default=1000, show_default=True, help="最大导出节点数")
def export_memories(agent, fmt, output, limit):
    """导出记忆为 JSON 或 Markdown 格式"""
    agent_id = agent or get_agent_id()

    async def _run_export():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                # 导出节点
                node_q = """
                MATCH (n {group_id: $gid})
                WHERE n.name IS NOT NULL
                RETURN n.uuid AS uuid, n.name AS name, n.summary AS summary,
                       n.entity_type AS type, n.created_at AS created_at,
                       n.memocore_confidence AS confidence,
                       n.memocore_status AS status
                ORDER BY n.created_at DESC
                LIMIT $limit
                """
                nodes = await (await session.run(node_q, gid=agent_id, limit=limit)).data()

                # 导出 edges
                edge_q = """
                MATCH (a {group_id: $gid})-[e]->(b {group_id: $gid})
                WHERE e.fact IS NOT NULL
                RETURN e.uuid AS uuid, a.name AS src, b.name AS tgt,
                       e.fact AS fact, e.created_at AS created_at
                ORDER BY e.created_at DESC
                LIMIT $limit
                """
                edges = await (await session.run(edge_q, gid=agent_id, limit=limit)).data()
                return nodes, edges
        finally:
            await driver.close()

    nodes, edges = _run(_run_export())

    if fmt == "json":
        data = {
            "memocore_export": {
                "agent_id": agent_id,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "nodes": nodes,
                "edges": edges,
            }
        }
        content = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    else:
        # Markdown 格式
        lines = [
            f"# Memocore 记忆导出",
            f"",
            f"- **Agent**: {agent_id}",
            f"- **导出时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"- **节点数**: {len(nodes)}  **关系数**: {len(edges)}",
            f"",
            f"---",
            f"",
            f"## 记忆节点",
            f"",
        ]
        for n in nodes:
            conf = f"{n['confidence']:.1f}" if n.get('confidence') else "1.0"
            lines.append(f"### [{n.get('type', '?')}] {n['name']}")
            lines.append(f"*conf={conf} | {str(n.get('created_at', ''))[:10]}*")
            if n.get('summary'):
                lines.append(f"")
                lines.append(n['summary'])
            lines.append(f"")

        if edges:
            lines += ["---", "", "## 关系（Facts）", ""]
            for e in edges:
                lines.append(f"- **{e['src']}** → **{e['tgt']}**: {e['fact']}")

        content = "\n".join(lines)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"已导出 {len(nodes)} 节点 + {len(edges)} 关系 到 {output}")
    else:
        click.echo(content)


# ── import ───────────────────────────────────────────────────────────────────────

@cli.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--agent", default=None, help="导入到的 Agent ID（默认用文件内的 agent_id）")
@click.option("--dry-run", is_flag=True, help="只预览，不写入")
def import_memories(file, agent, dry_run):
    """从 JSON 导出文件恢复记忆"""
    content = Path(file).read_text(encoding="utf-8")
    try:
        data = json.loads(content)
        export_data = data.get("memocore_export", data)
    except json.JSONDecodeError as e:
        click.echo(f"无效的 JSON 文件: {e}", err=True)
        sys.exit(1)

    source_agent = export_data.get("agent_id", "unknown")
    target_agent = agent or source_agent
    nodes = export_data.get("nodes", [])
    edges = export_data.get("edges", [])

    click.echo(f"导入来源: agent={source_agent} | 节点={len(nodes)} 关系={len(edges)}")
    click.echo(f"导入目标: agent={target_agent}")

    if dry_run:
        click.echo("[dry-run] 不执行实际写入")
        return

    if not click.confirm(f"确认导入 {len(nodes)} 个节点到 [{target_agent}]？", default=True):
        click.echo("已取消")
        return

    async def _do_import():
        driver = await _get_neo4j_driver()
        imported = 0
        skipped = 0
        try:
            async with driver.session() as session:
                for n in nodes:
                    if not n.get("uuid") or not n.get("name"):
                        skipped += 1
                        continue
                    q = """
                    MERGE (n {uuid: $uuid})
                    SET n.name = $name,
                        n.summary = $summary,
                        n.entity_type = $type,
                        n.group_id = $gid,
                        n.created_at = $created_at,
                        n.memocore_confidence = $confidence,
                        n.memocore_status = $status
                    """
                    await session.run(q,
                        uuid=n["uuid"], name=n["name"],
                        summary=n.get("summary", ""),
                        type=n.get("type", ""),
                        gid=target_agent,
                        created_at=n.get("created_at", ""),
                        confidence=n.get("confidence", 1.0),
                        status=n.get("status", "confirmed"),
                    )
                    imported += 1
        finally:
            await driver.close()
        return imported, skipped

    imported, skipped = _run(_do_import())
    click.echo(f"导入完成: 成功 {imported} 个，跳过 {skipped} 个")


# ── browse ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--agent", default=None, help="Agent ID")
@click.option("--entity", default=None, help="查看指定实体的编译知识")
@click.option("--report", is_flag=True, help="显示最新 Lint 健康报告")
def browse(agent, entity, report):
    """浏览编译后的知识页面和 Lint 报告"""
    agent_id = agent or get_agent_id()

    if report:
        # 显示最新的 Lint 报告
        report_dir = Path.home() / ".memocore" / "reports" / agent_id
        if not report_dir.exists():
            click.echo(f"[{agent_id}] 尚无 Lint 报告（需先运行 Dream）")
            return
        reports = sorted(report_dir.glob("*.md"), reverse=True)
        if not reports:
            click.echo(f"[{agent_id}] 尚无 Lint 报告")
            return
        click.echo(reports[0].read_text(encoding="utf-8"))
        return

    async def _browse():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                if entity:
                    # 查看特定实体的编译页面
                    q = """
                    MATCH (p:CompiledPage {group_id: $gid, title: $title})
                    RETURN p.content AS content, p.confidence AS confidence,
                           p.source_count AS source_count, p.compiled_at AS compiled_at
                    """
                    r = await session.run(q, gid=agent_id, title=entity)
                    rec = await r.single()
                    if rec:
                        return "single", rec
                    return "not_found", entity

                # 列出所有编译页面
                q = """
                MATCH (p:CompiledPage {group_id: $gid})
                WHERE p.page_type = 'entity'
                RETURN p.title AS title, p.confidence AS confidence,
                       p.source_count AS source_count,
                       p.compiled_at AS compiled_at
                ORDER BY p.source_count DESC
                """
                r = await session.run(q, gid=agent_id)
                pages = await r.data()

                # 获取 overview
                ov_q = """
                MATCH (p:CompiledPage {group_id: $gid, title: '__overview__'})
                RETURN p.content AS content
                """
                r = await session.run(ov_q, gid=agent_id)
                ov = await r.single()
                overview = ov["content"] if ov else None

                return "list", (pages, overview)
        finally:
            await driver.close()

    mode, data = _run(_browse())

    if mode == "not_found":
        click.echo(f"未找到实体「{data}」的编译知识")
        click.echo("使用 `memocore browse` 查看所有已编译页面")
        return

    if mode == "single":
        conf = f"{data['confidence']:.1f}" if data.get('confidence') else "—"
        click.echo(f"\n[conf={conf} | facts={data.get('source_count', 0)} | "
                   f"compiled={str(data.get('compiled_at', ''))[:10]}]\n")
        click.echo(data.get("content", "(empty)"))
        return

    # mode == "list"
    pages, overview = data

    if not pages:
        click.echo(f"[{agent_id}] 尚无编译知识（需先运行 Dream）")
        click.echo("  运行: python -m memocore.core.dream --agent-id " + agent_id)
        return

    click.echo(f"\nMemocore 编译知识 — agent: {agent_id}")
    click.echo("=" * 50)

    if overview:
        click.echo(f"\n{overview}\n")
        click.echo("-" * 50)

    click.echo(f"\n已编译实体页 ({len(pages)}):\n")
    for p in pages:
        conf = f"{p['confidence']:.1f}" if p.get('confidence') else "—"
        compiled = str(p.get('compiled_at', ''))[:10]
        click.echo(f"  {p['title']:20s}  conf={conf}  facts={p.get('source_count', 0):>3}  compiled={compiled}")

    click.echo(f"\n查看详情: memocore browse --entity <实体名>")
    click.echo(f"健康报告: memocore browse --report")


# ── privacy-scan ─────────────────────────────────────────────────────────────────

@cli.command(name="privacy-scan")
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="从文件读取文本")
def privacy_scan(text, file):
    """预览隐私过滤效果（不写入任何数据）"""
    if file:
        text = Path(file).read_text(encoding="utf-8")
    elif not text:
        click.echo("请提供文本参数或 --file", err=True)
        sys.exit(1)

    from memocore.core.privacy import PrivacyFilter
    f = PrivacyFilter()
    cleaned, report = f.process(text)

    click.echo(f"\n过滤报告: {report}")
    click.echo("\n--- 处理后文本 ---")
    click.echo(cleaned)


# ── 入口 ─────────────────────────────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
