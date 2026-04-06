"""
memocore CLI — memory management command-line tool

Commands:
  memocore init                         Interactive configuration setup (API key, Neo4j, etc.)
  memocore list   [--agent ID] [-n N]   List recent memories
  memocore search QUERY [--agent ID]    Semantic memory search
  memocore delete UUID [--force]        Delete a specific node
  memocore stats  [--agent ID]          Graph statistics
  memocore export [--agent ID] [--format json|md] [-o FILE]  Export memories
  memocore import FILE [--agent ID]     Import memories
  memocore privacy-scan TEXT            Preview privacy filtering results

Usage examples:
  memocore init
  memocore list -n 20
  memocore search "notification rules"
  memocore export --format md -o memories.md
  memocore import backup.json
"""

import asyncio
import json
import os
import sys
import uuid as _uuid
from datetime import datetime, timezone
from pathlib import Path

try:
    import click
except ImportError:
    print("CLI dependencies not installed, please run: pip install click", file=sys.stderr)
    sys.exit(1)

from memocore.core.config import (
    get_agent_id, get_global_config_path, get_state_dir,
    get_neo4j_config, write_global_config,
)


# ── Helper functions ────────────────────────────────────────────────────────────────────

def _run(coro):
    return asyncio.run(coro)


async def _get_neo4j_driver():
    from neo4j import AsyncGraphDatabase
    cfg = get_neo4j_config()
    return AsyncGraphDatabase.driver(cfg["uri"], auth=(cfg["user"], cfg["password"]))


# ── CLI main group ────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option("1.0.0", prog_name="memocore")
def cli():
    """Memocore — AI Agent persistent memory management tool"""
    pass


# ── init ────────────────────────────────────────────────────────────────────────

@cli.command()
def init():
    """Interactive Memocore configuration setup (writes to ~/.memocore/config.env)"""
    click.echo("\nMemocore Setup Wizard")
    click.echo("=" * 40)
    click.echo(f"Config file will be written to: {get_global_config_path()}\n")

    values = {}

    # Agent ID
    agent_id = click.prompt("Agent ID (your namespace)", default=get_agent_id())
    values["MEMOCORE_AGENT_ID"] = agent_id

    # LLM Provider
    click.echo("\n--- LLM Configuration ---")
    provider = click.prompt(
        "LLM Provider",
        type=click.Choice(["anthropic", "openai", "auto"], case_sensitive=False),
        default="auto",
    )
    values["MEMOCORE_LLM_PROVIDER"] = provider

    if provider in ("anthropic", "auto"):
        key = click.prompt("Anthropic API Key (leave blank to skip)", default="", hide_input=True)
        if key:
            values["ANTHROPIC_API_KEY"] = key

    if provider in ("openai", "auto"):
        key = click.prompt("OpenAI API Key (leave blank to skip)", default="", hide_input=True)
        if key:
            values["OPENAI_API_KEY"] = key

    # Embedding
    click.echo("\n--- Embedding Configuration ---")
    embed = click.prompt(
        "Embedding Provider (local requires no API key)",
        type=click.Choice(["auto", "openai", "local"], case_sensitive=False),
        default="auto",
    )
    values["MEMOCORE_EMBED_PROVIDER"] = embed

    # Neo4j
    click.echo("\n--- Neo4j Configuration ---")
    cfg = get_neo4j_config()
    neo4j_uri = click.prompt("Neo4j URI", default=cfg["uri"])
    neo4j_user = click.prompt("Neo4j username", default=cfg["user"])
    neo4j_password = click.prompt("Neo4j password", default=cfg["password"], hide_input=True)
    values["NEO4J_URI"] = neo4j_uri
    values["NEO4J_USER"] = neo4j_user
    values["NEO4J_PASSWORD"] = neo4j_password

    # Advanced configuration
    click.echo("\n--- Advanced Configuration (press Enter to use defaults) ---")
    dream_interval = click.prompt("Dream trigger interval (number of sessions)", default="5")
    values["MEMOCORE_DREAM_INTERVAL"] = dream_interval

    ttl_days = click.prompt("Memory TTL (days; expired low-confidence nodes will be pruned)", default="90")
    values["MEMOCORE_DREAM_TTL_DAYS"] = ttl_days

    privacy = click.confirm("Enable privacy filtering (auto-redact API keys and other sensitive info)", default=True)
    values["MEMOCORE_PRIVACY_ENABLED"] = "true" if privacy else "false"

    # Write config
    config_path = write_global_config(values)
    click.echo(f"\nConfig written to: {config_path}")
    click.echo("\nRun `memocore stats` to verify the connection.")


# ── list ────────────────────────────────────────────────────────────────────────

@cli.command(name="list")
@click.option("--agent", default=None, help="Agent ID (defaults to MEMOCORE_AGENT_ID)")
@click.option("-n", "--limit", default=20, show_default=True, help="Number of results to return")
@click.option("--type", "entity_type", default=None, help="Filter by entity type (e.g. Judgment)")
def list_memories(agent, limit, entity_type):
    """List recently written memory nodes"""
    agent_id = agent or get_agent_id()

    async def _run_list():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                type_filter = "AND n.entity_type = $etype" if entity_type else ""
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
                params = {"gid": agent_id, "limit": limit}
                if entity_type:
                    params["etype"] = entity_type
                result = await session.run(q, **params)
                records = await result.data()
                return records
        finally:
            await driver.close()

    records = _run(_run_list())

    if not records:
        click.echo(f"[{agent_id}] No memories found (graph is empty or agent_id does not match)")
        return

    click.echo(f"\n[{agent_id}] Most recent {len(records)} memories:\n")
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
@click.option("-k", "--top-k", default=10, show_default=True, help="Number of results to return")
@click.option("--no-rerank", is_flag=True, help="Skip LLM rerank (faster)")
def search(query, agent, top_k, no_rerank):
    """Semantic search of the memory graph"""
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
        click.echo(f"No memories found related to \"{query}\"")


# ── delete ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("uuid")
@click.option("--force", is_flag=True, help="Delete without confirmation prompt")
@click.option("--agent", default=None, help="Agent ID (used to verify ownership)")
def delete(uuid, force, agent):
    """Delete the memory node with the specified UUID"""
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
        click.echo(f"Node {uuid} not found (may already be deleted, or agent_id does not match)")
        return

    click.echo(f"Target node: [{rec['type']}] {rec['name']} ({uuid})")

    if not force:
        if not click.confirm("Confirm deletion? This action cannot be undone", default=False):
            click.echo("Cancelled")
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
    click.echo(f"Deleted: {uuid}")


# ── stats ────────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--agent", default=None, help="Agent ID (None shows all)")
def stats(agent):
    """Display memory graph statistics"""
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

    click.echo(f"\nMemocore Graph Statistics — agent: {agent_id}")
    click.echo("=" * 40)
    click.echo(f"  Total nodes:          {nodes}")
    click.echo(f"  Total relationships:  {edges}")
    click.echo(f"  Last updated:         {latest_str}")

    if types:
        click.echo("\n  Entity type distribution:")
        for t in types:
            click.echo(f"    {t['type']:20s} {t['cnt']:>5}")

    if any(r.get('status') for r in confs):
        click.echo("\n  Confidence distribution:")
        for c in confs:
            s = c.get('status') or 'confirmed'
            click.echo(f"    {s:12s} {c['cnt']:>5}")

    config_path = get_global_config_path()
    click.echo(f"\n  Config file:   {config_path}")
    click.echo(f"  State dir:     {get_state_dir()}")


# ── export ───────────────────────────────────────────────────────────────────────

@cli.command(name="export")
@click.option("--agent", default=None, help="Agent ID")
@click.option("--format", "fmt", type=click.Choice(["json", "md"]), default="json", show_default=True)
@click.option("-o", "--output", default=None, help="Output file path (defaults to stdout)")
@click.option("--limit", default=1000, show_default=True, help="Maximum number of nodes to export")
def export_memories(agent, fmt, output, limit):
    """Export memories as JSON or Markdown format"""
    agent_id = agent or get_agent_id()

    async def _run_export():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                # Export nodes
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

                # Export edges
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
        # Markdown format
        lines = [
            f"# Memocore Memory Export",
            f"",
            f"- **Agent**: {agent_id}",
            f"- **Export time**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"- **Nodes**: {len(nodes)}  **Relationships**: {len(edges)}",
            f"",
            f"---",
            f"",
            f"## Memory Nodes",
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
            lines += ["---", "", "## Relationships (Facts)", ""]
            for e in edges:
                lines.append(f"- **{e['src']}** → **{e['tgt']}**: {e['fact']}")

        content = "\n".join(lines)

    if output:
        Path(output).write_text(content, encoding="utf-8")
        click.echo(f"Exported {len(nodes)} nodes + {len(edges)} relationships to {output}")
    else:
        click.echo(content)


# ── import ───────────────────────────────────────────────────────────────────────

@cli.command(name="import")
@click.argument("file", type=click.Path(exists=True))
@click.option("--agent", default=None, help="Target Agent ID (defaults to agent_id from file)")
@click.option("--dry-run", is_flag=True, help="Preview only, do not write")
def import_memories(file, agent, dry_run):
    """Restore memories from JSON export file"""
    content = Path(file).read_text(encoding="utf-8")
    try:
        data = json.loads(content)
        export_data = data.get("memocore_export", data)
    except json.JSONDecodeError as e:
        click.echo(f"Invalid JSON file: {e}", err=True)
        sys.exit(1)

    source_agent = export_data.get("agent_id", "unknown")
    target_agent = agent or source_agent
    nodes = export_data.get("nodes", [])
    edges = export_data.get("edges", [])

    click.echo(f"Import source: agent={source_agent} | nodes={len(nodes)} relationships={len(edges)}")
    click.echo(f"Import target: agent={target_agent}")

    if dry_run:
        click.echo("[dry-run] No actual writes performed")
        return

    if not click.confirm(f"Confirm import of {len(nodes)} nodes to [{target_agent}]?", default=True):
        click.echo("Cancelled")
        return

    async def _do_import():
        driver = await _get_neo4j_driver()
        imported_nodes = 0
        imported_edges = 0
        skipped = 0
        try:
            async with driver.session() as session:
                # Import nodes
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
                    imported_nodes += 1

                # Import edges (matched by src/tgt name against existing nodes)
                for e in edges:
                    if not e.get("src") or not e.get("tgt") or not e.get("fact"):
                        skipped += 1
                        continue
                    eq = """
                    MATCH (a {group_id: $gid, name: $src})
                    MATCH (b {group_id: $gid, name: $tgt})
                    WITH a, b LIMIT 1
                    MERGE (a)-[r:RELATES_TO {uuid: $uuid}]->(b)
                    SET r.fact = $fact,
                        r.group_id = $gid,
                        r.created_at = $created_at
                    """
                    await session.run(eq,
                        gid=target_agent,
                        src=e["src"], tgt=e["tgt"],
                        uuid=e.get("uuid") or str(_uuid.uuid4()),
                        fact=e["fact"],
                        created_at=e.get("created_at", ""),
                    )
                    imported_edges += 1
        finally:
            await driver.close()
        return imported_nodes, imported_edges, skipped

    imported_nodes, imported_edges, skipped = _run(_do_import())
    click.echo(f"Import complete: {imported_nodes} nodes, {imported_edges} relationships, {skipped} skipped")


# ── browse ───────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--agent", default=None, help="Agent ID")
@click.option("--entity", default=None, help="View compiled knowledge for a specific entity")
@click.option("--report", is_flag=True, help="Show latest Lint health report")
def browse(agent, entity, report):
    """Browse compiled knowledge pages and Lint reports"""
    agent_id = agent or get_agent_id()

    if report:
        # Display the latest Lint report
        report_dir = get_state_dir() / "reports" / agent_id
        if not report_dir.exists():
            click.echo(f"[{agent_id}] No Lint report yet (run Dream first)")
            return
        reports = sorted(report_dir.glob("*.md"), reverse=True)
        if not reports:
            click.echo(f"[{agent_id}] No Lint report yet")
            return
        click.echo(reports[0].read_text(encoding="utf-8"))
        return

    async def _browse():
        driver = await _get_neo4j_driver()
        try:
            async with driver.session() as session:
                if entity:
                    # View compiled page for specific entity
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

                # List all compiled pages
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

                # Get overview
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
        click.echo(f"No compiled knowledge found for entity \"{data}\"")
        click.echo("Use `memocore browse` to view all compiled pages")
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
        click.echo(f"[{agent_id}] No compiled knowledge yet (run Dream first)")
        click.echo("  Run: python -m memocore.core.dream --agent-id " + agent_id)
        return

    click.echo(f"\nMemocore Compiled Knowledge — agent: {agent_id}")
    click.echo("=" * 50)

    if overview:
        click.echo(f"\n{overview}\n")
        click.echo("-" * 50)

    click.echo(f"\nCompiled entity pages ({len(pages)}):\n")
    for p in pages:
        conf = f"{p['confidence']:.1f}" if p.get('confidence') else "—"
        compiled = str(p.get('compiled_at', ''))[:10]
        click.echo(f"  {p['title']:20s}  conf={conf}  facts={p.get('source_count', 0):>3}  compiled={compiled}")

    click.echo(f"\nView details: memocore browse --entity <entity name>")
    click.echo(f"Health report: memocore browse --report")


# ── privacy-scan ─────────────────────────────────────────────────────────────────

@cli.command(name="privacy-scan")
@click.argument("text", required=False)
@click.option("--file", "-f", type=click.Path(exists=True), help="Read text from file")
def privacy_scan(text, file):
    """Preview privacy filtering (no data written)"""
    if file:
        text = Path(file).read_text(encoding="utf-8")
    elif not text:
        click.echo("Please provide text argument or --file", err=True)
        sys.exit(1)

    from memocore.core.privacy import PrivacyFilter
    f = PrivacyFilter()
    cleaned, report = f.process(text)

    click.echo(f"\nFilter report: {report}")
    click.echo("\n--- Processed text ---")
    click.echo(cleaned)


# ── Entrypoint ─────────────────────────────────────────────────────────────────────────

def main():
    cli()


if __name__ == "__main__":
    main()
