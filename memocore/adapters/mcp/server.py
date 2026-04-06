"""
Memocore MCP Server

Exposes Memocore's memory write and recall capabilities as MCP Tools.
Supports two deployment modes:

━━ Mode 1: stdio (personal / local tool) ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compatible with MCP clients such as Claude Desktop, Cursor, and Windsurf.
Each client runs in its own process; agent_id is configured via environment variable.

Start:
  python -m memocore.adapters.mcp.server
  # or
  memocore-mcp

Claude Desktop config (~/.claude/claude_desktop_config.json):
  {
    "mcpServers": {
      "memocore": {
        "command": "memocore-mcp",
        "env": {
          "MEMOCORE_AGENT_ID": "personal",
          "ANTHROPIC_API_KEY": "sk-ant-..."
        }
      }
    }
  }

━━ Mode 2: HTTP (enterprise / multi-tenant service) ━━━━━━━━━━━━━━━━━━━━━━━━━
Runs as a shared microservice; multiple Clawith Agents share one instance.
agent_id is passed as a parameter on each tool call for multi-tenant isolation.

Start:
  memocore-mcp --transport http --host 0.0.0.0 --port 8765

Environment variables (HTTP mode):
  MEMOCORE_API_KEY=your_secret_key   # Enable API Key auth (recommended for production)

Clawith call example (via MCP HTTP Client):
  await mcp_client.call_tool("memory_recall", {
      "query": "the user's current question",
      "agent_id": "clawith-agent-uuid-xxx",  # unique per agent
  })

Docker deployment:
  docker run -d --name memocore-mcp \\
    -p 8765:8765 \\
    -e NEO4J_URI=bolt://neo4j:7687 \\
    -e NEO4J_USERNAME=neo4j \\
    -e NEO4J_PASSWORD=your_password \\
    -e ANTHROPIC_API_KEY=sk-ant-... \\
    -e MEMOCORE_API_KEY=your_secret_key \\
    memocore:latest memocore-mcp --transport http --host 0.0.0.0 --port 8765
"""

import hmac
import logging
import os
import re
import sys
from typing import Optional

from memocore.core.config import get_agent_id, get_sessions_dir, get_logs_dir, get_neo4j_config, validate_agent_id, validate_identifier, validate_scope_id, make_safe_agent_key

# ── Input size limits ────────────────────────────────────────────────────────
MAX_CONVERSATION_BYTES = 64 * 1024  # 64 KB
MAX_QUERY_BYTES = 2 * 1024          # 2 KB
MAX_SOURCE_LEN = 64                 # source description

logger = logging.getLogger("memocore.mcp")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "MCP dependency not installed, please run: pip install 'mcp[cli]'\n"
        "or install full dependencies: pip install 'memocore[mcp]'",
        file=sys.stderr,
    )
    sys.exit(1)

# ── Shared Neo4j connection pool (all Agents share one driver in HTTP mode) ───
_shared_neo4j_driver = None


def _get_shared_driver():
    """
    Get shared Neo4j driver (built-in pool, default 50 connections).
    All tool calls in HTTP mode share one driver to avoid connection storms.
    """
    global _shared_neo4j_driver
    if _shared_neo4j_driver is None:
        from neo4j import AsyncGraphDatabase
        cfg = get_neo4j_config()
        pool_size = int(os.environ.get("MEMOCORE_NEO4J_POOL_SIZE", "50"))
        _shared_neo4j_driver = AsyncGraphDatabase.driver(
            cfg["uri"],
            auth=(cfg["user"], cfg["password"]),
            max_connection_pool_size=pool_size,
        )
    return _shared_neo4j_driver


mcp = FastMCP(
    name="memocore",
    instructions=(
        "Memocore is a persistent memory layer for AI Agents. "
        "Call memory_session_start at the beginning of a session for historical context, "
        "call memory_recall before each user message to retrieve relevant memories, "
        "and call memory_store after a conversation to persist important information. "
        "In HTTP mode, all tool calls must include the agent_id parameter."
    ),
)


# ── Auth utility functions ─────────────────────────────────────────────────────

def _check_api_key(provided_key: Optional[str]) -> Optional[str]:
    """
    Check if API Key is valid.
    Auth only enabled when the MEMOCORE_API_KEY environment variable is set.
    Returns error string (auth failed), or None (auth passed).
    """
    expected = os.environ.get("MEMOCORE_API_KEY", "").strip()
    if not expected:
        return None  # No API Key configured, skip auth
    if not provided_key or not hmac.compare_digest(provided_key.strip(), expected):
        logger.warning("API Key authentication failed")
        return "Authentication failed: please provide a valid api_key parameter"
    return None


def _resolve_agent_id(agent_id: Optional[str]) -> str:
    """
    Resolve agent_id:
    - HTTP mode: passed by caller, per-tenant isolation
    - stdio mode: read from MEMOCORE_AGENT_ID env var
    Validates the resolved agent_id to prevent path traversal and injection.
    """
    resolved = agent_id.strip() if agent_id and agent_id.strip() else get_agent_id()
    return validate_agent_id(resolved)


def _sanitize_session_id(session_id: str) -> str:
    """Sanitize session_id: keep only safe characters, truncate to 32 chars."""
    return re.sub(r'[^a-zA-Z0-9_\-]', '', session_id)[:32]


def _sanitize_log(value: str, max_len: int = 50) -> str:
    """Strip control characters from user input before logging."""
    return value.replace('\n', '\\n').replace('\r', '\\r')[:max_len]


def _is_first_message(session_id: str, agent_id: str) -> bool:
    """Check if this is the first message in the session (atomic, isolated by agent_id)."""
    agent_key = make_safe_agent_key(agent_id)
    safe_session = _sanitize_session_id(session_id)
    flag = get_sessions_dir() / f"mcp-{agent_key}-{safe_session}.flag"
    try:
        flag.open('x').close()  # atomic exclusive create
        return True
    except FileExistsError:
        return False


# ── Multi-scope helpers ────────────────────────────────────────────────────────

def _scope_to_group_id(scope: str, agent_id: str, team_id: Optional[str], tenant_id: Optional[str]) -> str:
    """Convert scope + related IDs to Neo4j group_id."""
    if scope == "team" and team_id:
        validate_scope_id(team_id, "team_id")
        return f"team:{team_id}"
    if scope == "org" and tenant_id:
        validate_scope_id(tenant_id, "tenant_id")
        return f"org:{tenant_id}"
    return agent_id  # personal (default)


# ── Tool 1: memory_recall ──────────────────────────────────────────────────────

@mcp.tool()
async def memory_recall(
    query: str,
    session_id: str = "default",
    top_k: int = 5,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Recall the most relevant historical memories from the knowledge graph based on the current prompt.

    Supports merged retrieval across three memory scopes:
    - Personal memory (always retrieved)
    - Team shared memory (enabled when team_id is provided)
    - Organization knowledge base (enabled when tenant_id is provided)

    Call this tool before processing each user message. Injects returned memories into your context.
    On the first message of a session, automatically triggers full background recall.

    Args:
        query: The user's current message or keywords
        session_id: Session identifier for distinguishing first/subsequent messages
        top_k: Number of memories to return (default 5)
        agent_id: Agent unique identifier (required in HTTP mode; read from env in stdio mode)
        team_id: Team ID, enables merged retrieval of team shared memory
        tenant_id: Organization ID, enables merged retrieval of organization knowledge base
        api_key: API Key (required when MEMOCORE_API_KEY is configured on the server)

    Returns:
        Formatted historical memory text (multi-scope results annotated with [Personal]/[Team]/[Org] source)
    """
    if err := _check_api_key(api_key):
        return err

    if len(query) > MAX_QUERY_BYTES:
        return f"Query too large ({len(query)} bytes > {MAX_QUERY_BYTES})"

    try:
        resolved_agent_id = _resolve_agent_id(agent_id)
        if team_id:
            team_id = validate_scope_id(team_id, "team_id")
        if tenant_id:
            tenant_id = validate_scope_id(tenant_id, "tenant_id")
    except ValueError as e:
        logger.warning(f"memory_recall: invalid parameters: {e}")
        return f"Invalid parameters: {e}"

    safe_sid = _sanitize_session_id(session_id)
    logger.info(
        f"memory_recall: agent={resolved_agent_id[:24]} session={safe_sid[:16]} "
        f"team={team_id} tenant={tenant_id} query={_sanitize_log(query)}"
    )

    try:
        from memocore.core.retriever import MemoryRetriever
        retriever = MemoryRetriever()

        try:
            if _is_first_message(safe_sid, resolved_agent_id):
                logger.info("memory_recall: first message, triggering full recall")
                result = await retriever.retrieve_for_session_start(
                    agent_id=resolved_agent_id,
                    top_k=max(top_k * 2, 10),
                    team_id=team_id,
                    tenant_id=tenant_id,
                )
            else:
                result = await retriever.retrieve(
                    query=query,
                    agent_id=resolved_agent_id,
                    top_k=top_k,
                    use_rerank=True,
                    team_id=team_id,
                    tenant_id=tenant_id,
                )
        finally:
            await retriever.close()

        if not result or not result.strip():
            return ""

        logger.info(f"memory_recall: recalled {len(result)} chars")
        return result

    except Exception as e:
        logger.error(f"memory_recall failed: {e}")
        return ""


# ── Tool 2: memory_session_start ──────────────────────────────────────────────

@mcp.tool()
async def memory_session_start(
    session_id: str = "default",
    top_k: int = 15,
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Call at the start of a new session to retrieve a full historical background memory bundle.

    Includes: recent project state, user preference rules, past decisions, etc.
    Supports merged retrieval across personal, team, and organization memory scopes.
    Recommended: inject the returned content at the beginning of the system prompt.

    Args:
        session_id: Session identifier
        top_k: Number of memories to return (default 15; more context at session start)
        agent_id: Agent unique identifier (required in HTTP mode; read from env in stdio mode)
        team_id: Team ID; enables merged retrieval of team memory when provided
        tenant_id: Organization ID; enables merged retrieval of org knowledge when provided
        api_key: API Key (required when MEMOCORE_API_KEY is configured on the server)

    Returns:
        Formatted background memory text (multi-scope results annotated with source)
    """
    if err := _check_api_key(api_key):
        return err

    try:
        resolved_agent_id = _resolve_agent_id(agent_id)
    except ValueError as e:
        logger.warning(f"memory_session_start: invalid agent_id: {e}")
        return f"Invalid agent_id: {e}"

    safe_sid = _sanitize_session_id(session_id)
    logger.info(
        f"memory_session_start: agent={resolved_agent_id[:24]} session={safe_sid[:16]} "
        f"team={team_id} tenant={tenant_id}"
    )

    # Mark as initialized to prevent memory_recall from re-triggering full recall
    safe_agent = make_safe_agent_key(resolved_agent_id)
    flag = get_sessions_dir() / f"mcp-{safe_agent}-{safe_sid}.flag"
    try:
        flag.open('x').close()
    except FileExistsError:
        pass

    try:
        from memocore.core.retriever import MemoryRetriever
        retriever = MemoryRetriever()
        try:
            result = await retriever.retrieve_for_session_start(
                agent_id=resolved_agent_id,
                top_k=top_k,
                team_id=team_id,
                tenant_id=tenant_id,
            )
        finally:
            await retriever.close()

        logger.info(f"memory_session_start: recalled {len(result)} chars")
        return result or ""

    except Exception as e:
        logger.error(f"memory_session_start failed: {e}")
        return ""


# ── Tool 3: memory_store ──────────────────────────────────────────────────────

@mcp.tool()
async def memory_store(
    conversation: str,
    session_id: str = "default",
    source: str = "MCP client",
    scope: str = "personal",
    agent_id: Optional[str] = None,
    team_id: Optional[str] = None,
    tenant_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    Persist conversation content to the memory knowledge graph.

    Call at the end of a conversation or whenever important information needs to be remembered.
    The system automatically extracts structured entities (preferences, decisions, project state,
    etc.) from the conversation and writes them to Neo4j.

    Args:
        conversation: Conversation text to store (User/Assistant format)
        session_id: Session identifier (used for log tracing)
        source: Data source description (default "MCP client")
        scope: Memory scope
               "personal" — readable only by this Agent (default)
               "team"     — shared with the team (requires team_id)
               "org"      — organization knowledge base (requires tenant_id; typically written by admins)
        agent_id: Agent unique identifier (required in HTTP mode; read from env in stdio mode)
        team_id: Team ID (required when scope="team")
        tenant_id: Organization ID (required when scope="org")
        api_key: API Key (required when MEMOCORE_API_KEY is configured on the server)

    Returns:
        Storage result description (success or failure message)
    """
    if err := _check_api_key(api_key):
        return err

    try:
        resolved_agent_id = _resolve_agent_id(agent_id)
        group_id = _scope_to_group_id(scope, resolved_agent_id, team_id, tenant_id)
    except ValueError as e:
        logger.warning(f"memory_store: invalid parameters: {e}")
        return f"Invalid parameters: {e}"

    safe_sid = _sanitize_session_id(session_id)
    # Sanitize source: only safe characters, limited length
    safe_source = re.sub(r'[^a-zA-Z0-9 _\-.]', '', source)[:MAX_SOURCE_LEN]
    logger.info(
        f"memory_store: agent={resolved_agent_id[:24]} session={safe_sid[:16]} "
        f"scope={scope} group={group_id[:24]} len={len(conversation)}"
    )

    if not conversation or not conversation.strip():
        return "Skipped: conversation content is empty"

    if len(conversation) < 100:
        return "Skipped: conversation too short (< 100 characters)"

    if len(conversation) > MAX_CONVERSATION_BYTES:
        return f"Store failed: conversation too large ({len(conversation)} bytes > {MAX_CONVERSATION_BYTES})"

    try:
        from memocore.core.extractor import extract_and_store
        result = await extract_and_store(
            conversation=conversation,
            agent_id=resolved_agent_id,
            source_description=f"{safe_source} | scope={scope} | session={safe_sid[:16]}",
            group_id=group_id,
        )

        if result["success"]:
            msg = (
                f"Memory stored: episode={result['episode_name']} "
                f"entities={result.get('entities_extracted', '?')}"
            )
            logger.info(f"memory_store: {msg}")
            return msg
        else:
            err = f"Store failed: {result.get('error', 'unknown error')}"
            logger.error(f"memory_store: {err}")
            return err

    except Exception as e:
        logger.error(f"memory_store error: {e}")
        return f"Store failed: {e}"


# ── Tool 4: health_check ───────────────────────────────────────────────────────

@mcp.tool()
async def health_check(
    api_key: Optional[str] = None,
) -> dict:
    """
    Check the health status of the Memocore service.

    Returns service version, Neo4j connection status, and other information.
    Suitable for calling from the Clawith backend at startup to confirm the service is available.

    Args:
        api_key: API Key (required when MEMOCORE_API_KEY is configured on the server)

    Returns:
        Health status dictionary
    """
    if err := _check_api_key(api_key):
        return {"status": "error", "message": err}

    status = {
        "status": "ok",
        "service": "memocore-mcp",
        "transport": os.environ.get("_MEMOCORE_TRANSPORT", "stdio"),
        "agent_id_source": "parameter" if os.environ.get("_MEMOCORE_HTTP_MODE") else "env",
        "neo4j_connected": False,
    }

    try:
        driver = _get_shared_driver()
        await driver.verify_connectivity()
        status["neo4j_connected"] = True
    except Exception as e:
        status["status"] = "degraded"
        status["neo4j_error"] = "connection failed"
        logger.error(f"health_check neo4j error: {e}")  # full details server-side only

    return status


# ── Entrypoint ─────────────────────────────────────────────────────────────────

def main():
    """
    Start Memocore MCP Server.

    Transport modes:
      stdio  (default) — personal tool, for Claude Desktop / Cursor
      http             — enterprise service, multi-tenant shared instance

    Usage:
      memocore-mcp                               # stdio mode
      memocore-mcp --transport http              # HTTP mode, default 0.0.0.0:8765
      memocore-mcp --transport http --port 9000  # custom port
    """
    import click

    # Configure logging for the server process (not at import time)
    log_file = get_logs_dir() / "mcp_server.log"
    handler = logging.FileHandler(str(log_file))
    handler.setFormatter(logging.Formatter("%(asctime)s [mcp_server] %(levelname)s %(message)s"))
    logging.getLogger("memocore").addHandler(handler)
    logging.getLogger("memocore").setLevel(logging.INFO)

    @click.command()
    @click.option(
        "--transport",
        type=click.Choice(["stdio", "http", "sse"]),
        default="stdio",
        show_default=True,
        help="Transport: stdio (local tool) | http (enterprise service) | sse (legacy HTTP)",
    )
    @click.option("--host", default="0.0.0.0", show_default=True, help="HTTP listen address")
    @click.option("--port", default=8765, show_default=True, help="HTTP listen port")
    def _main(transport: str, host: str, port: int):
        agent_id = get_agent_id()
        api_key_enabled = bool(os.environ.get("MEMOCORE_API_KEY", "").strip())

        if transport == "stdio":
            logger.info(f"Memocore MCP Server started | transport=stdio | agent_id={agent_id}")
            mcp.run(transport="stdio")

        elif transport in ("http", "sse"):
            # Mark as HTTP mode so health_check and other tools can read it
            os.environ["_MEMOCORE_HTTP_MODE"] = "1"
            os.environ["_MEMOCORE_TRANSPORT"] = transport

            logger.info(
                f"Memocore MCP Server started | transport={transport} | "
                f"host={host} | port={port} | "
                f"api_key_auth={'enabled' if api_key_enabled else 'disabled'}"
            )

            if not api_key_enabled:
                allow_no_auth = os.environ.get("MEMOCORE_ALLOW_NO_AUTH", "").strip() == "1"
                if not allow_no_auth:
                    click.echo(
                        "[ERROR] HTTP mode requires MEMOCORE_API_KEY to protect the service.\n"
                        "        Set MEMOCORE_API_KEY env var, or set MEMOCORE_ALLOW_NO_AUTH=1 to skip (dev only).",
                        err=True,
                    )
                    sys.exit(1)
                click.echo(
                    "[WARNING] MEMOCORE_API_KEY not set, HTTP service has no authentication (MEMOCORE_ALLOW_NO_AUTH=1).\n"
                    "          For production, set MEMOCORE_API_KEY or add auth at the gateway layer.",
                    err=True,
                )

            click.echo(
                f"Memocore MCP Server running\n"
                f"  Address: http://{host}:{port}\n"
                f"  Auth: {'API Key (MEMOCORE_API_KEY)' if api_key_enabled else 'disabled (dev mode)'}\n"
                f"  Multi-tenant: pass agent_id in tool calls\n"
            )

            mcp_transport = "sse" if transport == "sse" else "streamable-http"
            mcp.run(transport=mcp_transport, host=host, port=port)

    _main()


if __name__ == "__main__":
    main()
