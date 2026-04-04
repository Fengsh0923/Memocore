"""
Memocore MCP Server

将 Memocore 的记忆写入和召回能力暴露为 MCP Tools，支持两种部署模式：

━━ 模式一：stdio（个人/本地工具）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
兼容 Claude Desktop、Cursor、Windsurf 等 MCP 客户端。
每个客户端独立进程，agent_id 通过环境变量配置。

启动：
  python -m memocore.adapters.mcp.server
  # 或
  memocore-mcp

Claude Desktop 配置 (~/.claude/claude_desktop_config.json)：
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

━━ 模式二：HTTP（企业/多租户服务）━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
作为共享微服务运行，多个 Clawith Agent 共用同一实例。
agent_id 在每次 tool 调用时作为参数传入，实现多租户隔离。

启动：
  memocore-mcp --transport http --host 0.0.0.0 --port 8765

环境变量（HTTP 模式）：
  MEMOCORE_API_KEY=your_secret_key   # 启用 API Key 鉴权（推荐生产环境开启）

Clawith 调用示例（通过 MCP HTTP Client）：
  await mcp_client.call_tool("memory_recall", {
      "query": "用户的当前问题",
      "agent_id": "clawith-agent-uuid-xxx",  # 每个 Agent 独立
  })

Docker 部署：
  docker run -d --name memocore-mcp \\
    -p 8765:8765 \\
    -e NEO4J_URI=bolt://neo4j:7687 \\
    -e NEO4J_USERNAME=neo4j \\
    -e NEO4J_PASSWORD=your_password \\
    -e ANTHROPIC_API_KEY=sk-ant-... \\
    -e MEMOCORE_API_KEY=your_secret_key \\
    memocore:latest memocore-mcp --transport http --host 0.0.0.0 --port 8765
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# 确保能 import memocore（从 project root 运行时）
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

# 加载配置（优先级：环境变量 > 项目 .env > 全局 ~/.memocore/config.env）
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from memocore.core.config import get_agent_id, get_sessions_dir, get_logs_dir

LOG_FILE = get_logs_dir() / "mcp_server.log"
logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [mcp_server] %(levelname)s %(message)s",
)
logger = logging.getLogger("memocore.mcp")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print(
        "MCP 依赖未安装，请运行：pip install 'mcp[cli]'\n"
        "或安装完整依赖：pip install 'memocore[mcp]'",
        file=sys.stderr,
    )
    sys.exit(1)

mcp = FastMCP(
    name="memocore",
    instructions=(
        "Memocore 是 AI Agent 的持久化记忆层。"
        "在对话开始时调用 memory_session_start 获取历史背景，"
        "每条用户消息前调用 memory_recall 检索相关记忆，"
        "对话结束后调用 memory_store 持久化重要信息。"
        "HTTP 模式下，所有工具调用必须传入 agent_id 参数。"
    ),
)


# ── 鉴权工具函数 ───────────────────────────────────────────────────────────────

def _check_api_key(provided_key: Optional[str]) -> Optional[str]:
    """
    检查 API Key 是否有效。
    仅在 MEMOCORE_API_KEY 环境变量已设置时启用鉴权。
    返回错误信息字符串（认证失败），或 None（认证通过）。
    """
    expected = os.environ.get("MEMOCORE_API_KEY", "").strip()
    if not expected:
        return None  # 未配置 API Key，跳过鉴权
    if not provided_key or provided_key.strip() != expected:
        logger.warning("API Key 鉴权失败")
        return "鉴权失败：请在调用时提供有效的 api_key 参数"
    return None


def _resolve_agent_id(agent_id: Optional[str]) -> str:
    """
    解析 agent_id：
    - HTTP 模式：调用方传入，每个租户独立
    - stdio 模式：从环境变量 MEMOCORE_AGENT_ID 读取
    """
    if agent_id and agent_id.strip():
        return agent_id.strip()
    return get_agent_id()


def _is_first_message(session_id: str, agent_id: str) -> bool:
    """检查是否是 session 的第一条消息（按 agent_id 隔离）"""
    safe_agent = agent_id.replace("/", "_").replace(":", "_")[:32]
    flag = get_sessions_dir() / f"mcp-{safe_agent}-{session_id[:16]}.flag"
    if flag.exists():
        return False
    flag.touch()
    return True


# ── Tool 1: memory_recall ──────────────────────────────────────────────────────

@mcp.tool()
async def memory_recall(
    query: str,
    session_id: str = "default",
    top_k: int = 5,
    agent_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    根据当前 prompt 从记忆图谱召回最相关的历史记忆。

    在处理每条用户消息之前调用此工具，将返回的记忆注入到你的上下文中。
    如果是 session 的第一条消息，会自动触发全量背景召回。

    Args:
        query: 用户的当前消息或关键词
        session_id: 会话标识符，用于区分首次/后续消息
        top_k: 返回的记忆条数（默认 5）
        agent_id: Agent 唯一标识（HTTP 模式必填；stdio 模式从环境变量读取）
        api_key: API Key（当服务端配置了 MEMOCORE_API_KEY 时必填）

    Returns:
        格式化的历史记忆文本，可直接注入系统提示
    """
    if err := _check_api_key(api_key):
        return err

    resolved_agent_id = _resolve_agent_id(agent_id)
    logger.info(f"memory_recall: agent={resolved_agent_id[:24]} session={session_id[:16]} query={query[:50]}")

    try:
        from memocore.core.retriever import MemoryRetriever
        retriever = MemoryRetriever()

        try:
            if _is_first_message(session_id, resolved_agent_id):
                logger.info("memory_recall: 首条消息，触发全量召回")
                result = await retriever.retrieve_for_session_start(
                    agent_id=resolved_agent_id,
                    top_k=max(top_k * 2, 10),
                )
            else:
                result = await retriever.retrieve(
                    query=query,
                    agent_id=resolved_agent_id,
                    top_k=top_k,
                    use_rerank=True,
                )
        finally:
            await retriever.close()

        if not result or not result.strip():
            return ""

        logger.info(f"memory_recall: 召回 {len(result)} 字符")
        return result

    except Exception as e:
        logger.error(f"memory_recall 失败: {e}")
        return ""


# ── Tool 2: memory_session_start ──────────────────────────────────────────────

@mcp.tool()
async def memory_session_start(
    session_id: str = "default",
    top_k: int = 15,
    agent_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    在新会话开始时调用，返回完整的历史背景记忆包。

    包括：最近项目状态、用户偏好规则、历史决策等。
    建议将返回内容注入到系统提示的开头。

    Args:
        session_id: 会话标识符
        top_k: 召回条数（默认 15，会话开始时返回更多背景）
        agent_id: Agent 唯一标识（HTTP 模式必填；stdio 模式从环境变量读取）
        api_key: API Key（当服务端配置了 MEMOCORE_API_KEY 时必填）

    Returns:
        格式化的背景记忆文本
    """
    if err := _check_api_key(api_key):
        return err

    resolved_agent_id = _resolve_agent_id(agent_id)
    logger.info(f"memory_session_start: agent={resolved_agent_id[:24]} session={session_id[:16]}")

    # 标记为已初始化（避免 memory_recall 重复触发全量召回）
    safe_agent = resolved_agent_id.replace("/", "_").replace(":", "_")[:32]
    flag = get_sessions_dir() / f"mcp-{safe_agent}-{session_id[:16]}.flag"
    flag.touch()

    try:
        from memocore.core.retriever import MemoryRetriever
        retriever = MemoryRetriever()
        try:
            result = await retriever.retrieve_for_session_start(
                agent_id=resolved_agent_id,
                top_k=top_k,
            )
        finally:
            await retriever.close()

        logger.info(f"memory_session_start: 召回 {len(result)} 字符")
        return result or ""

    except Exception as e:
        logger.error(f"memory_session_start 失败: {e}")
        return ""


# ── Tool 3: memory_store ──────────────────────────────────────────────────────

@mcp.tool()
async def memory_store(
    conversation: str,
    session_id: str = "default",
    source: str = "MCP client",
    agent_id: Optional[str] = None,
    api_key: Optional[str] = None,
) -> str:
    """
    将对话内容持久化到记忆图谱。

    在对话结束或有重要信息需要记住时调用。
    系统会自动从对话中提取结构化实体（偏好、决策、项目状态等）并写入 Neo4j。

    Args:
        conversation: 要存储的对话文本（User/Assistant 格式）
        session_id: 会话标识符（用于日志追踪）
        source: 数据来源描述（默认 "MCP client"）
        agent_id: Agent 唯一标识（HTTP 模式必填；stdio 模式从环境变量读取）
        api_key: API Key（当服务端配置了 MEMOCORE_API_KEY 时必填）

    Returns:
        存储结果描述（成功或失败信息）
    """
    if err := _check_api_key(api_key):
        return err

    resolved_agent_id = _resolve_agent_id(agent_id)
    logger.info(f"memory_store: agent={resolved_agent_id[:24]} session={session_id[:16]} len={len(conversation)}")

    if not conversation or not conversation.strip():
        return "跳过：对话内容为空"

    if len(conversation) < 100:
        return "跳过：对话内容太短（< 100 字符）"

    try:
        from memocore.core.extractor import extract_and_store
        result = await extract_and_store(
            conversation=conversation,
            agent_id=resolved_agent_id,
            source_description=f"{source} | session={session_id[:16]}",
        )

        if result["success"]:
            msg = (
                f"已存储记忆: episode={result['episode_name']} "
                f"entities={result.get('entities_extracted', '?')}"
            )
            logger.info(f"memory_store: {msg}")
            return msg
        else:
            err = f"存储失败: {result.get('error', '未知错误')}"
            logger.error(f"memory_store: {err}")
            return err

    except Exception as e:
        logger.error(f"memory_store 异常: {e}")
        return f"存储失败: {e}"


# ── Tool 4: health_check ───────────────────────────────────────────────────────

@mcp.tool()
async def health_check(
    api_key: Optional[str] = None,
) -> dict:
    """
    检查 Memocore 服务健康状态。

    返回服务版本、Neo4j 连接状态等信息。
    适合 Clawith 后端在启动时调用，确认 Memocore 服务可用。

    Args:
        api_key: API Key（当服务端配置了 MEMOCORE_API_KEY 时必填）

    Returns:
        健康状态字典
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
        from neo4j import AsyncGraphDatabase
        from memocore.core.config import get_neo4j_config
        uri, user, password = get_neo4j_config()
        async with AsyncGraphDatabase.driver(uri, auth=(user, password)) as driver:
            await driver.verify_connectivity()
        status["neo4j_connected"] = True
    except Exception as e:
        status["status"] = "degraded"
        status["neo4j_error"] = str(e)

    return status


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    """
    启动 Memocore MCP Server。

    支持两种传输模式：
      stdio  （默认）— 个人工具，配合 Claude Desktop / Cursor
      http           — 企业服务，多租户共享实例

    用法：
      memocore-mcp                               # stdio 模式
      memocore-mcp --transport http              # HTTP 模式，默认 0.0.0.0:8765
      memocore-mcp --transport http --port 9000  # 自定义端口
    """
    import click

    @click.command()
    @click.option(
        "--transport",
        type=click.Choice(["stdio", "http", "sse"]),
        default="stdio",
        show_default=True,
        help="传输模式：stdio（本地工具）| http（企业服务）| sse（旧版 HTTP）",
    )
    @click.option("--host", default="0.0.0.0", show_default=True, help="HTTP 监听地址")
    @click.option("--port", default=8765, show_default=True, help="HTTP 监听端口")
    def _main(transport: str, host: str, port: int):
        agent_id = get_agent_id()
        api_key_enabled = bool(os.environ.get("MEMOCORE_API_KEY", "").strip())

        if transport == "stdio":
            logger.info(f"Memocore MCP Server 启动 | transport=stdio | agent_id={agent_id}")
            mcp.run(transport="stdio")

        elif transport in ("http", "sse"):
            # 标记为 HTTP 模式，供 health_check 等工具读取
            os.environ["_MEMOCORE_HTTP_MODE"] = "1"
            os.environ["_MEMOCORE_TRANSPORT"] = transport

            logger.info(
                f"Memocore MCP Server 启动 | transport={transport} | "
                f"host={host} | port={port} | "
                f"api_key_auth={'enabled' if api_key_enabled else 'disabled'}"
            )

            if not api_key_enabled:
                click.echo(
                    "[WARNING] MEMOCORE_API_KEY 未设置，HTTP 服务未启用鉴权。\n"
                    "          生产环境请设置 MEMOCORE_API_KEY 或在前置 Nginx/网关层做认证。",
                    err=True,
                )

            click.echo(
                f"Memocore MCP Server 运行中\n"
                f"  地址：http://{host}:{port}\n"
                f"  鉴权：{'API Key (MEMOCORE_API_KEY)' if api_key_enabled else '未启用（开发模式）'}\n"
                f"  多租户：工具调用时传入 agent_id 参数\n"
            )

            mcp_transport = "sse" if transport == "sse" else "streamable-http"
            mcp.run(transport=mcp_transport, host=host, port=port)

    _main()


if __name__ == "__main__":
    main()
