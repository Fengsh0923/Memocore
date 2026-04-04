"""
Memocore MCP Server

将 Memocore 的记忆写入和召回能力暴露为 MCP Tools，
兼容任何支持 MCP 协议的客户端：
  - Claude Desktop
  - Cursor
  - Windsurf
  - Continue.dev
  - 以及其他 MCP 兼容工具

暴露的 Tools：
  memory_recall(query, session_id?)  — 根据当前 prompt 召回相关记忆
  memory_store(conversation, session_id?)  — 把一段对话写入记忆图谱
  memory_session_start(session_id?)  — 会话开始时全量召回（返回背景记忆包）

启动方式：
  # stdio 模式（推荐，配合 Claude Desktop / Cursor）
  python -m memocore.adapters.mcp.server

Claude Desktop 配置示例 (~/.claude/claude_desktop_config.json)：
  {
    "mcpServers": {
      "memocore": {
        "command": "python3",
        "args": ["-m", "memocore.adapters.mcp.server"],
        "env": {
          "MEMOCORE_AGENT_ID": "my-project",
          "ANTHROPIC_API_KEY": "sk-..."
        }
      }
    }
  }

Cursor 配置示例 (~/.cursor/mcp.json)：
  {
    "mcpServers": {
      "memocore": {
        "command": "python3",
        "args": ["-m", "memocore.adapters.mcp.server"]
      }
    }
  }
"""

import logging
import sys
from pathlib import Path

# 确保能 import memocore（从 project root 运行时）
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

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
        "MCP 依赖未安装，请运行：pip install mcp\n"
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
    ),
)


def _is_first_message(session_id: str) -> bool:
    """检查是否是 session 的第一条消息"""
    flag = get_sessions_dir() / f"mcp-{session_id[:16]}.flag"
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
) -> str:
    """
    根据当前 prompt 从记忆图谱召回最相关的历史记忆。

    在处理每条用户消息之前调用此工具，将返回的记忆注入到你的上下文中。
    如果是 session 的第一条消息，会自动触发全量背景召回。

    Args:
        query: 用户的当前消息或关键词
        session_id: 会话标识符，用于区分首次/后续消息
        top_k: 返回的记忆条数（默认 5）

    Returns:
        格式化的历史记忆文本，可直接注入系统提示
    """
    agent_id = get_agent_id()
    logger.info(f"memory_recall: session={session_id[:16]} query={query[:50]}")

    try:
        from memocore.core.retriever import MemoryRetriever
        retriever = MemoryRetriever()

        try:
            if _is_first_message(session_id):
                logger.info(f"memory_recall: 首条消息，触发全量召回")
                result = await retriever.retrieve_for_session_start(
                    agent_id=agent_id,
                    top_k=max(top_k * 2, 10),
                )
            else:
                result = await retriever.retrieve(
                    query=query,
                    agent_id=agent_id,
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
) -> str:
    """
    在新会话开始时调用，返回完整的历史背景记忆包。

    包括：最近项目状态、用户偏好规则、历史决策等。
    建议将返回内容注入到系统提示的开头。

    Args:
        session_id: 会话标识符
        top_k: 召回条数（默认 15，会话开始时返回更多背景）

    Returns:
        格式化的背景记忆文本
    """
    agent_id = get_agent_id()
    logger.info(f"memory_session_start: session={session_id[:16]}")

    # 标记为已初始化（避免 memory_recall 重复触发全量召回）
    flag = get_sessions_dir() / f"mcp-{session_id[:16]}.flag"
    flag.touch()

    try:
        from memocore.core.retriever import MemoryRetriever
        retriever = MemoryRetriever()
        try:
            result = await retriever.retrieve_for_session_start(
                agent_id=agent_id,
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
) -> str:
    """
    将对话内容持久化到记忆图谱。

    在对话结束或有重要信息需要记住时调用。
    系统会自动从对话中提取结构化实体（偏好、决策、项目状态等）并写入 Neo4j。

    Args:
        conversation: 要存储的对话文本（User/Assistant 格式）
        session_id: 会话标识符（用于日志追踪）
        source: 数据来源描述（默认 "MCP client"）

    Returns:
        存储结果描述（成功或失败信息）
    """
    agent_id = get_agent_id()
    logger.info(f"memory_store: session={session_id[:16]} len={len(conversation)}")

    if not conversation or not conversation.strip():
        return "跳过：对话内容为空"

    if len(conversation) < 100:
        return "跳过：对话内容太短（< 100 字符）"

    try:
        from memocore.core.extractor import extract_and_store
        result = await extract_and_store(
            conversation=conversation,
            agent_id=agent_id,
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


# ── 入口 ───────────────────────────────────────────────────────────────────────

def main():
    """以 stdio 模式启动 MCP server（供 Claude Desktop / Cursor 调用）"""
    logger.info(f"Memocore MCP Server 启动 | agent_id={get_agent_id()}")
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
