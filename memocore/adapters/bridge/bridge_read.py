"""
MemOS Bridge Read Hook
供 claude-to-im 桥接在每次收到用户消息后调用，召回历史记忆注入 system prompt

输入（stdin JSON）：
  {
    "prompt": "用户消息",
    "session_id": "...",
    "is_first_message": true | false   （可选，默认 false）
  }

输出（stdout）：
  纯文本，可直接追加到 system prompt；若无记忆则输出空字符串

用法（在 daemon.mjs 的 processMessage 里调用）：
  const proc = spawn(pythonPath, [bridgeReadPath]);
  proc.stdin.write(JSON.stringify({prompt, session_id, is_first_message}));
  proc.stdin.end();
  // 读取 stdout 拼入 systemPrompt
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from memocore.core.config import get_agent_id, get_sessions_dir, get_logs_dir

LOG_FILE = get_logs_dir() / "bridge_read.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [bridge_read] %(levelname)s %(message)s",
)
logger = logging.getLogger("memos.bridge_read")


def is_first_message(session_id: str) -> bool:
    flag = get_sessions_dir() / f"bridge-{session_id[:16]}.flag"
    if flag.exists():
        return False
    flag.touch()
    return True


def should_retrieve(prompt: str) -> bool:
    p = prompt.strip()
    if len(p) < 5:
        return False
    greetings = {"你好", "hi", "hello", "嗯", "好的", "ok", "继续", "开始", "在吗"}
    if p.lower() in greetings:
        return False
    return True


async def run(data: dict) -> str:
    prompt = data.get("prompt", "").strip()
    session_id = data.get("session_id", "unknown")
    force_first = data.get("is_first_message", False)
    agent_id = get_agent_id()

    from memocore.core.retriever import MemoryRetriever

    retriever = MemoryRetriever()
    context_text = ""

    try:
        first_msg = force_first or is_first_message(session_id)

        if first_msg:
            logger.info(f"session={session_id[:16]} 首条消息，触发全量记忆召回")
            context_text = await retriever.retrieve_for_session_start(
                agent_id=agent_id,
                top_k=10,
            )
            logger.info(
                f"session={session_id[:16]} 全量召回完成，字符数={len(context_text)}"
            )

        elif should_retrieve(prompt):
            logger.info(f"session={session_id[:16]} 后续消息，基于 prompt 快速召回")
            context_text = await retriever.retrieve(
                query=prompt,
                agent_id=agent_id,
                top_k=5,
                use_rerank=False,  # 跳过 LLM rerank，压缩延迟
            )
            logger.info(
                f"session={session_id[:16]} 快速召回完成，字符数={len(context_text)}"
            )
        else:
            logger.info(f"session={session_id[:16]} prompt 太短或打招呼，跳过召回")

    except Exception as e:
        logger.error(f"session={session_id[:16]} 召回失败: {e}")
        context_text = ""

    finally:
        await retriever.close()

    return context_text


def main():
    stdin_data = sys.stdin.read()
    try:
        data = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        logger.error("无效的 JSON 输入")
        sys.stdout.write("")
        sys.exit(0)

    try:
        result = asyncio.run(run(data))
    except Exception as e:
        logger.error(f"bridge_read 主流程失败: {e}")
        result = ""

    # 输出到 stdout 供 daemon.mjs 读取
    if result and result.strip():
        sys.stdout.write(
            "\n--- MemOS 历史记忆（自动召回）---\n"
            + result
            + "\n--- 历史记忆结束 ---\n"
        )
    else:
        sys.stdout.write("")

    sys.exit(0)


if __name__ == "__main__":
    main()
