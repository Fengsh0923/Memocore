"""
MemOS UserPromptSubmit Hook — 对话开始时自动召回历史记忆

触发时机：每次用户发消息时（UserPromptSubmit）
策略：
  - Session 首条消息：retrieve_for_session_start（3 并行查询，top10）
    → 给鳌虾一个完整的"记忆背景包"
  - 后续消息：基于用户 prompt 快速召回 top5（跳过 rerank，控制延迟）
    → 只在 prompt 明确涉及某个项目/规则时追加

用法（在 settings.json 里注册）：
  {
    "type": "command",
    "command": "/Users/shenfeng/philosopher_env/bin/python3 /.../prompt_hook.py",
    "timeout": 25
  }
"""

import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

LOG_FILE = Path.home() / ".private" / "memos_prompt_hook.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [prompt_hook] %(levelname)s %(message)s",
)
logger = logging.getLogger("memos.prompt_hook")

# Session 首次召回的 flag 文件前缀
SESSION_FLAG_PREFIX = "/tmp/memos-session-"


def is_first_message(session_id: str) -> bool:
    """检查是否是该 session 的第一条用户消息"""
    flag = Path(f"{SESSION_FLAG_PREFIX}{session_id[:16]}.flag")
    if flag.exists():
        return False
    flag.touch()
    return True


def should_retrieve(prompt: str) -> bool:
    """
    判断是否需要基于 prompt 做追加召回
    过滤掉太短或纯粹是打招呼的消息
    """
    p = prompt.strip()
    if len(p) < 5:
        return False
    # 纯打招呼不需要召回
    greetings = {"你好", "hi", "hello", "嗯", "好的", "ok", "继续", "开始"}
    if p.lower() in greetings:
        return False
    return True


async def run(hook_input: dict) -> dict:
    """
    主流程，返回 additionalContext 字典
    """
    prompt = hook_input.get("prompt", "")
    session_id = hook_input.get("session_id", "unknown")

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from memos.core.retriever import MemoryRetriever

    retriever = MemoryRetriever()
    context_text = ""

    try:
        first_message = is_first_message(session_id)

        if first_message:
            # Session 首条消息：完整背景包
            logger.info(f"session={session_id[:16]} 首条消息，触发全量记忆召回")
            context_text = await retriever.retrieve_for_session_start(
                agent_id="aoxia",
                top_k=10,
            )
            logger.info(
                f"session={session_id[:16]} 全量召回完成，"
                f"注入字符数={len(context_text)}"
            )

        elif should_retrieve(prompt):
            # 后续消息：基于 prompt 快速召回（不 rerank，控制延迟）
            logger.info(f"session={session_id[:16]} 后续消息，基于 prompt 快速召回")
            context_text = await retriever.retrieve(
                query=prompt,
                agent_id="aoxia",
                top_k=5,
                use_rerank=False,   # 跳过 LLM rerank，压缩延迟
            )
            logger.info(
                f"session={session_id[:16]} 快速召回完成，"
                f"注入字符数={len(context_text)}"
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
        hook_input = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        hook_input = {}

    session_id = hook_input.get("session_id", "unknown")

    try:
        context_text = asyncio.run(run(hook_input))
    except Exception as e:
        logger.error(f"session={session_id[:16]} hook 主流程失败: {e}")
        # 失败不阻塞对话，静默退出
        sys.exit(0)

    if not context_text or not context_text.strip():
        # 无记忆可注入，静默退出
        sys.exit(0)

    # 输出注入格式
    output = {
        "hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": (
                "\n--- MemOS 历史记忆（自动召回）---\n"
                + context_text
                + "\n--- 历史记忆结束 ---\n"
            ),
        }
    }
    print(json.dumps(output, ensure_ascii=False))
    sys.exit(0)


if __name__ == "__main__":
    main()
