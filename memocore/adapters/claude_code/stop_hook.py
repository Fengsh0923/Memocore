#!/usr/bin/env python3
"""
Claude Code Stop Hook
触发时机：每次对话结束（Claude Code 的 Stop 事件）
功能：读取本次对话内容 → 提炼记忆 → 写入 Graphiti/Neo4j
      满足 MEMOCORE_DREAM_INTERVAL 次会话后后台触发 Dream 巩固

使用方式（在 ~/.claude/settings.json 里配置）：
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "python3 /path/to/memocore/adapters/claude_code/stop_hook.py",
        "timeout": 60
      }]
    }]
  }
}

Claude Code Stop hook 通过 stdin 传入 JSON：
{
  "session_id": "...",
  "transcript": [{"role": "user"|"assistant", "content": "..."}],
  "stop_hook_active": true
}

环境变量：
  MEMOCORE_AGENT_ID         — Agent namespace（默认 "aoxia"）
  MEMOCORE_DREAM_INTERVAL   — Dream 触发间隔会话数（默认 5）
  MEMOCORE_LLM_PROVIDER     — LLM provider（默认自动检测）
"""

import sys
import json
import asyncio
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# 确保能 import memocore
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

# 加载 .env
from dotenv import load_dotenv
_env_path = _project_root / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from memocore.core.config import get_agent_id, should_run_dream, get_logs_dir
from memocore.agents.aoxia.schema import AOXIA_PROFILE


def _get_profile(agent_id: str) -> dict:
    if agent_id == "aoxia":
        return AOXIA_PROFILE
    return {"user_display_name": "User", "assistant_display_name": "Assistant"}


LOG_FILE = get_logs_dir() / "stop_hook.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [stop_hook] %(message)s",
)


def parse_transcript(transcript: list, profile: dict) -> str:
    """把 transcript 列表转成可提炼的纯文本"""
    user_name = profile.get("user_display_name", "User")
    assistant_name = profile.get("assistant_display_name", "Assistant")

    lines = []
    for turn in transcript:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")

        # content 可能是字符串或列表（tool_use 等）
        if isinstance(content, list):
            text_parts = [
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = " ".join(text_parts)

        if content and content.strip():
            # 按 token 估算截断：约 4 字符 ≈ 1 token，限 300 token/条
            text = content.strip()
            if len(text) > 1200:
                text = text[:1200] + "…"
            prefix = user_name if role == "user" else assistant_name
            lines.append(f"{prefix}：{text}")

    return "\n".join(lines)


def should_extract(transcript_text: str) -> bool:
    """判断是否值得提炼（过滤掉没有实质内容的短对话）"""
    if len(transcript_text) < 200:
        return False
    skip_keywords = ["你好", "在吗", "测试", "hello", "test"]
    first_line = transcript_text.split("\n")[0].lower()
    if any(k in first_line for k in skip_keywords):
        return False
    return True


def _spawn_dream_background(agent_id: str):
    """
    在独立子进程后台运行 Dream，不阻塞 stop hook 返回
    """
    dream_script = Path(__file__).parent.parent.parent / "core" / "dream.py"
    try:
        subprocess.Popen(
            [sys.executable, str(dream_script), "--agent-id", agent_id],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,  # 与父进程完全解耦
        )
        logging.info(f"[dream] 已在后台启动 Dream 子进程 agent={agent_id}")
    except Exception as e:
        logging.warning(f"[dream] 启动后台 Dream 失败（非致命）: {e}")


async def run(hook_input: dict):
    from memocore.core.extractor import extract_and_store

    agent_id = get_agent_id()
    profile = _get_profile(agent_id)
    transcript = hook_input.get("transcript", [])
    session_id = hook_input.get("session_id", "unknown")

    if not transcript:
        logging.info(f"session={session_id} transcript 为空，跳过")
        return

    transcript_text = parse_transcript(transcript, profile)

    if not should_extract(transcript_text):
        logging.info(f"session={session_id} 内容太短或无实质内容，跳过")
        return

    logging.info(f"session={session_id} 开始提炼，文本长度={len(transcript_text)}")

    # Step 1: 提炼写入
    result = await extract_and_store(
        conversation=transcript_text,
        agent_id=agent_id,
        source_description=f"Claude Code 对话 | session={session_id} | {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    )

    if result["success"]:
        logging.info(
            f"session={session_id} 提炼成功: "
            f"episode={result['episode_name']} "
            f"entities={result.get('entities_extracted', '?')}"
        )
    else:
        logging.error(f"session={session_id} 提炼失败: {result.get('error')}")
        return  # 提炼失败时不触发 Dream

    # Step 2: Dream 巩固（后台子进程，不阻塞 hook 返回）
    # 使用持久化计数器，保证每 MEMOCORE_DREAM_INTERVAL 次必触发一次
    if should_run_dream(agent_id):
        logging.info(f"session={session_id} 计数器达到阈值，触发后台 Dream...")
        _spawn_dream_background(agent_id)
    else:
        logging.info(f"session={session_id} Dream 计数未达阈值，跳过")


def main():
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return
        hook_input = json.loads(raw)
    except Exception as e:
        logging.error(f"读取 hook 输入失败: {e}")
        return

    asyncio.run(run(hook_input))


if __name__ == "__main__":
    main()
