#!/usr/bin/env python3
"""
T6: Claude Code Stop Hook
触发时机：每次对话结束（Claude Code 的 Stop 事件）
功能：读取本次对话内容 → 提炼记忆 → 写入 Graphiti/Neo4j

使用方式（在 ~/.claude/settings.json 里配置）：
{
  "hooks": {
    "Stop": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "/Users/shenfeng/philosopher_env/bin/python3 /path/to/MemOS/memos/adapters/claude_code/stop_hook.py"
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
"""

import sys
import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# 确保能 import memocore
_project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(_project_root))

LOG_FILE = Path.home() / ".private" / "memos_hook.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [stop_hook] %(message)s",
)


def parse_transcript(transcript: list) -> str:
    """把 transcript 列表转成可提炼的纯文本"""
    lines = []
    for turn in transcript:
        role = turn.get("role", "unknown")
        content = turn.get("content", "")

        # content 可能是字符串或列表（tool_use等）
        if isinstance(content, list):
            text_parts = [
                c.get("text", "") for c in content
                if isinstance(c, dict) and c.get("type") == "text"
            ]
            content = " ".join(text_parts)

        if content and content.strip():
            prefix = "F哥" if role == "user" else "鳌虾"
            lines.append(f"{prefix}：{content.strip()[:500]}")  # 截断超长内容

    return "\n".join(lines)


def should_extract(transcript_text: str) -> bool:
    """判断是否值得提炼（过滤掉没有实质内容的短对话）"""
    if len(transcript_text) < 200:
        return False
    # 跳过纯查询类对话
    skip_keywords = ["你好", "在吗", "测试", "hello", "test"]
    first_line = transcript_text.split("\n")[0].lower()
    if any(k in first_line for k in skip_keywords):
        return False
    return True


async def run(hook_input: dict):
    from memocore.core.extractor import extract_and_store
    from memocore.core.dream import run_dream

    transcript = hook_input.get("transcript", [])
    session_id = hook_input.get("session_id", "unknown")

    if not transcript:
        logging.info(f"session={session_id} transcript为空，跳过")
        return

    transcript_text = parse_transcript(transcript)

    if not should_extract(transcript_text):
        logging.info(f"session={session_id} 内容太短或无实质内容，跳过")
        return

    logging.info(f"session={session_id} 开始提炼，文本长度={len(transcript_text)}")

    # Step 1: 提炼写入
    result = await extract_and_store(
        conversation=transcript_text,
        agent_id="aoxia",
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

    # Step 2: Dream 巩固（异步后台，不阻塞 hook 返回）
    # 仅在每隔 N 次会话后触发，避免每次都跑（通过简单的会话ID哈希）
    session_hash = hash(session_id) % 5  # 大约每5次触发一次
    if session_hash == 0:
        logging.info(f"session={session_id} 触发 Dream 记忆巩固...")
        try:
            dream_report = await run_dream(agent_id="aoxia")
            logging.info(f"session={session_id} Dream完成: {dream_report.summary()}")
        except Exception as e:
            logging.warning(f"session={session_id} Dream 运行失败（非致命）: {e}")
    else:
        logging.info(f"session={session_id} 跳过 Dream（本次 hash={session_hash}，每5次触发一次）")


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
