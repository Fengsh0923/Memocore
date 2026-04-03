"""
MemOS Bridge Write Hook
供 claude-to-im 桥接在每次对话结束后调用

输入（stdin JSON）：
  {
    "session_id": "...",
    "prompt": "用户消息",
    "response": "AI 回复",
    "channel": "feishu" | "wechat" | ...
  }

功能：
  - 把 prompt+response 组成对话文本
  - 调用 MemoryExtractor 写入 Neo4j
  - 过滤太短的对话（< 100 字）

用法（在 llm-provider.ts 里调用）：
  const proc = spawn(pythonPath, [bridgeWritePath]);
  proc.stdin.write(JSON.stringify({session_id, prompt, response, channel}));
  proc.stdin.end();
"""

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

LOG_FILE = Path.home() / ".private" / "memos_bridge.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s [bridge_write] %(levelname)s %(message)s",
)
logger = logging.getLogger("memos.bridge_write")

MIN_LENGTH = 100  # 少于100字不写入


async def run(data: dict):
    session_id = data.get("session_id", "unknown")
    prompt = data.get("prompt", "").strip()
    response = data.get("response", "").strip()
    channel = data.get("channel", "bridge")

    if not prompt and not response:
        logger.info(f"session={session_id[:16]} prompt和response都为空，跳过")
        return

    conversation = f"用户：{prompt}\n鳌虾：{response}"

    if len(conversation) < MIN_LENGTH:
        logger.info(f"session={session_id[:16]} 对话太短({len(conversation)}字)，跳过")
        return

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    from memos.core.extractor import extract_and_store

    source = f"桥接对话 | channel={channel} | session={session_id[:16]} | {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    logger.info(f"session={session_id[:16]} 开始写入，对话长度={len(conversation)}")

    result = await extract_and_store(
        conversation=conversation,
        agent_id="aoxia",
        source_description=source,
    )

    if result["success"]:
        logger.info(
            f"session={session_id[:16]} 写入成功: "
            f"episode={result['episode_name']} "
            f"entities={result.get('entities_extracted', '?')}"
        )
    else:
        logger.error(f"session={session_id[:16]} 写入失败: {result.get('error')}")


def main():
    stdin_data = sys.stdin.read()
    try:
        data = json.loads(stdin_data) if stdin_data.strip() else {}
    except json.JSONDecodeError:
        logger.error("无效的 JSON 输入")
        sys.exit(0)

    asyncio.run(run(data))


if __name__ == "__main__":
    main()
