"""
T4: 对话提炼脚本
功能：把一段对话文本 → 提炼实体+关系 → 写入 Graphiti

用法：
    from memocore.core.extractor import MemoryExtractor
    extractor = MemoryExtractor()
    await extractor.extract_and_store(conversation="...", agent_id="aoxia")
"""

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# 加载 .env（支持从任意目录调用）
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

from memocore.agents.aoxia.schema import AOXIA_ENTITY_TYPES


class MemoryExtractor:
    """
    鳌虾记忆提炼器
    - 接收对话文本
    - 调用 Graphiti 提炼实体+关系
    - 写入 Neo4j
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self._graphiti: Optional[Graphiti] = None

    async def _get_graphiti(self) -> Graphiti:
        if self._graphiti is None:
            self._graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            )
            await self._graphiti.build_indices_and_constraints()
        return self._graphiti

    async def extract_and_store(
        self,
        conversation: str,
        agent_id: str = "aoxia",
        source_description: str = "Claude Code 对话",
        episode_name: Optional[str] = None,
        reference_time: Optional[datetime] = None,
    ) -> dict:
        """
        核心方法：提炼对话内容并写入记忆图谱

        Args:
            conversation: 对话文本（可以是完整的 session 内容）
            agent_id: Agent 标识，用于 Neo4j namespace 隔离
            source_description: 数据来源描述
            episode_name: 本次 episode 的名称（默认自动生成）
            reference_time: 参考时间（默认当前时间）

        Returns:
            {success: bool, episode_name: str, entities_extracted: int, error: str}
        """
        if not conversation or not conversation.strip():
            return {"success": False, "error": "对话内容为空"}

        if episode_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            episode_name = f"{agent_id}_{ts}"

        if reference_time is None:
            reference_time = datetime.now(timezone.utc)

        try:
            graphiti = await self._get_graphiti()

            # 提炼系统提示词：告诉 Graphiti 用哪些实体类型
            extraction_instructions = """
你正在处理 F哥（Frank）与 AI 助手鳌虾的对话记录。

提炼重点：
1. F哥表达的偏好、规则、判断标准 → FrankPreference
2. 项目状态更新或决策 → ProjectStatus / Judgment
3. 技术方案选择和理由 → Judgment
4. 任务派发和结果 → TaskRecord
5. 故障或踩坑 → Incident
6. 服务端口、路径、API信息 → ExternalResource
7. 飞虾队各虾的配置变更 → AgentConfig

注意：
- 只提炼明确表达的信息，不要推断
- F哥说"好虾"表示认可，这之前的内容通常是已确认的判断
- 技术细节（端口号、路径、API key位置）要精确提取
"""

            result = await graphiti.add_episode(
                name=episode_name,
                episode_body=conversation,
                source_description=source_description,
                reference_time=reference_time,
                source=EpisodeType.message,
                group_id=agent_id,
                entity_types=AOXIA_ENTITY_TYPES,
                custom_extraction_instructions=extraction_instructions,
            )

            # 统计提炼出的实体数量
            entities_count = len(result.nodes) if hasattr(result, 'nodes') else 0

            return {
                "success": True,
                "episode_name": episode_name,
                "entities_extracted": entities_count,
                "group_id": agent_id,
            }

        except Exception as e:
            return {
                "success": False,
                "episode_name": episode_name,
                "error": str(e),
            }

    async def close(self):
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None


# ─── 便捷函数（供 hooks 直接调用）──────────────────────────────────────────────

async def extract_and_store(
    conversation: str,
    agent_id: str = "aoxia",
    source_description: str = "Claude Code 对话",
) -> dict:
    """
    无需手动管理 extractor 实例的便捷函数
    适合在 Claude Code hooks 里直接调用
    """
    extractor = MemoryExtractor()
    try:
        result = await extractor.extract_and_store(
            conversation=conversation,
            agent_id=agent_id,
            source_description=source_description,
        )
        return result
    finally:
        await extractor.close()


if __name__ == "__main__":
    # 快速测试
    test_conv = """
    F哥：MemOS 技术选型定了吗？
    鳌虾：确认用 Graphiti + Neo4j。Neo4j 本机已有，不需要额外运维。
    F哥：好虾。另外记住，所有给我的产出文件必须存到坚果云，不能只给链接。
    鳌虾：收到，文件统一存 FS_KM/00_Agent_Sandbox/Inbox/Daily/。
    F哥：对，还有飞书通知统一走单聊，用 open_id，不发群聊。
    """

    async def run():
        result = await extract_and_store(
            conversation=test_conv,
            source_description="T4 提炼脚本测试"
        )
        print("提炼结果:", result)

    asyncio.run(run())
