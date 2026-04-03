"""
T5: 记忆召回函数 (v2: 加入二次筛选)
功能：根据当前对话上下文 → 召回最相关历史记忆 → 输出可注入 system prompt 的文本

召回策略（v2）：
  Stage 1: Graphiti 向量+图混合检索，top_k=20（粗筛）
  Stage 2: gpt-4o-mini 根据 query 对 20 条结果评分，取 top_5（精筛）
  → 精筛后的结果更精准，注入 system prompt 不会稀释信号

用法：
    from memos.core.retriever import MemoryRetriever
    retriever = MemoryRetriever()
    context = await retriever.retrieve("F哥对飞书通知的要求", agent_id="aoxia")
    # context 是可直接注入 system prompt 的 Markdown 文本
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from graphiti_core import Graphiti
from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF


class MemoryRetriever:
    """
    记忆召回器 v2
    - Stage 1: Graphiti 语义+图遍历混合检索（top_k_stage1 条，默认20）
    - Stage 2: gpt-4o-mini 精筛（top_k_final 条，默认5）
    - 输出格式化的 Markdown，可直接注入 system prompt
    """

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        top_k_stage1: int = 20,
        top_k_final: int = 5,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "")
        self.top_k_stage1 = top_k_stage1
        self.top_k_final = top_k_final
        self._graphiti: Optional[Graphiti] = None

    async def _get_graphiti(self) -> Graphiti:
        if self._graphiti is None:
            self._graphiti = Graphiti(
                uri=self.neo4j_uri,
                user=self.neo4j_user,
                password=self.neo4j_password,
            )
        return self._graphiti

    async def _rerank_with_llm(
        self,
        query: str,
        candidates: list,
        top_k: int,
    ) -> list:
        """
        Stage 2: 用 gpt-4o-mini 对候选结果精筛
        输入：query + 最多20条 fact 文本
        输出：按相关性排序的 top_k 索引列表
        """
        if len(candidates) <= top_k:
            return candidates  # 候选不超过 top_k，直接返回

        try:
            import openai
            client = openai.AsyncOpenAI(api_key=self.openai_api_key)

            # 构建候选列表文本
            items = []
            for i, r in enumerate(candidates):
                fact = getattr(r, 'fact', None) or str(r)[:150]
                items.append(f"{i}: {fact}")
            candidates_text = "\n".join(items)

            prompt = f"""你是记忆召回助手。用户当前的问题/上下文是：

"{query}"

以下是从知识图谱召回的候选记忆（共{len(candidates)}条）：

{candidates_text}

请从中选出最相关的 {top_k} 条，按相关性从高到低排序。
仅返回 JSON 数组，包含选中条目的索引号，例如：[3, 0, 7, 1, 5]"""

            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=100,
            )

            content = response.choices[0].message.content.strip()
            # 提取 JSON 数组
            import re
            match = re.search(r'\[[\d,\s]+\]', content)
            if match:
                indices = json.loads(match.group())
                selected = []
                for idx in indices[:top_k]:
                    if 0 <= idx < len(candidates):
                        selected.append(candidates[idx])
                return selected

        except Exception:
            pass  # 精筛失败时 fallback 到 top_k 截断

        return candidates[:top_k]

    async def retrieve(
        self,
        query: str,
        agent_id: str = "aoxia",
        top_k: int = 5,
        use_rerank: bool = True,
        as_markdown: bool = True,
    ) -> str:
        """
        召回与 query 最相关的历史记忆

        Args:
            query: 当前对话上下文或关键词
            agent_id: Agent namespace，默认 aoxia
            top_k: 最终返回条数（精筛后），默认5
            use_rerank: True 时启用二次精筛（Stage 2），False 直接返回 Stage 1 结果
            as_markdown: True 返回可注入 system prompt 的 Markdown，False 返回原始结果

        Returns:
            格式化的记忆文本，可直接放入 system prompt
        """
        if not query.strip():
            return ""

        try:
            graphiti = await self._get_graphiti()

            # Stage 1: 粗筛（top_k_stage1，默认20）
            stage1_k = self.top_k_stage1 if use_rerank else top_k
            results = await graphiti.search(
                query=query,
                group_ids=[agent_id],
                num_results=stage1_k,
            )

            if not results:
                return ""

            # Stage 2: 精筛（gpt-4o-mini rerank）
            if use_rerank and len(results) > top_k:
                results = await self._rerank_with_llm(query, results, top_k)

            if not as_markdown:
                return str(results)

            return self._format_as_context(results, agent_id)

        except Exception as e:
            return f"<!-- 记忆召回失败: {e} -->"

    async def retrieve_for_session_start(
        self,
        agent_id: str = "aoxia",
        top_k: int = 15,
    ) -> str:
        """
        会话开始时的全量相关记忆召回
        召回最近的项目状态、规则和判断
        """
        queries = [
            "F哥最近的项目和决策",
            "F哥的偏好规则和判断标准",
            "飞虾队当前状态和任务",
        ]

        all_results = []
        graphiti = await self._get_graphiti()

        for q in queries:
            try:
                results = await graphiti.search(
                    query=q,
                    group_ids=[agent_id],
                    num_results=top_k // len(queries),
                )
                all_results.extend(results)
            except Exception:
                continue

        if not all_results:
            return ""

        # 去重（按 uuid）
        seen = set()
        unique_results = []
        for r in all_results:
            uid = getattr(r, 'uuid', str(r))
            if uid not in seen:
                seen.add(uid)
                unique_results.append(r)

        return self._format_as_context(unique_results, agent_id, is_session_start=True)

    def _format_as_context(
        self,
        results: list,
        agent_id: str,
        is_session_start: bool = False,
    ) -> str:
        """把召回结果格式化为 system prompt 可用的 Markdown"""

        header = "## 鳌虾记忆召回\n" if is_session_start else "## 相关历史记忆\n"
        lines = [header]
        lines.append(f"*来源：{agent_id} 知识图谱 | {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        for i, r in enumerate(results, 1):
            # Graphiti 返回的是 edge 对象，尝试提取 fact 字段
            fact = getattr(r, 'fact', None)
            if fact:
                lines.append(f"{i}. {fact}")
            else:
                # fallback：取字符串表示的前200字
                lines.append(f"{i}. {str(r)[:200]}")

        return "\n".join(lines)

    async def close(self):
        if self._graphiti:
            await self._graphiti.close()
            self._graphiti = None


# ─── 便捷函数 ─────────────────────────────────────────────────────────────────

async def retrieve(
    query: str,
    agent_id: str = "aoxia",
    top_k: int = 10,
) -> str:
    """单次召回便捷函数，适合 hooks 调用"""
    retriever = MemoryRetriever()
    try:
        return await retriever.retrieve(query=query, agent_id=agent_id, top_k=top_k)
    finally:
        await retriever.close()


async def retrieve_for_session_start(agent_id: str = "aoxia") -> str:
    """会话开始时调用，召回全量相关记忆"""
    retriever = MemoryRetriever()
    try:
        return await retriever.retrieve_for_session_start(agent_id=agent_id)
    finally:
        await retriever.close()


if __name__ == "__main__":
    import asyncio

    async def run():
        print("=== T5 记忆召回测试 ===\n")

        # 测试1：精准召回
        result = await retrieve("飞书通知规则", agent_id="aoxia")
        print("[精准召回] 飞书通知规则：")
        print(result or "（无结果）")
        print()

        # 测试2：会话开始全量召回
        result2 = await retrieve_for_session_start(agent_id="aoxia")
        print("[会话开始] 全量记忆召回：")
        print(result2 or "（无结果）")

    asyncio.run(run())
