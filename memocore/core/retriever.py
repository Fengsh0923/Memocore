"""
T5: 记忆召回函数 (v2: 加入二次筛选)
功能：根据当前对话上下文 → 召回最相关历史记忆 → 输出可注入 system prompt 的文本

召回策略（v2）：
  Stage 1: Graphiti 向量+图混合检索，top_k=20（粗筛）
  Stage 2: gpt-4o-mini 根据 query 对 20 条结果评分，取 top_5（精筛）
  → 精筛后的结果更精准，注入 system prompt 不会稀释信号

用法：
    from memocore.core.retriever import MemoryRetriever
    retriever = MemoryRetriever()
    context = await retriever.retrieve("F哥对飞书通知的要求", agent_id="aoxia")
    # context 是可直接注入 system prompt 的 Markdown 文本
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)

from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
from memocore.core.llm_adapter import rerank as llm_rerank
from memocore.core.graphiti_factory import build_graphiti
from memocore.agents.aoxia.schema import AOXIA_PROFILE


def _filter_by_confidence(results: list, min_confidence: float = 0.2) -> list:
    """
    过滤掉置信度过低的节点（memocore_confidence < min_confidence 或 status='stale'）
    Graphiti edge 对象通常不直接携带这些属性，尝试读取；读不到则默认保留
    """
    filtered = []
    for r in results:
        confidence = getattr(r, 'memocore_confidence', None)
        status = getattr(r, 'memocore_status', None)
        # 如果节点已被标记为 stale 且置信度很低，跳过
        if status == 'stale' and confidence is not None and confidence < min_confidence:
            continue
        filtered.append(r)
    return filtered if filtered else results  # 若全被过滤则返回原列表（降级处理）


def _get_profile(agent_id: str) -> dict:
    if agent_id == "aoxia":
        return AOXIA_PROFILE
    return {
        "session_start_queries": ["recent decisions and preferences", "active projects"],
    }


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
        top_k_stage1: int = 20,
        top_k_final: int = 5,
    ):
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = neo4j_user or os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "")
        self.top_k_stage1 = top_k_stage1
        self.top_k_final = top_k_final
        self._graphiti: Optional[Graphiti] = None

    async def _get_graphiti(self):
        if self._graphiti is None:
            self._graphiti = await build_graphiti(
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
        """Stage 2: 通过 LLM adapter 对候选结果精筛"""
        return await llm_rerank(query=query, candidates=candidates, top_k=top_k)

    async def _search_scoped(
        self,
        graphiti,
        query: str,
        group_ids: list[str],
        num_results: int,
        scope_label: str,
    ) -> list:
        """对单个 scope 执行搜索，并在结果上打标（_scope_label 属性）"""
        try:
            results = await graphiti.search(
                query=query,
                group_ids=group_ids,
                num_results=num_results,
            )
            for r in results:
                r._scope_label = scope_label
            return results
        except Exception:
            return []

    async def retrieve(
        self,
        query: str,
        agent_id: str = "aoxia",
        top_k: int = 5,
        use_rerank: bool = True,
        as_markdown: bool = True,
        team_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        召回与 query 最相关的历史记忆，支持多范围合并检索

        Args:
            query: 当前对话上下文或关键词
            agent_id: 个人 namespace
            top_k: 最终返回条数，默认 5
            use_rerank: 启用 LLM 二次精筛
            as_markdown: True 返回 Markdown，False 返回原始结果
            team_id: 团队 ID，传入后同时检索团队记忆
            tenant_id: 组织 ID，传入后同时检索组织知识

        Returns:
            格式化的记忆文本，可直接放入 system prompt
        """
        import asyncio as _asyncio

        if not query.strip():
            return ""

        try:
            graphiti = await self._get_graphiti()
            stage1_k = self.top_k_stage1 if use_rerank else top_k

            # 构建多 scope 并行查询任务
            search_tasks = [
                self._search_scoped(graphiti, query, [agent_id], stage1_k, "个人")
            ]
            if team_id:
                search_tasks.append(
                    self._search_scoped(graphiti, query, [f"team:{team_id}"], stage1_k // 2, "团队")
                )
            if tenant_id:
                search_tasks.append(
                    self._search_scoped(graphiti, query, [f"org:{tenant_id}"], stage1_k // 2, "组织")
                )

            scoped_results = await _asyncio.gather(*search_tasks)

            # 合并，去重（按 uuid）
            seen = set()
            all_results = []
            for batch in scoped_results:
                for r in batch:
                    uid = getattr(r, 'uuid', id(r))
                    if uid not in seen:
                        seen.add(uid)
                        all_results.append(r)

            if not all_results:
                return ""

            # 过滤置信度过低的 stale 节点
            all_results = _filter_by_confidence(all_results)

            # Stage 2: LLM 精筛
            if use_rerank and len(all_results) > top_k:
                all_results = await self._rerank_with_llm(query, all_results, top_k)

            if not as_markdown:
                return str(all_results)

            return self._format_as_context(all_results, agent_id)

        except Exception as e:
            return f"<!-- 记忆召回失败: {e} -->"

    async def retrieve_for_session_start(
        self,
        agent_id: str = "aoxia",
        top_k: int = 15,
        team_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
    ) -> str:
        """
        会话开始时的全量相关记忆召回，支持多范围

        Args:
            agent_id: 个人 namespace
            top_k: 召回总条数
            team_id: 传入后同时召回团队记忆
            tenant_id: 传入后同时召回组织知识
        """
        import asyncio as _asyncio

        profile = _get_profile(agent_id)
        queries = profile.get("session_start_queries", ["recent decisions", "active projects"])

        # 构建每个 scope 的召回任务（每个 query × 每个 scope）
        graphiti = await self._get_graphiti()
        per_query_k = max(1, top_k // len(queries))

        tasks = []
        for q in queries:
            tasks.append(self._search_scoped(graphiti, q, [agent_id], per_query_k, "个人"))
            if team_id:
                tasks.append(self._search_scoped(graphiti, q, [f"team:{team_id}"], per_query_k // 2, "团队"))
            if tenant_id:
                tasks.append(self._search_scoped(graphiti, q, [f"org:{tenant_id}"], per_query_k // 2, "组织"))

        all_batches = await _asyncio.gather(*tasks)

        # 合并去重
        seen = set()
        unique_results = []
        for batch in all_batches:
            for r in batch:
                uid = getattr(r, 'uuid', id(r))
                if uid not in seen:
                    seen.add(uid)
                    unique_results.append(r)

        if not unique_results:
            return ""

        return self._format_as_context(unique_results, agent_id, is_session_start=True)

    def _format_as_context(
        self,
        results: list,
        agent_id: str,
        is_session_start: bool = False,
    ) -> str:
        """把召回结果格式化为 system prompt 可用的 Markdown，多范围结果标注来源"""

        profile = _get_profile(agent_id)
        assistant_name = profile.get("assistant_display_name", "Memocore")
        header = f"## {assistant_name}记忆召回\n" if is_session_start else "## 相关历史记忆\n"
        lines = [header]
        lines.append(f"*来源：{agent_id} 知识图谱 | {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        for i, r in enumerate(results, 1):
            fact = getattr(r, 'fact', None)
            scope_label = getattr(r, '_scope_label', None)
            prefix = f"[{scope_label}] " if scope_label else ""

            if fact:
                lines.append(f"{i}. {prefix}{fact}")
            else:
                lines.append(f"{i}. {prefix}{str(r)[:200]}")

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
