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
from memocore.core.llm_adapter import rerank as llm_rerank, chat_complete
from memocore.core.graphiti_factory import build_graphiti
from memocore.agents.aoxia.schema import AOXIA_PROFILE


async def llm_rerank_titles(query: str, pages: list[dict], top_k: int) -> list[dict]:
    """用 LLM 从 CompiledPage 标题列表中挑选与 query 最相关的页面"""
    import json as _json

    index = "\n".join(
        f"{i}. {p['title']}" for i, p in enumerate(pages)
    )
    prompt = f"""从以下知识页面标题中，选出与问题最相关的 {top_k} 个。

问题: {query}

页面列表:
{index}

仅返回 JSON 数组，包含选中页面的序号（0-indexed），如: [0, 3, 5]"""

    content = await chat_complete(prompt=prompt, max_tokens=100, temperature=0.0, json_mode=True)
    indices = _json.loads(content)
    return [pages[i] for i in indices if 0 <= i < len(pages)]


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
    记忆召回器 v3

    recall 优先级:
    1. CompiledPage（Dream 编译的结构化知识）→ 直接 Neo4j 查询，命中即返回
    2. Graphiti 语义+图遍历混合检索 → 碎片 facts 降级路径

    CompiledPage 是 Karpathy LLM Wiki 理念的实现：
    知识被编译一次（Dream Phase 7），recall 时直接读取已编译的页面，
    而非每次从碎片 facts 重新推导。
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
        self._driver = None

    async def _get_neo4j_driver(self):
        """获取 Neo4j 驱动（用于 CompiledPage 直接查询，不走 Graphiti）"""
        if self._driver is None:
            from neo4j import AsyncGraphDatabase
            self._driver = AsyncGraphDatabase.driver(
                self.neo4j_uri, auth=(self.neo4j_user, self.neo4j_password)
            )
        return self._driver

    async def _recall_compiled_pages(
        self, query: str, group_ids: list[str], top_k: int = 5
    ) -> Optional[str]:
        """
        从 CompiledPage 节点中检索相关页面。
        使用 Neo4j 全文索引或 title 模糊匹配。
        返回编译好的知识文本，或 None（无 CompiledPage 时降级到 Graphiti）。
        """
        driver = await self._get_neo4j_driver()

        all_pages = []
        async with driver.session() as session:
            for gid in group_ids:
                # 查询该 group 的所有 CompiledPage
                q = """
                MATCH (p:CompiledPage {group_id: $gid})
                WHERE p.page_type = 'entity' AND p.stale = false
                RETURN p.title AS title, p.content AS content,
                       p.confidence AS confidence, p.source_count AS source_count,
                       p.compiled_at AS compiled_at
                ORDER BY p.source_count DESC
                """
                r = await session.run(q, gid=gid)
                pages = await r.data()
                for p in pages:
                    # 标记 scope
                    if gid.startswith("team:"):
                        p["_scope_label"] = "团队"
                    elif gid.startswith("org:"):
                        p["_scope_label"] = "组织"
                    else:
                        p["_scope_label"] = "个人"
                all_pages.extend(pages)

        if not all_pages:
            return None  # 没有 CompiledPage，降级

        # 构建 index（标题列表），让 LLM 选择相关页面
        # 页面少时（<= top_k）直接全部返回，不需要 LLM 筛选
        if len(all_pages) <= top_k:
            relevant_pages = all_pages
        else:
            # 用 LLM 从 index 中挑选与 query 最相关的页面
            index_text = "\n".join(
                f"{i+1}. [{p.get('_scope_label', '')}] {p['title']} (facts: {p.get('source_count', 0)})"
                for i, p in enumerate(all_pages)
            )
            try:
                selection = await llm_rerank_titles(query, all_pages, top_k)
                relevant_pages = selection
            except Exception:
                # LLM 筛选失败时取 source_count 最高的
                relevant_pages = all_pages[:top_k]

        if not relevant_pages:
            return None

        # 格式化输出
        now = datetime.now().strftime('%Y-%m-%d %H:%M')
        lines = [f"## 相关记忆（已编译知识）\n*{now}*\n"]
        for p in relevant_pages:
            scope = p.get("_scope_label", "")
            prefix = f"[{scope}] " if scope else ""
            lines.append(f"### {prefix}{p['title']}\n")
            lines.append(p.get("content", "") + "\n")
        return "\n".join(lines)

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

        # ── 优先尝试 CompiledPage（编译后的结构化知识）──
        group_ids = [agent_id]
        if team_id:
            group_ids.append(f"team:{team_id}")
        if tenant_id:
            group_ids.append(f"org:{tenant_id}")

        try:
            compiled = await self._recall_compiled_pages(query, group_ids, top_k)
            if compiled:
                return compiled  # 命中编译知识，直接返回
        except Exception as e:
            pass  # CompiledPage 查询失败，降级到 Graphiti

        # ── 降级：Graphiti 碎片 facts 检索 ──
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

        # ── 优先尝试 CompiledPage（overview + 全部 entity 页面）──
        group_ids = [agent_id]
        if team_id:
            group_ids.append(f"team:{team_id}")
        if tenant_id:
            group_ids.append(f"org:{tenant_id}")

        try:
            driver = await self._get_neo4j_driver()
            pages_content = []
            async with driver.session() as session:
                for gid in group_ids:
                    # overview 页面
                    ov_q = """
                    MATCH (p:CompiledPage {group_id: $gid, title: '__overview__'})
                    RETURN p.content AS content
                    """
                    r = await session.run(ov_q, gid=gid)
                    rec = await r.single()
                    if rec and rec["content"]:
                        scope = "个人" if not gid.startswith(("team:", "org:")) else ("团队" if gid.startswith("team:") else "组织")
                        pages_content.append(f"### [{scope}] 总览\n\n{rec['content']}")

                    # 所有 entity 页面（top_k 个）
                    ep_q = """
                    MATCH (p:CompiledPage {group_id: $gid})
                    WHERE p.page_type = 'entity' AND p.stale = false
                    RETURN p.title AS title, p.content AS content
                    ORDER BY p.source_count DESC
                    LIMIT $limit
                    """
                    r = await session.run(ep_q, gid=gid, limit=top_k // max(1, len(group_ids)))
                    entity_pages = await r.data()
                    for ep in entity_pages:
                        scope = "个人" if not gid.startswith(("team:", "org:")) else ("团队" if gid.startswith("team:") else "组织")
                        pages_content.append(f"### [{scope}] {ep['title']}\n\n{ep.get('content', '')}")

            if pages_content:
                now = datetime.now().strftime('%Y-%m-%d %H:%M')
                return f"## 记忆背景（已编译知识）\n*{now}*\n\n" + "\n\n".join(pages_content)
        except Exception:
            pass  # 降级到 Graphiti

        # ── 降级：Graphiti 碎片 facts 检索 ──
        profile = _get_profile(agent_id)
        queries = profile.get("session_start_queries", ["recent decisions", "active projects"])

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
        if self._driver:
            await self._driver.close()
            self._driver = None


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
