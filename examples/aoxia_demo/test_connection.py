#!/usr/bin/env python3
"""
T2: Graphiti × Neo4j 连通性验证
验证：写入一条 episode → 召回 → 确认链路通
"""

import asyncio
import sys
import os
from pathlib import Path
from datetime import datetime

# 加载 .env
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    from dotenv import load_dotenv
    load_dotenv(env_path)

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")


async def test_connection():
    print("=" * 50)
    print("MemOS T2: Graphiti × Neo4j 连通性验证")
    print("=" * 50)

    # Step 1: 验证 Neo4j 直连
    print("\n[1/4] 验证 Neo4j 连接...")
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    print(f"  ✅ Neo4j 连通: {NEO4J_URI}")
    driver.close()

    # Step 2: 初始化 Graphiti
    print("\n[2/4] 初始化 Graphiti client...")
    from graphiti_core import Graphiti
    from graphiti_core.llm_client.openai_client import OpenAIClient
    from graphiti_core.embedder.openai import OpenAIEmbedder

    if not OPENAI_API_KEY:
        print("  ⚠️  OPENAI_API_KEY 未设置，跳过 LLM 测试，只测图数据库连接")
        print("\n✅ T2 基础连通验证通过（无 LLM）")
        print("下一步：在 .env 填入 OPENAI_API_KEY，再跑完整流程")
        return

    graphiti = Graphiti(
        uri=NEO4J_URI,
        user=NEO4J_USER,
        password=NEO4J_PASSWORD,
    )
    await graphiti.build_indices_and_constraints()
    print("  ✅ Graphiti 初始化，索引建立完成")

    # Step 3: 写入一条测试 episode
    print("\n[3/4] 写入测试记忆...")
    test_content = """
    F哥今天决定启动 MemOS 项目，用于 Agent 记忆管理。
    技术选型：Graphiti + Neo4j。
    第一个验证场景：鳌虾记忆闭环。
    判断：企业 Agent 记忆差是 AI 落地失败的最隐性卡点。
    """
    await graphiti.add_episode(
        name=f"memos_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        episode_body=test_content,
        source_description="MemOS 连通性测试",
        reference_time=datetime.now(),
        group_id="aoxia",
    )
    print("  ✅ Episode 写入成功")

    # Step 4: 召回验证
    print("\n[4/4] 召回验证...")
    results = await graphiti.search(
        query="MemOS 项目技术选型",
        group_ids=["aoxia"],
        num_results=3,
    )
    print(f"  ✅ 召回 {len(results)} 条结果")
    for i, r in enumerate(results, 1):
        print(f"  [{i}] {str(r)[:100]}...")

    await graphiti.close()

    print("\n" + "=" * 50)
    print("✅ T2 全部通过 — Graphiti × Neo4j 链路正常")
    print("=" * 50)


if __name__ == "__main__":
    asyncio.run(test_connection())
