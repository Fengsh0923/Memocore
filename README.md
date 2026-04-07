# Memocore — AI Agent 持久化记忆层

> 让你的 AI Agent 拥有持久、自愈的记忆。
> 两条路径可选：**`memocore`**（Graphiti + Neo4j，图谱 + 向量召回）或
> **[`memocore.lite`](memocore/lite/README.md)**（SQLite + FTS5，Karpathy 风格 markdown wiki）。
> 支持 Claude Code Hooks / MCP / IM Bridge 多种接入方式。

---

## 两条路径怎么选？

| | `memocore` (full) | `memocore.lite` |
|---|---|---|
| 核心思路 | LLM 提取实体 → 图谱 + 向量召回 | Karpathy 风格：LLM 编译 markdown → FTS5 召回 |
| 存储 | Neo4j + Graphiti + 向量索引 | 一个 SQLite 文件 |
| 外部依赖 | Neo4j / Graphiti / embedder / rerank LLM | Python stdlib `sqlite3` |
| 写入路径的 LLM 调用 | 多次（extraction + embedding + rerank） | 零 |
| 代码规模 | ~3000+ LOC | ~600 LOC |
| 可调试性 | 需要 Neo4j Browser + 向量工具 | `sqlite3 agent.db` 命令行 |
| 最擅长 | 跨 entity 的语义相似（改写的概念能召回） | LLM 维护的 markdown 页的词汇召回 |
| 适合规模 | 数百万 entity 的跨企业查询 | 数千 agent × 数千页 / agent |

- 默认的图谱 + 向量路径 → 继续往下看本 README
- 更轻的 Karpathy 风格 → 看 [`memocore/lite/README.md`](memocore/lite/README.md)

两条路径可以共存 — `memocore.lite` 是独立子模块，不 import `memocore.core`
的任何东西，所以可以只装 lite（`pip install memocore`，不装 `[legacy]` extras），
也可以两条路径都启用。

---

## 核心问题

每次开始新对话，AI Agent 都会忘记一切——你的偏好、项目背景、上周的决策。
**Memocore** 是解决这个问题的通用记忆层。

## Memocore 做什么

- **自动提取**: 对话结束后，LLM 自动从对话中提取结构化知识实体
- **自动召回**: 每条消息前，自动检索相关历史记忆注入上下文
- **自我维护**: 定期 Dream 巩固——去重、冲突解决、过期清理、知识编译
- **多租户隔离**: 每个 `agent_id` 是独立命名空间，4000+ Agent 可共享同一 Neo4j 实例
- **多语言**: 默认中文，所有 LLM 提示和 UI 字符串支持 zh/en 切换 (`MEMOCORE_LANG`)

## 架构

```
┌────────────────────────────────────────────────────────────┐
│                      AI Agent 对话                         │
│                                                            │
│  每条消息 ──► prompt_hook / MCP recall                     │
│       │         (检索记忆 → 注入上下文)                     │
│       ▼                                                    │
│   [ 对话进行中 ]  ◄── 历史记忆自动注入                      │
│       │                                                    │
│  对话结束 ──► stop_hook / MCP store                        │
│                 (提取知识 → 写入图谱)                       │
│                 + Dream 巩固 (每 ~5 次对话)                 │
└────────────────────────────────────────────────────────────┘
                        │
            ┌───────────▼───────────┐
            │   Graphiti Engine     │
            │  时序知识图谱 + RAG    │
            └───────────┬───────────┘
                        │
            ┌───────────▼───────────┐
            │   Neo4j Graph DB      │
            │  节点 + 关系 + 向量    │
            └───────────────────────┘
```

**三个核心循环:**

| 循环 | 触发时机 | 功能 |
|------|---------|------|
| **写入** | 对话结束 | 对话 → LLM 提取 → 结构化实体 → Neo4j |
| **召回** | 每条用户消息 | 查询 → 向量搜索 → LLM 重排 → 注入上下文 |
| **Dream** | 每 ~5 次对话 (异步) | 去重 → 冲突解决 → 过期清理 → 知识编译 → 健康报告 |

## 主要特性

### 插件化 Agent Profile

通过 Registry 注册自定义 Agent 配置，每个 Agent 可以有独立的实体类型和提取策略:

```python
from memocore.agents.registry import register_profile

register_profile("my-agent", {
    "user_display_name": "Alice",
    "assistant_display_name": "Bot",
    "extraction_instructions": "从对话中提取用户偏好和决策...",
    "session_start_queries": ["最近的项目进展", "用户偏好"],
}, entity_types=MY_ENTITY_TYPES)
```

未注册的 Agent 自动使用默认 Profile（支持 GenericPreference / GenericDecision / GenericTask 三种通用实体类型）。

### 两阶段召回

```
用户提问 → 向量 + 图谱混合搜索 (top 20) → LLM 重排序 (top 5) → 注入上下文
```

LLM 重排可关闭 (`use_rerank=False`) 以降低延迟。首条消息自动触发全量召回（编译知识页 + 实体概览）。

### Dream 巩固 (8 阶段)

知识图谱会随时间退化——重复节点堆积、事实矛盾、孤立节点膨胀。Dream 自动维护:

| 阶段 | 功能 |
|------|------|
| Phase 1-2 | 扫描图谱，识别重复/矛盾/孤立候选 |
| Phase 3-4 | LLM 决策: 合并/保留最新/删除/跳过，执行变更 |
| Phase 5-6 | TTL 过期清理，置信度衰减/恢复 |
| Phase 7 | 知识编译: 将碎片事实编译为结构化知识页 (Karpathy LLM Wiki) |
| Phase 8 | 健康检查: 生成 Lint 报告 (矛盾/孤立/缺失/过期) |

### 多接入方式

| 适配器 | 场景 | 说明 |
|--------|------|------|
| **Claude Code Hooks** | 个人开发 | prompt_hook + stop_hook，零配置自动记忆 |
| **MCP Server** | stdio / HTTP | 支持 Claude Desktop, Cursor 等 MCP 客户端；HTTP 模式支持多租户 |
| **IM Bridge** | Slack/Teams 等 | bridge_read + bridge_write，IM 消息自动记忆 |

### 隐私保护

- 内置正则规则自动脱敏 (API Key, 密码, 手机号, 身份证号等)
- 可选 LLM 二次审查
- 黑名单关键词跳过
- 通过 `MEMOCORE_PRIVACY_ENABLED` 开关

### 生产级特性 (v1.0)

- **并发安全**: session counter 使用 `fcntl` 文件锁；lazy-init 使用 `asyncio.Lock` / `threading.Lock`
- **LLM 超时**: Dream 所有 LLM 调用 60s 超时，不会无限阻塞
- **输入校验**: 所有入口 `validate_agent_id()` 防路径穿越和注入
- **安全路径**: 文件名使用 `make_safe_agent_key()` (SHA-256 哈希) 防碰撞
- **日志隔离**: 库模块不调用 `logging.basicConfig()`，不污染宿主应用日志
- **容器兼容**: `Path.home()` 失败时自动降级到 `/tmp/.memocore`

---

## 快速开始

### 前置条件

- Python >= 3.10
- Neo4j (本地或云端) — [Neo4j Desktop](https://neo4j.com/download/)
- Anthropic 或 OpenAI API Key

### 安装

```bash
git clone https://github.com/Fengsh0923/memocore.git
cd Memocore
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 配置

```bash
# 交互式配置向导
memocore init

# 或手动
cp .env.example .env
# 编辑 .env 填入 API Key 和 Neo4j 信息
```

关键配置项:
```env
NEO4J_URI=bolt://localhost:7687
NEO4J_PASSWORD=your_password
ANTHROPIC_API_KEY=sk-ant-...
MEMOCORE_AGENT_ID=my-agent
MEMOCORE_LANG=zh          # zh (默认) 或 en
```

### 验证连接

```bash
memocore stats
```

### API 用法

```python
import asyncio
from memocore.core.extractor import extract_and_store
from memocore.core.retriever import MemoryRetriever

async def main():
    # 写入对话记忆
    result = await extract_and_store(
        conversation="User: 我们决定用 Graphiti 做记忆层\nAssistant: 好的，记录这个决策",
        agent_id="my-agent",
    )
    print(result)  # {'success': True, 'entities_extracted': 1, ...}

    # 召回相关记忆
    retriever = MemoryRetriever()
    context = await retriever.retrieve(
        query="记忆层的技术选型",
        agent_id="my-agent",
        top_k=5,
    )
    print(context)  # Markdown 格式，可直接注入 system prompt
    await retriever.close()

asyncio.run(main())
```

### Dream 巩固

```bash
# 试运行 (不修改图谱)
python -m memocore.core.dream --agent-id my-agent --dry-run

# 正式运行
python -m memocore.core.dream --agent-id my-agent
```

### CLI 工具

```bash
memocore list                          # 列出最近的记忆节点
memocore search "通知规则"              # 语义搜索
memocore browse                        # 浏览编译知识页
memocore browse --report               # 查看 Lint 健康报告
memocore export --format md -o mem.md  # 导出
memocore delete UUID                   # 删除节点
memocore privacy-scan "测试文本"        # 预览隐私过滤
```

---

## Claude Code Hook 集成

在 `~/.claude/settings.json` 中添加:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [{
          "type": "command",
          "command": "python3 -m memocore.adapters.claude_code.prompt_hook",
          "timeout": 25
        }]
      }
    ],
    "Stop": [
      {
        "hooks": [{
          "type": "command",
          "command": "python3 -m memocore.adapters.claude_code.stop_hook",
          "timeout": 60
        }]
      }
    ]
  }
}
```

## MCP Server

```bash
# stdio 模式 (个人使用，配合 Claude Desktop)
memocore-mcp

# HTTP 模式 (企业多租户)
memocore-mcp --transport http --host 0.0.0.0 --port 8765
```

---

## 项目结构

```
Memocore/
├── memocore/
│   ├── core/
│   │   ├── config.py         # 集中配置 + 输入校验
│   │   ├── extractor.py      # 对话 → 知识提取
│   │   ├── retriever.py      # 两阶段记忆召回
│   │   ├── dream.py          # 8 阶段记忆巩固
│   │   ├── embedder.py       # 向量嵌入 (OpenAI / fastembed / sentence-transformers)
│   │   ├── llm_adapter.py    # LLM 调用层 (Anthropic + OpenAI 双 fallback)
│   │   ├── privacy.py        # 隐私过滤
│   │   ├── graphiti_factory.py # Graphiti 实例工厂
│   │   └── locale.py         # 双语字符串表
│   ├── agents/
│   │   ├── default/schema.py # 默认实体类型和 Profile
│   │   └── registry.py       # Agent Profile 注册表
│   ├── adapters/
│   │   ├── claude_code/      # Claude Code Hooks (prompt_hook + stop_hook)
│   │   ├── mcp/server.py     # MCP Server (stdio + HTTP)
│   │   └── bridge/           # IM Bridge (bridge_read + bridge_write)
│   └── cli/main.py           # CLI 工具
├── tests/                    # 69 个单元测试
└── pyproject.toml
```

---

## 自定义 Agent

1. 定义 Pydantic 实体类型:

```python
from pydantic import BaseModel, Field

class ProjectDecision(BaseModel):
    topic: str = Field(description="决策主题")
    conclusion: str = Field(description="决策结论")
    rationale: str = Field(description="决策理由")
```

2. 注册 Profile:

```python
from memocore.agents.registry import register_profile

register_profile("my-agent", {
    "extraction_instructions": "提取项目决策和技术选型...",
    "session_start_queries": ["最近的项目决策", "技术选型"],
}, entity_types={"ProjectDecision": ProjectDecision})
```

3. 设置环境变量: `MEMOCORE_AGENT_ID=my-agent`

每个 `agent_id` 在 Neo4j 中是独立命名空间，多个 Agent 共享同一数据库不会相互干扰。

---

## 致谢

- **[Graphiti](https://github.com/getzep/graphiti)** (Zep AI) — 时序知识图谱引擎，Memocore 的核心基础设施
- **[Claude Code](https://claude.ai/code)** (Anthropic) — Hook 系统使无摩擦集成成为可能；Dream 巩固机制受其内部记忆架构启发
- **[Neo4j](https://neo4j.com)** — 图 + 向量数据库，支撑混合检索

## License

MIT

---

*Built by [Frank Shen](https://github.com/Frankshen923)*
