# Memocore 实施计划 — M1 记忆闭环

> 目标：Graphiti 跑通完整记忆写入+召回闭环，替代 memory.md
> 时间：2026-04-03 起，约2-3周
> 注：本文档为早期实施计划，v1.0 已全部完成，以 README.md 为准。

---

## Task 1：项目初始化【30分钟】

**文件**：`pyproject.toml`, `.env.example`, `memocore/__init__.py`

步骤：
1. 写 `pyproject.toml`，依赖：`graphiti-core`, `neo4j`, `python-dotenv`, `openai`
2. 写 `.env.example`：NEO4J_URI / NEO4J_PASSWORD / API Keys
3. 验证：`pip install -e .` 成功，`python -c "import memocore"` 无报错

---

## Task 2：Graphiti 连通性验证【1小时】

步骤：
1. 连接本机 Neo4j（bolt://localhost:7687）
2. 初始化 Graphiti client，指定 agent_id
3. 写入一条测试 episode（一段假对话）
4. 召回：用相关 query 验证能取回刚写入的记忆

验证标准：写入成功 + 召回结果包含写入内容，延迟 < 2s

---

## Task 3：Agent 记忆 Schema 定义【2小时】

**文件**：`memocore/agents/default/schema.py`

定义通用实体类型（Pydantic）：

```python
class GenericPreference(BaseModel):
    """用户偏好"""
    category: str
    content: str
    strength: str  # strong/medium/weak

class GenericDecision(BaseModel):
    """决策记录"""
    topic: str
    conclusion: str
    rationale: str

class GenericTask(BaseModel):
    """任务状态"""
    task_name: str
    status: str
    details: str
```

验证标准：Schema 定义完整，可通过 registry 注册自定义 Agent Profile

---

## Task 4：对话提炼脚本【3小时】

**文件**：`memocore/core/extractor.py`

功能：把一段对话文本 → 提炼实体+关系 → 写入 Graphiti

步骤：
1. 实现 `extract_and_store(conversation: str, agent_id: str)`
2. 用 LLM（Claude/OpenAI）提炼实体，按 Schema 结构化
3. 写入 Graphiti，打时间戳，标 agent_id namespace

---

## Task 5：记忆召回函数【2小时】

**文件**：`memocore/core/retriever.py`

功能：根据当前对话上下文 → 召回最相关历史记忆

步骤：
1. 实现 `retrieve(query: str, agent_id: str, top_k: int = 10) -> str`
2. 混合检索：语义 + 图遍历
3. 输出格式：可直接注入 system prompt 的 Markdown 文本

---

## Task 6：Claude Code hooks 接入【2小时】

**文件**：`memocore/adapters/claude_code/prompt_hook.py`, `stop_hook.py`
配置：`~/.claude/settings.json`

步骤：
1. 写 Stop hook 脚本：读取当前会话摘要，触发 `extract_and_store`
2. 写 Prompt hook 脚本：触发 `retrieve`，把记忆注入 system prompt
3. 配置到 `~/.claude/settings.json`

---

## Task 7：端到端验证【1小时】

步骤：
1. 跑一次真实对话（讨论某个项目）
2. 对话结束 → hook 自动触发 → 记忆写入
3. 新开对话 → 记忆召回 → 验证"记住了"
4. 修改一个判断 → 验证旧记忆作废、新记忆生效

---

## 完成标准（M1 Done）

- [x] Neo4j 连通，Graphiti 读写正常
- [x] 对话结束自动提炼写入
- [x] 新对话开始自动召回注入
- [x] 时序作废机制工作（旧判断不污染新召回）
- [x] 主观感受：比 memory.md 明显更聪明

---

## 后续（已完成）

M1 跑通后，已升级为通用 Agent Profile 注册机制，任何 Agent 可通过 `register_profile()` 注册自定义记忆 Schema。
详见 `memocore/agents/registry.py` 和 README.md。
