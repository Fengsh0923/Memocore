# MemOS 实施计划 — M1 鳌虾记忆闭环

> 目标：鳌虾 × Graphiti 跑通完整记忆写入+召回闭环，替代 memory.md
> 时间：2026-04-03 起，约2-3周

---

## Task 1：项目初始化【30分钟】

**文件**：`pyproject.toml`, `.env.example`, `memos/__init__.py`

步骤：
1. 写 `pyproject.toml`，依赖：`graphiti-core`, `neo4j`, `python-dotenv`, `openai`
2. 写 `.env.example`：NEO4J_URI / NEO4J_PASSWORD / OPENAI_API_KEY
3. 验证：`pip install -e .` 成功，`python -c "import memos"` 无报错

---

## Task 2：Graphiti 连通性验证【1小时】

**文件**：`examples/aoxia_demo/test_connection.py`

步骤：
1. 连接本机 Neo4j（bolt://localhost:7687）
2. 初始化 Graphiti client，指定 agent_id = "aoxia"
3. 写入一条测试 episode（一段假对话）
4. 召回：用相关 query 验证能取回刚写入的记忆

验证标准：写入成功 + 召回结果包含写入内容，延迟 < 2s

---

## Task 3：鳌虾记忆 Schema 定义【2小时】

**文件**：`memos/agents/aoxia/schema.py`

定义鳌虾专属的实体类型（Pydantic）：

```python
# 示例
class FrankPreference(BaseNode):
    """F哥的偏好和判断标准"""
    category: str  # 技术/写作/管理/审美
    content: str
    strength: str  # strong/medium/weak

class ProjectStatus(BaseNode):
    """项目状态快照"""
    project_name: str
    status: str
    last_action: str

class Judgment(BaseNode):
    """F哥做过的判断或决策"""
    topic: str
    conclusion: str
    context: str
```

验证标准：Schema 定义完整，覆盖 memory.md 现有内容

---

## Task 4：对话提炼脚本【3小时】

**文件**：`memos/core/extractor.py`

功能：把一段对话文本 → 提炼实体+关系 → 写入 Graphiti

步骤：
1. 实现 `extract_and_store(conversation: str, agent_id: str)`
2. 用 LLM（Claude/OpenAI）提炼实体，按 Schema 结构化
3. 写入 Graphiti，打时间戳，标 agent_id namespace

验证标准：输入一段真实的鳌虾对话，能正确提炼出 F哥偏好/项目状态/判断

---

## Task 5：记忆召回函数【2小时】

**文件**：`memos/core/retriever.py`

功能：根据当前对话上下文 → 召回最相关历史记忆

步骤：
1. 实现 `retrieve(query: str, agent_id: str, top_k: int = 10) -> str`
2. 混合检索：语义 + 图遍历
3. 输出格式：可直接注入 system prompt 的 Markdown 文本

验证标准：用一句新的话，能召回相关的历史判断或偏好

---

## Task 6：Claude Code hooks 接入【2小时】

**文件**：`memos/adapters/claude_code/hooks.py`
配置：`~/.claude/settings.json`（Stop hook）

步骤：
1. 写 Stop hook 脚本：读取当前会话摘要，触发 `extract_and_store`
2. 写会话开始脚本：触发 `retrieve`，把记忆注入 system prompt
3. 配置到 `~/.claude/settings.json`

验证标准：一次完整对话结束后，Neo4j 里能看到新的节点和关系

---

## Task 7：端到端验证【1小时】

**文件**：`examples/aoxia_demo/e2e_test.md`

步骤：
1. 跑一次真实的鳌虾对话（讨论某个项目）
2. 对话结束 → hook 自动触发 → 记忆写入
3. 新开对话 → 记忆召回 → 验证"记住了"
4. 修改一个判断（F哥改变了某个偏好）→ 验证旧记忆作废、新记忆生效

验证标准：F哥能感受到"这个虾记住我说过的事了"

---

## 完成标准（M1 Done）

- [ ] Neo4j 连通，Graphiti 读写正常
- [ ] 鳌虾对话结束自动提炼写入
- [ ] 新对话开始自动召回注入
- [ ] 时序作废机制工作（旧判断不污染新召回）
- [ ] F哥主观感受：比 memory.md 明显更聪明

---

## 下一步（M2 预告）

M1 跑通后，把 Task 2-6 模板化，复制到甜虾、龙虾、麦虾。
差异只在 Schema 定义（每个虾的记忆结构不同）。
