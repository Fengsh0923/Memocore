"""
Memocore Locale — bilingual prompt templates and UI strings.

All user-facing text and LLM prompts are centralized here.
Controlled by MEMOCORE_LANG env var ("zh" or "en", default "zh").

Usage:
    from memocore.core.locale import t
    prompt = t("dream.consolidate_duplicate", entity_name="foo", details="...")
"""

from memocore.core.config import get_lang

# ── String tables ─────────────────────────────────────────────────────────────

_STRINGS: dict[str, dict[str, str]] = {
    # ── Dream Phase 3: Consolidate ──
    "dream.consolidate_duplicate": {
        "zh": """你是记忆图谱清洁工。以下是同名节点的详情：

{context}

节点详情：
{details}

请决定：
1. merge — 合并为一个，指定保留的 uuid（最完整的那个）
2. skip — 数据不足，跳过

仅返回 JSON：{{"action": "merge或skip", "keep_uuid": "...", "reason": "..."}}""",
        "en": """You are a memory graph cleaner. Below are details of nodes with the same name:

{context}

Node details:
{details}

Decide:
1. merge — merge into one, specify the uuid to keep (the most complete one)
2. skip — insufficient data, skip

Return only JSON: {{"action": "merge or skip", "keep_uuid": "...", "reason": "..."}}""",
    },

    "dream.consolidate_conflict": {
        "zh": """你是记忆图谱清洁工。以下是两条可能矛盾的关系：

{context}

请决定：
1. keep_latest — 保留时间更新的那条（通常是正确的），删除旧的
2. keep_both — 两条含义不同，都保留
3. skip — 数据不足

仅返回 JSON：{{"action": "keep_latest或keep_both或skip", "delete_uuid": "...", "reason": "..."}}""",
        "en": """You are a memory graph cleaner. Below are two potentially contradictory relationships:

{context}

Decide:
1. keep_latest — keep the newer one (usually correct), delete the older one
2. keep_both — both have different meanings, keep both
3. skip — insufficient data

Return only JSON: {{"action": "keep_latest or keep_both or skip", "delete_uuid": "...", "reason": "..."}}""",
    },

    # ── Dream Phase 7: Compile ──
    "dream.compile_entity": {
        "zh": """你是知识编译器。将以下关于「{entity_name}」的碎片信息编译成一个结构化的知识页面。

实体类型: {entity_type}
当前置信度: {confidence_label}

碎片信息（{fact_count} 条）:
{facts_text}

要求:
1. 合并重复信息，去除冗余
2. 如有矛盾，以最新的为准，旧的标注为「存疑」
3. 按主题分段（如：概述、偏好、关系、近期动态）
4. 用简洁的 Markdown 格式
5. 不要编造碎片中没有的信息
6. 总长度控制在 500 字以内""",
        "en": """You are a knowledge compiler. Compile the following fragments about "{entity_name}" into a structured knowledge page.

Entity type: {entity_type}
Current confidence: {confidence_label}

Fragments ({fact_count} items):
{facts_text}

Requirements:
1. Merge duplicate information, remove redundancy
2. If contradictions exist, prefer the latest; mark older ones as "disputed"
3. Organize by topic (e.g.: overview, preferences, relationships, recent activity)
4. Use concise Markdown format
5. Do not fabricate information not in the fragments
6. Keep total length under 500 words""",
    },

    "dream.compile_overview": {
        "zh": """你是知识编译器。根据以下已编译的实体页面列表，写一段 200 字以内的总体概述，描述这个 Agent 记忆中的核心人物、关注领域和重要决策。

已编译页面（{page_count} 个）:
{page_index}

要求: 简洁，不要罗列，提炼模式和核心信息。""",
        "en": """You are a knowledge compiler. Based on the compiled entity pages listed below, write a summary (under 200 words) describing the core people, focus areas, and key decisions in this agent's memory.

Compiled pages ({page_count}):
{page_index}

Requirements: Be concise, do not enumerate, extract patterns and core information.""",
    },

    # ── Retriever ──
    "retriever.rerank_prompt": {
        "zh": "你是记忆召回助手。用户当前的问题/上下文是：\n\n\"{query}\"\n\n以下是从知识图谱召回的候选记忆（共{count}条）：\n\n{candidates_text}\n\n请从中选出最相关的 {top_k} 条，按相关性从高到低排序。\n仅返回 JSON 数组，包含选中条目的索引号，例如：[3, 0, 7, 1, 5]",
        "en": "You are a memory recall assistant. The user's current question/context is:\n\n\"{query}\"\n\nBelow are candidate memories retrieved from the knowledge graph ({count} items):\n\n{candidates_text}\n\nSelect the {top_k} most relevant items, ranked by relevance (highest first).\nReturn only a JSON array of selected indices, e.g.: [3, 0, 7, 1, 5]",
    },

    "retriever.rerank_titles_prompt": {
        "zh": "从以下知识页面标题中，选出与问题最相关的 {top_k} 个。\n\n问题: {query}\n\n页面列表:\n{index}\n\n仅返回 JSON 数组，包含选中页面的序号（0-indexed），如: [0, 3, 5]",
        "en": "From the knowledge page titles below, select the {top_k} most relevant to the question.\n\nQuestion: {query}\n\nPage list:\n{index}\n\nReturn only a JSON array of selected page indices (0-indexed), e.g.: [0, 3, 5]",
    },

    # ── Privacy LLM check ──
    "privacy.llm_check_prompt": {
        "zh": "你是隐私审查助手。以下是一段待存储的对话摘要。\n请判断其中是否包含以下类型的敏感信息：\n  - 密码、密钥、token（未被 [REDACTED] 标记的）\n  - 个人身份信息（姓名+地址+身份证的组合）\n  - 信用卡号\n\n对话内容：\n{text}\n\n请仅回复：SAFE 或 SENSITIVE（不要解释）",
        "en": "You are a privacy review assistant. Below is a conversation summary to be stored.\nDetermine if it contains the following types of sensitive information:\n  - Passwords, keys, tokens (not already [REDACTED])\n  - Personally identifiable information (name+address+ID combinations)\n  - Credit card numbers\n\nConversation:\n{text}\n\nReply only: SAFE or SENSITIVE (no explanation)",
    },

    # ── UI strings ──
    "ui.scope_personal": {"zh": "个人", "en": "Personal"},
    "ui.scope_team": {"zh": "团队", "en": "Team"},
    "ui.scope_org": {"zh": "组织", "en": "Org"},

    "ui.compiled_memory_header": {
        "zh": "## 相关记忆（已编译知识）\n*{now}*\n",
        "en": "## Relevant Memory (Compiled Knowledge)\n*{now}*\n",
    },
    "ui.session_start_header": {
        "zh": "## 记忆背景（已编译知识）\n*{now}*\n\n",
        "en": "## Memory Context (Compiled Knowledge)\n*{now}*\n\n",
    },
    "ui.recall_header_session": {
        "zh": "## {name}记忆召回\n",
        "en": "## {name} Memory Recall\n",
    },
    "ui.recall_header_normal": {
        "zh": "## 相关历史记忆\n",
        "en": "## Related Historical Memory\n",
    },
    "ui.recall_source": {
        "zh": "*来源：{agent_id} 知识图谱 | {now}*\n",
        "en": "*Source: {agent_id} knowledge graph | {now}*\n",
    },
    "ui.overview_label": {
        "zh": "总览",
        "en": "Overview",
    },
    "ui.hook_memory_start": {
        "zh": "\n--- Memocore 历史记忆（自动召回）---\n",
        "en": "\n--- Memocore Historical Memory (auto-recalled) ---\n",
    },
    "ui.hook_memory_end": {
        "zh": "\n--- 历史记忆结束 ---\n",
        "en": "\n--- End of historical memory ---\n",
    },

    # ── Dream report ──
    "dream.stale_reason": {
        "zh": "孤立节点超过 {days} 天",
        "en": "Orphan node older than {days} days",
    },
}


def t(key: str, **kwargs) -> str:
    """
    Get a localized string by key, with optional format arguments.

    Args:
        key: dot-separated string key (e.g. "dream.compile_entity")
        **kwargs: format arguments for the template

    Returns:
        Formatted localized string
    """
    entry = _STRINGS.get(key)
    if entry is None:
        return f"[MISSING_LOCALE:{key}]"
    lang = get_lang()
    template = entry.get(lang, entry.get("en", f"[NO_LANG:{key}:{lang}]"))
    if kwargs:
        return template.format(**kwargs)
    return template
