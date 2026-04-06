"""
Default Agent Profile

Used when an agent_id has no custom profile registered.
Provides sensible default extraction instructions suitable for any AI Agent.
"""

from pydantic import BaseModel, Field
from typing import Optional


class GenericPreference(BaseModel):
    """User preference, rule, or standard."""
    category: str = Field(description="Preference category: tools/workflow/communication/technical/other")
    rule: str = Field(description="Specific rule or preference, one sentence, directly actionable")
    strength: str = Field(default="medium", description="Strength: strong/medium/weak")


class GenericDecision(BaseModel):
    """A confirmed decision or conclusion."""
    topic: str = Field(description="Decision topic")
    conclusion: str = Field(description="Decision conclusion, one sentence")
    context: Optional[str] = Field(default=None, description="Decision context")


class GenericTask(BaseModel):
    """Task execution record."""
    task_name: str = Field(description="Task name")
    result: str = Field(description="Execution result: success/failed/partial")
    lesson: Optional[str] = Field(default=None, description="Lesson learned")


DEFAULT_ENTITY_TYPES: dict = {
    "GenericPreference": GenericPreference,
    "GenericDecision": GenericDecision,
    "GenericTask": GenericTask,
}

_EXTRACTION_INSTRUCTIONS = {
    "zh": """
从对话中提取以下内容：
1. 用户的偏好、规则和标准 -> GenericPreference
2. 做出的决策和结论 -> GenericDecision
3. 完成的任务和经验教训 -> GenericTask

仅提取明确表述的信息，不要推测。
""",
    "en": """
Extract from the conversation:
1. User preferences, rules, and standards -> GenericPreference
2. Decisions made -> GenericDecision
3. Tasks completed and lessons learned -> GenericTask

Only extract explicitly stated information. Do not infer.
""",
}

_SESSION_START_QUERIES = {
    "zh": [
        "最近的决策和进行中的项目",
        "用户偏好和工作风格",
    ],
    "en": [
        "recent decisions and active projects",
        "user preferences and working style",
    ],
}


def get_default_profile() -> dict:
    """Return default profile with language-aware extraction instructions."""
    from memocore.core.config import get_lang
    lang = get_lang()
    return {
        "user_display_name": "User",
        "assistant_display_name": "Assistant",
        "extraction_instructions": _EXTRACTION_INSTRUCTIONS.get(lang, _EXTRACTION_INSTRUCTIONS["zh"]),
        "session_start_queries": _SESSION_START_QUERIES.get(lang, _SESSION_START_QUERIES["zh"]),
    }


# Static fallback for backward compatibility (uses zh as default)
DEFAULT_PROFILE: dict = {
    "user_display_name": "User",
    "assistant_display_name": "Assistant",
    "extraction_instructions": _EXTRACTION_INSTRUCTIONS["zh"],
    "session_start_queries": _SESSION_START_QUERIES["zh"],
}
