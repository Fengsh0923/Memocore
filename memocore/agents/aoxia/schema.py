"""
鳌虾记忆 Schema — T3 (v2: 加入执行元数据)
基于 F哥 memory.md 的内容结构，定义 Graphiti 实体类型

设计原则：
- 每个实体类型对应 memory.md 里一类信息
- 用 Pydantic BaseModel（Graphiti EntityNode 格式）
- 字段尽量语义清晰，供 LLM 提炼时直接映射
- v2 新增：when_to_save / how_to_apply 执行元数据（借鉴 Claude Code Dream 机制）
  * when_to_save：触发写入这条记忆的场景，LLM提炼时自动填充
  * how_to_apply：下次召回后如何使用，让记忆可直接驱动行动
"""

from pydantic import BaseModel, Field
from typing import Optional


# ─── 核心实体类型 ──────────────────────────────────────────────────────────────

class FrankPreference(BaseModel):
    """
    F哥的偏好、判断标准、审美习惯
    来源：frank_standards.md 里的各类判断条目
    示例：'文件 < 1KB 视为空文件' / '飞书通知发单聊不发群聊'
    """
    category: str = Field(
        description="偏好类别：技术/写作/管理/审美/工具/沟通/产出质量"
    )
    rule: str = Field(
        description="具体规则或偏好，一句话描述，可直接执行"
    )
    trigger_scene: Optional[str] = Field(
        default=None,
        description="触发这条规则的场景或上下文"
    )
    strength: str = Field(
        default="strong",
        description="规则强度：strong(不可违反) / medium(默认遵守) / weak(参考)"
    )
    when_to_save: Optional[str] = Field(
        default=None,
        description="触发保存这条偏好的场景，如：F哥明确表达不满或纠正时"
    )
    how_to_apply: Optional[str] = Field(
        default=None,
        description="召回后如何应用，如：每次产出文件前检查此规则是否适用"
    )


class ProjectStatus(BaseModel):
    """
    项目状态快照
    来源：各 project_*.md 文件里的项目信息
    示例：MemOS 项目当前在 Phase 4，鳌虾已跑通 T2
    """
    project_name: str = Field(description="项目名称，如 MemOS / OpenTiger / 飞虾队")
    current_phase: Optional[str] = Field(
        default=None,
        description="当前阶段，如 Phase4-Writing Plans / M1-连通验证"
    )
    status: str = Field(
        description="状态：active(进行中) / blocked(卡住) / done(完成) / paused(暂停)"
    )
    last_action: Optional[str] = Field(
        default=None,
        description="最近一次做了什么"
    )
    next_step: Optional[str] = Field(
        default=None,
        description="下一步行动"
    )
    key_decisions: Optional[str] = Field(
        default=None,
        description="重要技术/产品决策，逗号分隔"
    )
    when_to_save: Optional[str] = Field(
        default=None,
        description="触发保存的场景，如：对话中提到项目进展、阶段切换时"
    )
    how_to_apply: Optional[str] = Field(
        default=None,
        description="召回后如何应用，如：新对话开始时检查项目状态，主动告知 next_step"
    )


class Judgment(BaseModel):
    """
    F哥做过的判断、结论、决策
    来源：对话中 F哥明确表态的内容
    示例：'MemOS 技术选型：Neo4j + Graphiti，不造轮子'
    """
    topic: str = Field(description="判断的主题，如 MemOS技术选型 / 记忆存储方案")
    conclusion: str = Field(description="具体结论，一句话，可被后续对话直接引用")
    context: Optional[str] = Field(
        default=None,
        description="做出这个判断时的背景和理由"
    )
    confidence: str = Field(
        default="confirmed",
        description="确定程度：confirmed(明确) / tentative(倾向) / exploring(探索中)"
    )
    when_to_save: Optional[str] = Field(
        default=None,
        description="触发保存的场景，如：F哥说'好虾'、'对'、'就这么定了'之后"
    )
    how_to_apply: Optional[str] = Field(
        default=None,
        description="召回后如何应用，如：遇到同类话题时直接引用此结论，无需重新讨论"
    )


class AgentConfig(BaseModel):
    """
    飞虾队成员配置和状态
    来源：project_flying_shrimp.md
    示例：甜虾负责市场日报，每天03:00跑，SSH via mDNS
    """
    agent_name: str = Field(description="虾的名字：鳌虾/甜虾/麦虾/龙虾/虎虾")
    role: str = Field(description="职责描述")
    schedule: Optional[str] = Field(default=None, description="定时任务，如 每天03:00")
    access_method: Optional[str] = Field(
        default=None,
        description="访问方式，如 SSH via FrankshendeMac-mini.local"
    )
    known_issues: Optional[str] = Field(
        default=None,
        description="已知问题或注意事项"
    )
    when_to_save: Optional[str] = Field(
        default=None,
        description="触发保存的场景，如：某虾配置变更、新虾上线、cron 任务调整时"
    )
    how_to_apply: Optional[str] = Field(
        default=None,
        description="召回后如何应用，如：派单前先查此虾的 known_issues 和 access_method"
    )


class TaskRecord(BaseModel):
    """
    任务执行记录——鳌虾派出去的任务和结果
    来源：日常派单、巡检、任务回调
    示例：'20260401 派甜虾跑GetNote同步，成功，产出2篇笔记'
    """
    task_name: str = Field(description="任务名称")
    assigned_to: Optional[str] = Field(default=None, description="派给哪个虾或系统")
    result: str = Field(description="结果：success / failed / partial")
    output_summary: Optional[str] = Field(default=None, description="产出摘要")
    lesson: Optional[str] = Field(
        default=None,
        description="这次任务的经验教训，可固化为规则"
    )
    when_to_save: Optional[str] = Field(
        default=None,
        description="触发保存的场景，如：任务完成或失败后，lesson 不为空时优先保存"
    )
    how_to_apply: Optional[str] = Field(
        default=None,
        description="召回后如何应用，如：下次派同类任务前先查历史 lesson，避免重复踩坑"
    )


class Incident(BaseModel):
    """
    故障、踩坑、错误模式记录
    来源：feedback_*.md / 飞虾队历史事件
    示例：'launchd代理配置：HTTP代理对WebSocket不友好，需要NO_PROXY'
    """
    title: str = Field(description="事件标题，一句话")
    root_cause: str = Field(description="根本原因")
    fix: str = Field(description="解决方法")
    prevention: Optional[str] = Field(
        default=None,
        description="预防措施，可固化为检查项"
    )
    affected_systems: Optional[str] = Field(
        default=None,
        description="影响的系统或虾"
    )
    when_to_save: Optional[str] = Field(
        default=None,
        description="触发保存的场景，如：遇到报错、踩坑、F哥反馈'又出这个问题'时"
    )
    how_to_apply: Optional[str] = Field(
        default=None,
        description="召回后如何应用，如：遇到同类系统操作时主动检查 prevention 清单"
    )


class ExternalResource(BaseModel):
    """
    外部资源和服务端口记录
    来源：reference_external.md
    示例：'Neo4j Bolt port=7687 / Clash proxy port=7890'
    """
    resource_name: str = Field(description="资源名称，如 Neo4j / Clash / WPS API")
    resource_type: str = Field(
        description="类型：service(本地服务) / api(外部API) / file(文件路径)"
    )
    location: str = Field(description="端口/路径/URL")
    notes: Optional[str] = Field(default=None, description="注意事项")
    when_to_save: Optional[str] = Field(
        default=None,
        description="触发保存的场景，如：新服务上线、端口变更、API key 轮换时"
    )
    how_to_apply: Optional[str] = Field(
        default=None,
        description="召回后如何应用，如：写代码前先查此资源的 location 和 notes，避免硬编码错误"
    )


# ─── 实体类型注册表（供 Graphiti add_episode 使用）─────────────────────────────

AOXIA_ENTITY_TYPES: dict = {
    "FrankPreference": FrankPreference,
    "ProjectStatus": ProjectStatus,
    "Judgment": Judgment,
    "AgentConfig": AgentConfig,
    "TaskRecord": TaskRecord,
    "Incident": Incident,
    "ExternalResource": ExternalResource,
}


# ─── Agent 档案（供 core 模块使用，避免硬编码）────────────────────────────────

AOXIA_PROFILE: dict = {
    # 对话角色名称（用于 transcript 解析和 prompt 构建）
    "user_display_name": "F哥",
    "assistant_display_name": "鳌虾",

    # 提炼指令：告诉 LLM 从对话里提取哪些内容
    "extraction_instructions": """
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
""",

    # 会话开始时的全量召回查询（覆盖用户最关心的三类信息）
    "session_start_queries": [
        "F哥最近的项目和决策",
        "F哥的偏好规则和判断标准",
        "飞虾队当前状态和任务",
    ],
}


__all__ = [
    "FrankPreference",
    "ProjectStatus",
    "Judgment",
    "AgentConfig",
    "TaskRecord",
    "Incident",
    "ExternalResource",
    "AOXIA_ENTITY_TYPES",
    "AOXIA_PROFILE",
]
