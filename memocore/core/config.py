"""
Memocore 中央配置模块

配置加载优先级（高 → 低）：
  1. 环境变量
  2. 项目目录 .env
  3. 全局配置文件 ~/.memocore/config.env  ← 新增，用于持久化 API key 等

支持的配置项：
  MEMOCORE_AGENT_ID          — Agent 命名空间（默认 "aoxia"）
  MEMOCORE_LLM_PROVIDER      — LLM provider: "anthropic" | "openai"（默认自动检测）
  MEMOCORE_ANTHROPIC_MODEL   — Anthropic 模型（默认 claude-haiku-4-5-20251001）
  MEMOCORE_OPENAI_MODEL      — OpenAI 模型（默认 gpt-4o-mini）
  MEMOCORE_EMBED_PROVIDER    — Embedding provider: "openai" | "local" | "auto"
  MEMOCORE_EMBED_MODEL       — OpenAI embedding 模型
  MEMOCORE_LOCAL_EMBED_MODEL — 本地 embedding 模型名称
  MEMOCORE_DREAM_INTERVAL    — Dream 触发间隔（会话次数，默认 5）
  MEMOCORE_DREAM_TTL_DAYS    — 记忆 TTL（超过此天数且未被引用则过期，默认 90）
  MEMOCORE_PRIVACY_ENABLED   — 是否启用隐私过滤（默认 true）
  MEMOCORE_STATE_DIR         — 状态文件目录（默认 ~/.memocore）
  NEO4J_URI / NEO4J_USER / NEO4J_PASSWORD
  ANTHROPIC_API_KEY / OPENAI_API_KEY
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def _get_raw_state_dir() -> Path:
    """不创建目录，仅返回路径（用于 config 初始化时）"""
    return Path(os.getenv("MEMOCORE_STATE_DIR", Path.home() / ".memocore"))


def _load_global_config():
    """
    加载全局配置文件 ~/.memocore/config.env
    使用 dotenv override=False，确保环境变量优先级更高
    """
    config_file = _get_raw_state_dir() / "config.env"
    if config_file.exists():
        load_dotenv(config_file, override=False)


# 模块加载时自动加载全局配置
_load_global_config()


# ── 基础路径 ────────────────────────────────────────────────────────────────────

def get_state_dir() -> Path:
    """返回 memocore 状态目录，自动创建"""
    base = _get_raw_state_dir()
    base.mkdir(parents=True, exist_ok=True)
    return base


def get_sessions_dir() -> Path:
    """Session flag 文件目录（替代 /tmp，重启不丢失）"""
    d = get_state_dir() / "sessions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_logs_dir() -> Path:
    """日志目录"""
    d = get_state_dir() / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def get_global_config_path() -> Path:
    """全局配置文件路径"""
    return get_state_dir() / "config.env"


# ── Agent 配置 ──────────────────────────────────────────────────────────────────

def get_agent_id() -> str:
    """从环境变量/配置文件读取 agent_id，默认 'aoxia'"""
    return os.getenv("MEMOCORE_AGENT_ID", "aoxia")


# ── 通用 getter ─────────────────────────────────────────────────────────────────

def get(key: str, default: str = "") -> str:
    """读取任意配置项（环境变量或全局 config.env）"""
    return os.getenv(key, default)


def get_neo4j_config() -> dict:
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "user": os.getenv("NEO4J_USER", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", ""),
    }


# ── Dream 配置 ──────────────────────────────────────────────────────────────────

def get_dream_interval() -> int:
    """Dream 触发间隔（会话次数）"""
    try:
        return int(os.getenv("MEMOCORE_DREAM_INTERVAL", "5"))
    except ValueError:
        return 5


def get_dream_ttl_days() -> int:
    """记忆 TTL：超过此天数且未被引用则过期"""
    try:
        return int(os.getenv("MEMOCORE_DREAM_TTL_DAYS", "90"))
    except ValueError:
        return 90


def _get_counter_path(agent_id: str) -> Path:
    return get_state_dir() / f"dream_counter_{agent_id}.txt"


def increment_session_counter(agent_id: str) -> int:
    path = _get_counter_path(agent_id)
    try:
        current = int(path.read_text().strip()) if path.exists() else 0
    except (ValueError, OSError):
        current = 0
    current += 1
    path.write_text(str(current))
    return current


def reset_session_counter(agent_id: str) -> None:
    _get_counter_path(agent_id).write_text("0")


def should_run_dream(agent_id: str) -> bool:
    interval = get_dream_interval()
    count = increment_session_counter(agent_id)
    if count >= interval:
        reset_session_counter(agent_id)
        return True
    return False


# ── 隐私过滤配置 ────────────────────────────────────────────────────────────────

def is_privacy_enabled() -> bool:
    return os.getenv("MEMOCORE_PRIVACY_ENABLED", "true").lower() != "false"


def get_privacy_blacklist() -> list[str]:
    """从配置文件读取自定义黑名单词（逗号分隔）"""
    raw = os.getenv("MEMOCORE_PRIVACY_BLACKLIST", "")
    return [w.strip() for w in raw.split(",") if w.strip()]


# ── 配置写入（供 memocore init 使用）────────────────────────────────────────────

def write_global_config(values: dict) -> Path:
    """
    将 key=value 写入全局配置文件 ~/.memocore/config.env
    不覆盖已有配置行，只追加/更新
    """
    config_path = get_global_config_path()
    existing: dict[str, str] = {}

    if config_path.exists():
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                existing[k.strip()] = v.strip()

    existing.update(values)

    lines = [
        "# Memocore 全局配置",
        "# 由 memocore init 生成，可手动编辑",
        "# 优先级低于环境变量，高于默认值",
        "",
    ]
    for k, v in existing.items():
        lines.append(f"{k}={v}")

    config_path.write_text("\n".join(lines) + "\n")
    return config_path
