"""
隐私过滤模块

在记忆写入前自动扫描对话内容，识别并 redact 敏感信息：

内置规则（正则）：
  - API Keys: OpenAI (sk-...), Anthropic (sk-ant-...), GitHub (ghp_...) 等主流格式
  - 私钥 / 证书: -----BEGIN ... KEY-----
  - 密码参数: --password xxx, -p xxx, password=xxx
  - 数据库连接串: postgresql://user:pass@host
  - Bearer Token: Authorization: Bearer ...
  - 手机号（中国大陆）: 1[3-9]xxxxxxxxx
  - 身份证号（中国）: 18 位

可配置：
  MEMOCORE_PRIVACY_ENABLED=false       — 禁用过滤
  MEMOCORE_PRIVACY_BLACKLIST=word1,w2  — 自定义黑名单短语（命中则整条跳过）
  MEMOCORE_PRIVACY_LLM_CHECK=true      — 启用 LLM 二次检测（默认 false，增加延迟）

使用：
  from memocore.core.privacy import PrivacyFilter
  f = PrivacyFilter()
  clean_text, report = f.process(raw_text)
  if report.should_skip:
      # 整条对话不写入
  else:
      # clean_text 已 redact 敏感字段
"""

import logging
import os
import re
from dataclasses import dataclass, field

from memocore.core.config import get_privacy_blacklist, is_privacy_enabled

logger = logging.getLogger("memocore.privacy")

# ── 内置正则规则 ─────────────────────────────────────────────────────────────────

_REDACT_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # OpenAI API key
    ("openai_key", re.compile(r'\bsk-[A-Za-z0-9\-_]{20,}'), "[OPENAI_KEY]"),
    # Anthropic API key
    ("anthropic_key", re.compile(r'\bsk-ant-[A-Za-z0-9\-_]{20,}'), "[ANTHROPIC_KEY]"),
    # GitHub token
    ("github_token", re.compile(r'\bghp_[A-Za-z0-9]{36,}'), "[GITHUB_TOKEN]"),
    # Generic Bearer token (Authorization header)
    ("bearer_token", re.compile(
        r'(?i)(Authorization:\s*Bearer\s+)[A-Za-z0-9\-._~+/]+=*'
    ), r'\1[BEARER_TOKEN]'),
    # 私钥 / PEM
    ("pem_key", re.compile(
        r'-----BEGIN\s+(?:RSA\s+)?(?:PRIVATE|PUBLIC)\s+KEY-----.*?-----END[^-]*?KEY-----',
        re.DOTALL
    ), "[PEM_KEY]"),
    # 数据库连接串（含密码）
    ("db_url", re.compile(
        r'(?i)(postgresql|mysql|mongodb|redis)://[^:@\s]+:[^@\s]+@'
    ), r'\1://[USER]:[PASSWORD]@'),
    # CLI 密码参数 --password xxx / -p xxx / password=xxx
    ("cli_password", re.compile(
        r'(?i)(?:--password|-p\s+|password[=\s]+)([^\s\'"]{4,})'
    ), "[PASSWORD]"),
    # AWS Access Key / Secret
    ("aws_key", re.compile(r'\b(AKIA|ASIA)[A-Z0-9]{16}\b'), "[AWS_KEY]"),
    ("aws_secret", re.compile(
        r'(?i)aws[_\-]?secret[_\-]?(?:access[_\-]?)?key[=:\s]+[A-Za-z0-9/+=]{20,}'
    ), "[AWS_SECRET]"),
    # 中国大陆手机号（严格模式：1[3-9]开头，11位）
    ("cn_phone", re.compile(r'\b1[3-9]\d{9}\b'), "[PHONE]"),
    # 中国居民身份证（18位）
    ("cn_id", re.compile(r'\b\d{17}[\dXx]\b'), "[ID_NUMBER]"),
]

# 命中任一规则则整条对话跳过（不 redact，直接放弃写入）
_SKIP_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("private_key_block", re.compile(
        r'-----BEGIN\s+(?:ENCRYPTED\s+)?PRIVATE\s+KEY-----'
    )),
]


@dataclass
class PrivacyReport:
    redacted_count: int = 0
    redacted_types: list[str] = field(default_factory=list)
    should_skip: bool = False
    skip_reason: str = ""

    def __str__(self):
        if self.should_skip:
            return f"SKIP: {self.skip_reason}"
        if self.redacted_count > 0:
            return f"REDACTED {self.redacted_count} items: {', '.join(self.redacted_types)}"
        return "CLEAN"


class PrivacyFilter:
    """
    对话内容隐私过滤器。
    process() 返回 (cleaned_text, PrivacyReport)。
    """

    def __init__(self):
        self._enabled = is_privacy_enabled()
        self._blacklist = get_privacy_blacklist()
        self._llm_check = os.getenv("MEMOCORE_PRIVACY_LLM_CHECK", "false").lower() == "true"

    def process(self, text: str) -> tuple[str, PrivacyReport]:
        """
        扫描并处理文本中的隐私信息

        Returns:
            (processed_text, PrivacyReport)
            若 report.should_skip=True，调用方应跳过整条记忆写入
        """
        report = PrivacyReport()

        if not self._enabled:
            return text, report

        # 1. 黑名单短语检测（整条跳过）
        for phrase in self._blacklist:
            if phrase.lower() in text.lower():
                report.should_skip = True
                report.skip_reason = f"blacklist hit: '{phrase}'"
                logger.info(f"[privacy] 跳过: {report.skip_reason}")
                return text, report

        # 2. skip 模式检测（含私钥 block 等高危内容，整条跳过）
        for name, pattern in _SKIP_PATTERNS:
            if pattern.search(text):
                report.should_skip = True
                report.skip_reason = f"high-risk pattern: {name}"
                logger.info(f"[privacy] 跳过: {report.skip_reason}")
                return text, report

        # 3. Redact 模式（替换敏感字段）
        result = text
        for name, pattern, replacement in _REDACT_PATTERNS:
            new_text, count = pattern.subn(replacement, result)
            if count > 0:
                result = new_text
                report.redacted_count += count
                report.redacted_types.append(name)

        if report.redacted_count > 0:
            logger.info(f"[privacy] {report}")

        return result, report

    async def process_async(self, text: str) -> tuple[str, PrivacyReport]:
        """
        异步版本，支持 LLM 二次检测（MEMOCORE_PRIVACY_LLM_CHECK=true 时启用）
        """
        text, report = self.process(text)

        if report.should_skip or not self._llm_check:
            return text, report

        # LLM 二次检测：对正则未能覆盖的隐私信息做模糊判断
        try:
            from memocore.core.llm_adapter import chat_complete
            prompt = (
                "你是隐私审查助手。以下是一段待存储的对话摘要。\n"
                "请判断其中是否包含以下类型的敏感信息：\n"
                "  - 密码、密钥、token（未被 [REDACTED] 标记的）\n"
                "  - 个人身份信息（姓名+地址+身份证的组合）\n"
                "  - 信用卡号\n\n"
                f"对话内容：\n{text[:800]}\n\n"
                "请仅回复：SAFE 或 SENSITIVE（不要解释）"
            )
            verdict = await chat_complete(prompt, max_tokens=10, temperature=0.0)
            if "SENSITIVE" in verdict.upper():
                report.should_skip = True
                report.skip_reason = "LLM detected sensitive content"
                logger.info(f"[privacy] LLM 判定为敏感内容，跳过")
        except Exception as e:
            logger.warning(f"[privacy] LLM 检测失败（跳过检测步骤）: {e}")

        return text, report


# ── 便捷函数 ────────────────────────────────────────────────────────────────────

_default_filter: PrivacyFilter | None = None


def get_filter() -> PrivacyFilter:
    global _default_filter
    if _default_filter is None:
        _default_filter = PrivacyFilter()
    return _default_filter
