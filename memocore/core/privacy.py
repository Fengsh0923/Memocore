"""
Privacy Filtering Module

Automatically scans conversation content before memory write, identifies and redacts sensitive information:

Built-in rules (regex):
  - API Keys: OpenAI (sk-...), Anthropic (sk-ant-...), GitHub (ghp_...) and other common formats
  - Private keys / certificates: -----BEGIN ... KEY-----
  - Password parameters: --password xxx, -p xxx, password=xxx
  - Database connection strings: postgresql://user:pass@host
  - Bearer Token: Authorization: Bearer ...
  - Phone numbers (China mainland): 1[3-9]xxxxxxxxx
  - National ID numbers (China): 18 digits

Configurable:
  MEMOCORE_PRIVACY_ENABLED=false       — Disable filtering
  MEMOCORE_PRIVACY_BLACKLIST=word1,w2  — Custom blacklist phrases (skip entire entry if matched)
  MEMOCORE_PRIVACY_LLM_CHECK=true      — Enable LLM secondary check (default false, adds latency)

Usage:
  from memocore.core.privacy import PrivacyFilter
  f = PrivacyFilter()
  clean_text, report = f.process(raw_text)
  if report.should_skip:
      # Do not write entire conversation
  else:
      # clean_text has redacted sensitive fields
"""

import logging
import os
import re
from dataclasses import dataclass, field

from memocore.core.config import get_privacy_blacklist, is_privacy_enabled

logger = logging.getLogger("memocore.privacy")

# ── Built-in regex rules ─────────────────────────────────────────────────────────────────

_REDACT_PATTERNS: list[tuple[str, re.Pattern, str]] = [
    # Anthropic API key (must be before openai_key, because sk-ant- also matches the sk- prefix)
    ("anthropic_key", re.compile(r'\bsk-ant-[A-Za-z0-9\-_]{20,}'), "[ANTHROPIC_KEY]"),
    # OpenAI API key
    ("openai_key", re.compile(r'\bsk-[A-Za-z0-9\-_]{20,}'), "[OPENAI_KEY]"),
    # GitHub token
    ("github_token", re.compile(r'\bghp_[A-Za-z0-9]{36,}'), "[GITHUB_TOKEN]"),
    # Generic Bearer token (Authorization header)
    ("bearer_token", re.compile(
        r'(?i)(Authorization:\s*Bearer\s+)[A-Za-z0-9\-._~+/]+=*'
    ), r'\1[BEARER_TOKEN]'),
    # Private key / PEM
    ("pem_key", re.compile(
        r'-----BEGIN\s+(?:RSA\s+)?(?:PRIVATE|PUBLIC)\s+KEY-----.*?-----END[^-]*?KEY-----',
        re.DOTALL
    ), "[PEM_KEY]"),
    # Database connection string (with password)
    ("db_url", re.compile(
        r'(?i)(postgresql|mysql|mongodb|redis)://[^:@\s]+:[^@\s]+@'
    ), r'\1://[USER]:[PASSWORD]@'),
    # CLI password parameters --password xxx / -p xxx / password=xxx
    ("cli_password", re.compile(
        r'(?i)(?:--password|-p\s+|password[=\s]+)([^\s\'"]{4,})'
    ), "[PASSWORD]"),
    # AWS Access Key / Secret
    ("aws_key", re.compile(r'\b(AKIA|ASIA)[A-Z0-9]{16}\b'), "[AWS_KEY]"),
    ("aws_secret", re.compile(
        r'(?i)aws[_\-]?secret[_\-]?(?:access[_\-]?)?key[=:\s]+[A-Za-z0-9/+=]{20,}'
    ), "[AWS_SECRET]"),
    # China mainland phone number (strict mode: starts with 1[3-9], 11 digits)
    ("cn_phone", re.compile(r'\b1[3-9]\d{9}\b'), "[PHONE]"),
    # China national ID (18 digits)
    ("cn_id", re.compile(r'\b\d{17}[\dXx]\b'), "[ID_NUMBER]"),
]

# Skip entire conversation if any rule matches (no redact, discard write directly)
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
    Conversation content privacy filter.
    process() returns (cleaned_text, PrivacyReport).
    """

    def __init__(self):
        self._enabled = is_privacy_enabled()
        self._blacklist = get_privacy_blacklist()
        self._llm_check = os.getenv("MEMOCORE_PRIVACY_LLM_CHECK", "false").lower() == "true"

    def process(self, text: str) -> tuple[str, PrivacyReport]:
        """
        Scan and process privacy information in text

        Returns:
            (processed_text, PrivacyReport)
            If report.should_skip=True, caller should skip entire memory write
        """
        report = PrivacyReport()

        if not self._enabled:
            return text, report

        # 1. Blacklist phrase detection (skip entire entry)
        for phrase in self._blacklist:
            if phrase.lower() in text.lower():
                report.should_skip = True
                report.skip_reason = f"blacklist hit: '{phrase}'"
                logger.info(f"[privacy] skip: {report.skip_reason}")
                return text, report

        # 2. Skip pattern detection (contains high-risk content like private key blocks, skip entire entry)
        for name, pattern in _SKIP_PATTERNS:
            if pattern.search(text):
                report.should_skip = True
                report.skip_reason = f"high-risk pattern: {name}"
                logger.info(f"[privacy] skip: {report.skip_reason}")
                return text, report

        # 3. Redact mode (replace sensitive fields)
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
        Async version, supports LLM secondary check (enabled when MEMOCORE_PRIVACY_LLM_CHECK=true)
        """
        text, report = self.process(text)

        if report.should_skip or not self._llm_check:
            return text, report

        # LLM secondary check: fuzzy detection of privacy info not covered by regex
        try:
            from memocore.core.llm_adapter import chat_complete
            from memocore.core.locale import t
            prompt = t("privacy.llm_check_prompt", text=text[:800])
            verdict = await chat_complete(prompt, max_tokens=10, temperature=0.0)
            if "SENSITIVE" in verdict.upper():
                report.should_skip = True
                report.skip_reason = "LLM detected sensitive content"
                logger.info(f"[privacy] LLM classified as sensitive content, skipping")
        except Exception as e:
            logger.warning(f"[privacy] LLM check failed (skipping check step): {e}")

        return text, report


# ── Convenience functions ────────────────────────────────────────────────────────────────────

_default_filter: PrivacyFilter | None = None


def get_filter() -> PrivacyFilter:
    global _default_filter
    if _default_filter is None:
        _default_filter = PrivacyFilter()
    return _default_filter
