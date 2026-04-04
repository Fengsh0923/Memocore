"""
LLM Adapter — 统一的 LLM 调用层
支持 Claude (Anthropic) 和 OpenAI，自动检测可用 provider

Provider 选择逻辑：
  1. 读取环境变量 MEMOCORE_LLM_PROVIDER = "anthropic" | "openai"
  2. 若未设置，优先用 Anthropic（如果 ANTHROPIC_API_KEY 存在），否则用 OpenAI
  3. 若目标 provider 的 API key 缺失，自动 fallback 到另一个

主要功能：
  - chat_complete()   : 通用对话补全（返回字符串）
  - rerank()          : 记忆重排序（返回精筛后的候选列表）
"""

import json
import logging
import os
import re
from typing import Any, Optional

logger = logging.getLogger("memocore.llm_adapter")


def _detect_provider() -> str:
    """自动检测可用的 LLM provider"""
    explicit = os.getenv("MEMOCORE_LLM_PROVIDER", "").lower()
    if explicit in ("anthropic", "openai"):
        return explicit

    # 自动检测：优先 Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    # 兜底：返回 anthropic（调用时再报错）
    return "anthropic"


async def chat_complete(
    prompt: str,
    system: str = "",
    max_tokens: int = 300,
    temperature: float = 0.0,
    json_mode: bool = False,
    provider: Optional[str] = None,
) -> str:
    """
    通用 LLM 对话补全

    Args:
        prompt: 用户消息
        system: 系统提示（可选）
        max_tokens: 最大输出 token 数
        temperature: 温度
        json_mode: True 时强制输出 JSON（仅 OpenAI 支持 json_object 模式，
                   Anthropic 通过 prompt 引导）
        provider: 强制指定 provider，None 则自动检测

    Returns:
        LLM 返回的文本内容
    """
    p = provider or _detect_provider()

    try:
        if p == "anthropic":
            return await _anthropic_complete(prompt, system, max_tokens, temperature, json_mode)
        else:
            return await _openai_complete(prompt, system, max_tokens, temperature, json_mode)
    except Exception as e:
        # provider 失败时尝试 fallback
        fallback = "openai" if p == "anthropic" else "anthropic"
        logger.warning(f"[llm_adapter] {p} 调用失败: {e}，尝试 fallback 到 {fallback}")
        try:
            if fallback == "anthropic":
                return await _anthropic_complete(prompt, system, max_tokens, temperature, json_mode)
            else:
                return await _openai_complete(prompt, system, max_tokens, temperature, json_mode)
        except Exception as e2:
            raise RuntimeError(f"LLM 调用失败（{p} 和 {fallback} 均不可用）: {e} | {e2}")


async def rerank(
    query: str,
    candidates: list[Any],
    top_k: int,
    fact_extractor=None,
    provider: Optional[str] = None,
) -> list[Any]:
    """
    用 LLM 对候选记忆精筛排序

    Args:
        query: 当前查询上下文
        candidates: 候选结果列表
        top_k: 精筛后保留数量
        fact_extractor: 从候选对象提取文本的函数，默认用 getattr(r, 'fact', str(r))
        provider: 强制指定 provider

    Returns:
        精筛后的候选列表（最多 top_k 条）
    """
    if len(candidates) <= top_k:
        return candidates

    if fact_extractor is None:
        def fact_extractor(r):
            return getattr(r, 'fact', None) or str(r)[:150]

    items = []
    for i, r in enumerate(candidates):
        items.append(f"{i}: {fact_extractor(r)}")
    candidates_text = "\n".join(items)

    prompt = (
        f"你是记忆召回助手。用户当前的问题/上下文是：\n\n"
        f"\"{query}\"\n\n"
        f"以下是从知识图谱召回的候选记忆（共{len(candidates)}条）：\n\n"
        f"{candidates_text}\n\n"
        f"请从中选出最相关的 {top_k} 条，按相关性从高到低排序。\n"
        f"仅返回 JSON 数组，包含选中条目的索引号，例如：[3, 0, 7, 1, 5]"
    )

    try:
        content = await chat_complete(
            prompt=prompt,
            max_tokens=150,
            temperature=0.0,
            json_mode=False,
            provider=provider,
        )
        match = re.search(r'\[[\d,\s]+\]', content)
        if match:
            indices = json.loads(match.group())
            selected = []
            for idx in indices[:top_k]:
                if 0 <= idx < len(candidates):
                    selected.append(candidates[idx])
            if selected:
                return selected
    except Exception as e:
        logger.warning(f"[llm_adapter] rerank 失败，fallback 到截断: {e}")

    return candidates[:top_k]


# ── 内部实现 ────────────────────────────────────────────────────────────────────

async def _anthropic_complete(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> str:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY 未设置")

    client = anthropic.AsyncAnthropic(api_key=api_key)

    # Anthropic 没有原生 json_mode，通过 prompt 引导
    effective_prompt = prompt
    if json_mode and "JSON" not in prompt.upper():
        effective_prompt = prompt + "\n\n请仅返回合法的 JSON，不要包含其他文字。"

    kwargs: dict = dict(
        model=os.getenv("MEMOCORE_ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"),
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": effective_prompt}],
    )
    if system:
        kwargs["system"] = system

    response = await client.messages.create(**kwargs)
    return response.content[0].text


async def _openai_complete(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> str:
    import openai

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY 未设置")

    client = openai.AsyncOpenAI(api_key=api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict = dict(
        model=os.getenv("MEMOCORE_OPENAI_MODEL", "gpt-4o-mini"),
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await client.chat.completions.create(**kwargs)
    return response.choices[0].message.content
