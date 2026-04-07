"""
Memocore LLM Adapter — unified LLM call layer.
Supports Claude (Anthropic) and OpenAI with automatic provider detection.

Provider selection:
  1. MEMOCORE_LLM_PROVIDER env var = "anthropic" | "openai"
  2. If unset, prefer Anthropic (if ANTHROPIC_API_KEY exists), else OpenAI
  3. If target provider's API key is missing, auto-fallback to the other

Main functions:
  - chat_complete()   : general chat completion (returns string)
  - rerank()          : memory reranking (returns refined candidate list)
"""

import json
import logging
import os
import re
import threading
from typing import Any, Optional

logger = logging.getLogger("memocore.llm_adapter")

# ── LLM client singletons (avoid creating new HTTP connection per call) ────────

_anthropic_client = None
_openai_client = None
_client_lock = threading.Lock()


def _get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        with _client_lock:
            if _anthropic_client is None:
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise RuntimeError("ANTHROPIC_API_KEY is not set")
                _anthropic_client = anthropic.AsyncAnthropic(api_key=api_key)
    return _anthropic_client


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        with _client_lock:
            if _openai_client is None:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise RuntimeError("OPENAI_API_KEY is not set")
                base_url = os.getenv("OPENAI_BASE_URL")
                if base_url:
                    _openai_client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)
                else:
                    _openai_client = openai.AsyncOpenAI(api_key=api_key)
    return _openai_client


# ── JSON parsing utility (fault-tolerant LLM output handling) ──────────────────

def parse_llm_json(content: str) -> Any:
    """
    Extract JSON from LLM output, automatically strip markdown fences and prose preamble.
    Raises json.JSONDecodeError if no valid JSON found.
    """
    content = content.strip()
    # try direct parse
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # strip markdown fence
    cleaned = re.sub(r'^```(?:json)?\s*\n?', '', content, flags=re.IGNORECASE)
    cleaned = re.sub(r'\n?\s*```\s*$', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # extract first JSON object or array
    match = re.search(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}|\[[\s\S]*?\])', content)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("No JSON found in LLM output", content, 0)


def _detect_provider() -> str:
    """Auto-detect available LLM provider"""
    explicit = os.getenv("MEMOCORE_LLM_PROVIDER", "").lower()
    if explicit in ("anthropic", "openai"):
        return explicit

    # auto-detect: prefer Anthropic
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"

    # fallback: return anthropic (error reported at call time)
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
    General LLM chat completion

    Args:
        prompt: user message
        system: system prompt (optional)
        max_tokens: max output tokens
        temperature: temperature
        json_mode: when True, force JSON output (only OpenAI supports json_object mode;
                   Anthropic is guided via prompt)
        provider: force a specific provider; None means auto-detect

    Returns:
        Text content returned by the LLM
    """
    p = provider or _detect_provider()

    try:
        if p == "anthropic":
            return await _anthropic_complete(prompt, system, max_tokens, temperature, json_mode)
        else:
            return await _openai_complete(prompt, system, max_tokens, temperature, json_mode)
    except Exception as e:
        # try fallback when provider fails
        fallback = "openai" if p == "anthropic" else "anthropic"
        logger.warning(f"[llm_adapter] {p} call failed: {e}, falling back to {fallback}")
        try:
            if fallback == "anthropic":
                return await _anthropic_complete(prompt, system, max_tokens, temperature, json_mode)
            else:
                return await _openai_complete(prompt, system, max_tokens, temperature, json_mode)
        except Exception as e2:
            raise RuntimeError(f"LLM call failed (both {p} and {fallback} unavailable): {e} | {e2}") from e2


async def rerank(
    query: str,
    candidates: list[Any],
    top_k: int,
    fact_extractor=None,
    provider: Optional[str] = None,
) -> list[Any]:
    """
    Use LLM to rerank candidate memories

    Args:
        query: current query context
        candidates: list of candidate results
        top_k: number of results to keep after reranking
        fact_extractor: function to extract text from a candidate object; defaults to getattr(r, 'fact', str(r))
        provider: force a specific provider

    Returns:
        Reranked candidate list (at most top_k entries)
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

    from memocore.core.locale import t
    prompt = t(
        "retriever.rerank_prompt",
        query=query,
        count=len(candidates),
        candidates_text=candidates_text,
        top_k=top_k,
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
        logger.warning(f"[llm_adapter] rerank failed, falling back to truncation: {e}")

    return candidates[:top_k]


# ── internal implementation ─────────────────────────────────────────────────────

async def _anthropic_complete(
    prompt: str,
    system: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
) -> str:
    client = _get_anthropic_client()

    # Anthropic has no native json_mode; guide via prompt suffix
    effective_prompt = prompt
    if json_mode and "JSON" not in prompt.upper():
        effective_prompt = prompt + "\n\nReturn only valid JSON. Do not include any other text."

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
    client = _get_openai_client()
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
