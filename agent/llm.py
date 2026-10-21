"""LLM client for the research agent.

Talks to any OpenAI-compatible chat endpoint (the organizers' Qwen, or a
local llama.cpp/vLLM server while developing). Configuration comes from the
same env variables the organizers' baseline.py uses: OPENAI_API_KEY,
OPENAI_MODEL, OPENAI_BASE_URL. The model name is never hardcoded.

Every answer is parsed as JSON and validated against a pydantic schema; on a
parse/validation error the model gets one short repair round with the error
text. Calls and tokens are counted against a per-task budget (the team's
LLM limit is shared and daily), and once that budget is spent — or if the
endpoint is unreachable — the client raises LLMUnavailable, which the agent
treats as "continue without the LLM", never as a crash. The proxy's
"Budget has been exceeded" error (the team's daily limit is spent) disables
the LLM at once, without retries.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

# transport(messages[, think=False]) -> (text, prompt_tokens, completion_tokens[, cost])
# think=False asks for an answer without reasoning; cost is what the proxy reports
# in usage.cost (0.0 when it reports nothing)
Transport = Callable[[list[dict]], tuple]


class LLMUnavailable(Exception):
    """The LLM cannot be used any more for this task (no config, network, budget)."""


class LLMBadResponse(Exception):
    """The LLM answered, but not with valid JSON for the requested schema."""


@dataclass
class LLMUsage:
    calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    seconds: float = 0.0
    log: list[dict] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


_BUDGET_RE = re.compile(r"budget\s+has\s+been\s+exceeded|budget_exceeded", re.IGNORECASE)


def is_budget_exceeded(exc: BaseException) -> bool:
    """The team's daily LLM limit is spent: retrying is pointless until the reset."""
    texts = [str(exc), str(getattr(exc, "body", "") or "")]
    return any(_BUDGET_RE.search(t) for t in texts)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


class LLMClient:
    def __init__(
        self,
        transport: Transport,
        *,
        model: str = "unknown",
        max_calls: int = 40,
        max_total_tokens: int = 400_000,
        max_consecutive_errors: int = 3,
    ) -> None:
        self._transport = transport
        self.model = model
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self.max_consecutive_errors = max_consecutive_errors
        self.usage = LLMUsage()
        self._consecutive_errors = 0
        self.disabled_reason: str | None = None

    @classmethod
    def from_env(cls) -> "LLMClient | None":
        """None when the LLM is not configured (or AGENT_DISABLE_LLM=1) — and also, defensively,
        on any unexpected construction error. This runs inside run_research()'s outer try, before
        researcher.prepare() ever gets to write the fallback submission.csv — a malformed env var
        (e.g. AGENT_LLM_TEMPERATURE set to something float() can't parse) must degrade to "no LLM"
        rather than take the whole run down before any file exists. Same lesson as the real
        child_main()/os.sched_setaffinity incident (see docs/INFRASTRUCTURE.md), applied here
        pre-emptively rather than after a second one.
        """
        if os.environ.get("AGENT_DISABLE_LLM", "").strip() in ("1", "true", "yes"):
            return None
        model = os.environ.get("OPENAI_MODEL")
        if not model or not os.environ.get("OPENAI_API_KEY"):
            return None
        try:
            return cls(
                openai_transport(
                    model,
                    temperature=_env_float("AGENT_LLM_TEMPERATURE", 0.4),
                    max_tokens=_env_int("AGENT_LLM_MAX_OUTPUT_TOKENS", 6144),
                    timeout_s=_env_float("AGENT_LLM_TIMEOUT_S", 180.0),
                    reasoning_tokens=_env_int("AGENT_LLM_REASONING_TOKENS", 3000),
                    no_think=os.environ.get("AGENT_LLM_NO_THINK", "").strip() in ("1", "true", "yes"),
                ),
                model=model,
                max_calls=_env_int("AGENT_LLM_MAX_CALLS", 40),
                max_total_tokens=_env_int("AGENT_LLM_MAX_TOKENS", 400_000),
            )
        except Exception:
            return None

    @property
    def available(self) -> bool:
        return self.disabled_reason is None

    def _check_budget(self) -> None:
        if self.disabled_reason:
            raise LLMUnavailable(self.disabled_reason)
        if self.usage.calls >= self.max_calls:
            self.disabled_reason = f"call budget spent ({self.max_calls} calls)"
        elif self.usage.total_tokens >= self.max_total_tokens:
            self.disabled_reason = f"token budget spent ({self.usage.total_tokens} tokens)"
        if self.disabled_reason:
            raise LLMUnavailable(self.disabled_reason)

    def _call(self, messages: list[dict], purpose: str, think: bool = True) -> str:
        self._check_budget()
        start = time.monotonic()
        try:
            answer = self._transport(messages) if think else self._transport(messages, think=False)
            text, prompt_tokens, completion_tokens, *rest = answer
        except Exception as exc:
            self.usage.failed_calls += 1
            if is_budget_exceeded(exc):
                self.disabled_reason = f"LLM budget exceeded: {str(exc)[:300]}"
                raise LLMUnavailable(self.disabled_reason) from exc
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self.disabled_reason = f"{self._consecutive_errors} consecutive LLM errors, last: {type(exc).__name__}: {exc}"
            raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc
        self._consecutive_errors = 0
        elapsed = time.monotonic() - start
        self.usage.calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        cost = float(rest[0] or 0.0) if rest else 0.0
        self.usage.cost += cost
        self.usage.seconds += elapsed
        self.usage.log.append(
            {"purpose": purpose, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
             "cost": round(cost, 6), "seconds": round(elapsed, 2)}
        )
        return text

    def complete_json(self, system: str, user: str, schema: type[T], *, purpose: str = "", repair_rounds: int = 1) -> T:
        """Ask for a JSON object matching `schema`. Raises LLMUnavailable or LLMBadResponse."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = ""
        think = True
        for attempt in range(repair_rounds + 1):
            text = self._call(messages, purpose or schema.__name__, think)
            try:
                return parse_json_response(text, schema)
            except LLMBadResponse as exc:
                last_error = str(exc)
                if not _THINK_RE.sub("", text).strip():
                    # Qwen spent the whole max_tokens on reasoning and wrote no answer: retry without reasoning
                    think = False
                    repair = ("You ran out of output tokens while thinking and gave no answer. Answer right away "
                              "with ONE compact JSON object only, matching the requested format.")
                else:
                    repair = ("Your answer could not be used: " + last_error[:1500]
                              + "\nReply again with ONE valid JSON object only, matching the requested format.")
                messages = messages + [
                    {"role": "assistant", "content": text[-4000:]},
                    {"role": "user", "content": repair},
                ]
        raise LLMBadResponse(last_error)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of a chat answer (handles <think>, code fences, prose)."""
    cleaned = _THINK_RE.sub("", text or "")
    if "<think>" in cleaned:  # unterminated reasoning block
        cleaned = cleaned.split("</think>")[-1]
    candidates = [m.strip() for m in _FENCE_RE.findall(cleaned)] + [cleaned.strip()]
    decoder = json.JSONDecoder()
    for candidate in candidates:
        for start in [i for i, ch in enumerate(candidate) if ch == "{"][:20]:
            try:
                obj, _ = decoder.raw_decode(candidate[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                return obj
    raise LLMBadResponse("no JSON object found in the answer")


def parse_json_response(text: str, schema: type[T]) -> T:
    obj = extract_json_object(text)
    try:
        return schema.model_validate(obj)
    except ValidationError as exc:
        raise LLMBadResponse(f"JSON does not match the expected format: {exc}") from exc


def openai_transport(
    model: str,
    *,
    temperature: float = 0.4,
    max_tokens: int = 6144,
    timeout_s: float = 180.0,
    reasoning_tokens: int = 3000,
    no_think: bool = False,
    retries: int = 1,
    retry_wait_s: float = 2.0,
) -> Transport:
    """Transport over the `openai` SDK (reads OPENAI_API_KEY / OPENAI_BASE_URL itself).

    Answers are streamed: the organizers' proxy cuts a non-streamed request at
    ~30 s ("upstream request failed"), and Qwen's reasoning alone can take that long.
    The SDK's own retries are off: it would also retry a 429 "Budget has been
    exceeded". Network errors, timeouts, 5xx and other 429s get `retries` more tries
    (few: a failed attempt is still billed for what the model generated).

    Reasoning is capped at `reasoning_tokens` (0 = no cap) so that max_tokens always
    leaves room for the answer; think=False / no_think turn it off. Both use the
    proxy's OpenRouter-style `reasoning` field (Qwen ignores "/no_think" too often).
    """
    import httpx
    from openai import APIConnectionError, InternalServerError, OpenAI, RateLimitError

    client = OpenAI(timeout=timeout_s, max_retries=0)

    def stream_once(messages: list[dict], extra_body: dict | None) -> tuple[str, object]:
        stream = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
            stream=True, stream_options={"include_usage": True}, extra_body=extra_body,
        )
        parts, usage = [], None
        for chunk in stream:
            if chunk.usage is not None:
                usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                parts.append(chunk.choices[0].delta.content)
        return "".join(parts), usage

    def create(messages: list[dict], extra_body: dict | None) -> tuple[str, object]:
        for attempt in range(retries + 1):
            try:
                return stream_once(messages, extra_body)
            # httpx.TransportError: the stream broke mid-answer (e.g. RemoteProtocolError, seen on the proxy)
            except (APIConnectionError, InternalServerError, RateLimitError, httpx.TransportError) as exc:
                if attempt >= retries or is_budget_exceeded(exc):
                    raise
                time.sleep(retry_wait_s * 2**attempt)
        raise AssertionError("unreachable")

    def transport(messages: list[dict], think: bool = True) -> tuple[str, int, int, float]:
        if no_think or not think:
            extra_body = {"reasoning": {"effort": "none"}}
        else:
            extra_body = {"reasoning": {"max_tokens": reasoning_tokens}} if reasoning_tokens > 0 else None
        text, usage = create(messages, extra_body)
        if usage is None:  # proxy sent no usage chunk: rough estimate, no cost
            return text, sum(len(m["content"]) for m in messages) // 4, len(text) // 4, 0.0
        return (
            text,
            int(getattr(usage, "prompt_tokens", 0) or 0),
            int(getattr(usage, "completion_tokens", 0) or 0),
            float(getattr(usage, "cost", 0.0) or 0.0),
        )

    return transport


def scripted_transport(answers: list[str]) -> Transport:
    """Deterministic transport for tests: returns the queued answers in order."""
    queue = list(answers)

    def transport(messages: list[dict]) -> tuple[str, int, int]:
        if not queue:
            raise RuntimeError("scripted transport has no answers left")
        prompt = sum(len(m["content"]) for m in messages) // 4
        answer = queue.pop(0)
        return answer, prompt, len(answer) // 4

    return transport
