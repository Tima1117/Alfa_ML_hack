import pytest

from agent.llm import LLMBadResponse, LLMClient, LLMUnavailable, extract_json_object, scripted_transport
from agent.schemas import HypothesisBatch, TaskUnderstanding


def test_extracts_json_from_reasoning_fences_and_prose():
    text = '<think>maybe {"not": "this"}</think>Sure! ```json\n{"metric_name": "rmse", "summary": "x"}\n``` done'
    assert extract_json_object(text) == {"metric_name": "rmse", "summary": "x"}
    assert extract_json_object('Answer: {"a": {"b": 1}} trailing') == {"a": {"b": 1}}
    with pytest.raises(LLMBadResponse):
        extract_json_object("no json here")


def test_lenient_schema_tolerates_extra_keys_and_nulls():
    batch = HypothesisBatch.model_validate(
        {"analysis": "a", "hypotheses": [{"title": "t", "kind": "weird", "hyperparameters": None, "extra": 1}], "foo": 2}
    )
    assert batch.hypotheses[0].kind == "other"
    assert batch.hypotheses[0].hyperparameters == {}


def test_repair_round_fixes_a_bad_answer_and_counts_usage():
    client = LLMClient(scripted_transport(["I think rmse.", '{"metric_name": "rmse"}']), max_calls=5)
    result = client.complete_json("sys", "user", TaskUnderstanding)
    assert result.metric_name == "rmse"
    assert client.usage.calls == 2 and client.usage.total_tokens > 0


def test_budget_and_errors_make_the_client_unavailable():
    client = LLMClient(scripted_transport(['{"a": 1}'] * 3), max_calls=1)
    client.complete_json("s", "u", TaskUnderstanding)
    with pytest.raises(LLMUnavailable):
        client.complete_json("s", "u", TaskUnderstanding)
    assert not client.available

    def broken(messages):
        raise ConnectionError("down")

    client = LLMClient(broken, max_consecutive_errors=2)
    for _ in range(2):
        with pytest.raises(LLMUnavailable):
            client.complete_json("s", "u", TaskUnderstanding)
    assert not client.available and "consecutive" in client.disabled_reason


def test_from_env_returns_none_without_configuration(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert LLMClient.from_env() is None
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENAI_MODEL", "m")
    monkeypatch.setenv("AGENT_DISABLE_LLM", "1")
    assert LLMClient.from_env() is None


def test_budget_exceeded_disables_the_llm_after_one_call():
    calls = []

    def spent(messages):
        calls.append(1)
        raise RuntimeError("Error code: 400 - {'error': {'message': 'Budget has been exceeded! Current cost: 600'}}")

    client = LLMClient(spent, max_consecutive_errors=3)
    with pytest.raises(LLMUnavailable):
        client.complete_json("s", "u", TaskUnderstanding)
    assert not client.available and "budget exceeded" in client.disabled_reason
    with pytest.raises(LLMUnavailable):
        client.complete_json("s", "u", TaskUnderstanding)
    assert len(calls) == 1


class _FakeOpenAI:
    """Stands in for openai.OpenAI: create() pops the next item of `script` (an exception or a chunk list)."""

    script: list = []
    calls: list = []

    def __init__(self, **kwargs):
        assert kwargs["max_retries"] == 0
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        assert kwargs["stream"] is True
        type(self).calls.append(kwargs)
        item = type(self).script.pop(0)
        if isinstance(item, Exception):
            raise item
        return iter(item)


def _chunk(content=None, usage=None):
    from types import SimpleNamespace

    choices = [SimpleNamespace(delta=SimpleNamespace(content=content))] if usage is None else []
    return SimpleNamespace(choices=choices, usage=usage)


def test_openai_transport_streams_and_does_not_retry_a_spent_budget(monkeypatch):
    from types import SimpleNamespace

    import httpx
    import openai

    from agent.llm import openai_transport

    monkeypatch.setattr(openai, "OpenAI", _FakeOpenAI)
    response = httpx.Response(429, request=httpx.Request("POST", "http://proxy/v1/chat/completions"))
    messages = [{"role": "user", "content": "hi"}]
    transport = openai_transport("m", retries=2, retry_wait_s=0)

    usage = SimpleNamespace(prompt_tokens=10, completion_tokens=3, cost=0.5)
    _FakeOpenAI.calls, _FakeOpenAI.script = [], [[_chunk('{"a"'), _chunk(None), _chunk(": 1}"), _chunk(usage=usage)]]
    assert transport(messages) == ('{"a": 1}', 10, 3, 0.5)
    assert _FakeOpenAI.calls[-1]["extra_body"] == {"reasoning": {"max_tokens": 3000}}
    _FakeOpenAI.script = [[_chunk("{}")]]
    transport(messages, think=False)
    assert _FakeOpenAI.calls[-1]["extra_body"] == {"reasoning": {"effort": "none"}}

    _FakeOpenAI.calls, _FakeOpenAI.script = [], [openai.RateLimitError("Budget has been exceeded", response=response, body=None)]
    with pytest.raises(openai.RateLimitError):
        transport(messages)
    assert len(_FakeOpenAI.calls) == 1

    _FakeOpenAI.calls, _FakeOpenAI.script = [], [httpx.RemoteProtocolError("peer closed connection"), [_chunk("{}")]]
    assert transport(messages)[0] == "{}"  # a stream broken mid-answer is retried
    assert len(_FakeOpenAI.calls) == 2

    busy = openai.RateLimitError("Too many requests", response=response, body=None)
    _FakeOpenAI.calls, _FakeOpenAI.script = [], [busy, busy, [_chunk("{}")]]
    assert transport(messages)[0] == "{}"  # an ordinary 429 is retried; no usage chunk -> estimate
    assert len(_FakeOpenAI.calls) == 3


def test_usage_counts_the_cost_reported_by_the_proxy():
    client = LLMClient(lambda messages: ('{"metric_name": "rmse"}', 100, 20, 0.25))
    client.complete_json("s", "u", TaskUnderstanding)
    assert client.usage.cost == 0.25 and client.usage.log[0]["cost"] == 0.25


def test_an_answer_cut_off_while_reasoning_is_retried_without_thinking():
    seen = []

    def transport(messages, think=True):
        seen.append((messages, think))
        return ("" if len(seen) == 1 else '{"metric_name": "rmse"}'), 100, 6144

    client = LLMClient(transport)
    assert client.complete_json("s", "u", TaskUnderstanding).metric_name == "rmse"
    assert [think for _, think in seen] == [True, False]
    assert "ran out of output tokens" in seen[1][0][-1]["content"]
