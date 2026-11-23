from types import SimpleNamespace

import litellm
import pytest

from factory.adapters.llm import client
from factory.adapters.llm.client import _calls_from_text


def fake_completion(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )


def test_ask_maps_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.002)
    reply = client.ask("any/model", [{"role": "user", "content": "ping"}])
    assert reply == client.Reply(
        text="pong", model="any/model", tokens_in=3, tokens_out=1, cost_usd=0.002
    )


def test_un_priced_model_comes_with_no_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_price(**_: object) -> float:
        raise ValueError("model not in price map")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", no_price)
    assert client.ask("ollama_chat/x", []).cost_usd == 0.0


def test_tool_calls_written_as_text_are_recovered() -> None:
    calls = _calls_from_text('{"name": "run", "arguments": {"command": "pytest -q"}}')
    assert [(c.name, c.args) for c in calls] == [("run", {"command": "pytest -q"})]


def test_calls_wrapped_in_prose_and_tags_are_found() -> None:
    text = (
        "The seconds factor is wrong {see below}."
        + '\n<tool_request>\n{"name": "read_file", "arguments": {"path": "durations.py"}}'
        + "\n</tool_request>"
    )
    assert [c.name for c in _calls_from_text(text)] == ["read_file"]


def test_ordinary_answers_are_left_alone() -> None:
    assert _calls_from_text("All tests pass now. I changed the seconds multiplier.") == ()
    assert _calls_from_text('{"summary": "done"}') == ()
