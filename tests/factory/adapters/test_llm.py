from types import SimpleNamespace

import litellm
import pytest

from factory.adapters import llm


def fake_completion(**kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content="pong"))],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1),
    )


def test_ask_maps_the_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", lambda **_: 0.002)
    reply = llm.ask("any/model", [{"role": "user", "content": "ping"}])
    assert reply == llm.Reply(
        text="pong", model="any/model", tokens_in=3, tokens_out=1, cost_usd=0.002
    )


def test_un_priced_model_comes_with_no_cost(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_price(**_: object) -> float:
        raise ValueError("model not in price map")

    monkeypatch.setattr(litellm, "completion", fake_completion)
    monkeypatch.setattr(litellm, "completion_cost", no_price)
    assert llm.ask("ollama_chat/x", []).cost_usd == 0.0
