from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from factory.adapters.agents import loop as agent
from factory.adapters.agents.workspace import Workspace
from factory.adapters.llm.client import ToolCall, Turn


def scripted(turns: list[Turn]) -> Callable[..., Turn]:
    remaining = iter(turns)
    return lambda *args, **kwargs: next(remaining)


def say(text: str, *calls: ToolCall) -> Turn:
    raw: dict[str, Any] = {"role": "assistant", "content": "text"}
    return Turn(text=text, tool_calls=calls, cost_usd=0.01, raw=raw)


def test_runs_tools_then_stops(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        agent,
        "complete",
        scripted(
            [
                say("", ToolCall("c1", "write_file", {"path": "a.txt", "content": "hi"})),
                say("wrote the file"),
            ]
        ),
    )

    result = agent.run_agent(
        "fake/model", "write a file", Workspace(tmp_path), api_base="", budget_usd=10.0
    )
    assert (tmp_path / "a.txt").read_text() == "hi"
    assert result.turns == 2
    assert result.cost_usd == pytest.approx(0.02)


def test_stops_at_the_turn_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    forever = ToolCall("c", "list_files", {})
    monkeypatch.setattr(agent, "complete", lambda *a, **k: say("", forever))

    with pytest.raises(agent.TurnLimitReachedError):
        agent.run_agent(
            "fake/model", "loop", Workspace(tmp_path), max_turns=3, budget_usd=10, api_base=""
        )


def test_every_advertised_tool_is_dispatched(tmp_path: Path) -> None:
    from factory.adapters.agents.tools import TOOL_NAMES

    ws = Workspace(tmp_path)
    for name in TOOL_NAMES:
        assert "not available" not in ws.call(name, {}), name


def test_path_outside_the_workspace_are_refused(tmp_path: Path) -> None:
    print(f"\n\nPath: {tmp_path}\n\n")
    ws = Workspace(tmp_path)
    assert ws.call("read_file", {"path": "../secrets.txt"}).startswith("error: PermissionError")
