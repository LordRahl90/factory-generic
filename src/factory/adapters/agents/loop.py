import logging
from dataclasses import dataclass
from typing import Any

from factory.adapters.agents.tools import SCHEMAS, TOOL_NAMES
from factory.adapters.agents.workspace import Workspace
from factory.adapters.llm.client import complete

log = logging.getLogger(__name__)

SYSTEM = """
You are a careful senior experienced software engineer working in a git repository.
Use the tools to look around, make the smallest change that completes the task,
and run the tests to check.
When the tests pass, reply with a one-paragraph summary and no tool calls."""


class AgentError(Exception): ...


class TurnLimitReachedError(Exception): ...


class BudgetExceededError(Exception): ...


@dataclass(frozen=True, slots=True)
class AgentResult:
    summary: str
    turns: int
    cost_usd: float


def run_agent(
    model: str,
    task: str,
    ws: Workspace,
    *,
    system: str = SYSTEM,
    tools: frozenset[str] = TOOL_NAMES,
    max_turns: int = 20,
    budget_usd: float,
    api_base: str | None = None,
) -> AgentResult:
    schemas = [s for s in SCHEMAS if s["function"]["name"] in tools]
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{task}\n\nFiles in the repository:\n{ws.list_files()}"},
    ]
    spent = 0.0
    for turn_no in range(1, max_turns + 1):
        log.info(f"model is {model}")
        turn = complete(model, messages, tools=schemas, api_base=api_base)
        spent += turn.cost_usd
        messages.append(turn.raw)
        if not turn.tool_calls:
            return AgentResult(summary=turn.text, turns=turn_no, cost_usd=spent)
        for call in turn.tool_calls:
            output = ws.call(call.name, call.args, allowed=tools)
            log.info("turn %d %s(%s) -> %s", turn_no, call.name, call.args, output[:1000])
            messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        if spent > budget_usd:
            raise BudgetExceededError(f"spent ${spent:.2f} of ${budget_usd:.2f}")
    raise TurnLimitReachedError(f"no answer after {max_turns} turns")
