import json
from dataclasses import dataclass
from typing import Any

import litellm

# litellm.turn_on_debug()


@dataclass(frozen=True, slots=True)
class Reply:
    text: str
    model: str
    tokens_in: int
    tokens_out: int
    cost_usd: float


def ask(model: str, messages: list[dict[str, str]], *, api_base: str | None = None) -> Reply:
    resp: Any = litellm.completion(model=model, messages=messages, api_base=api_base)
    usage = getattr(resp, "usage", None)
    return Reply(
        text=resp.choices[0].message.content or "",
        model=model,
        tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
        tokens_out=getattr(usage, "completion_tokens", 0) or 0,
        cost_usd=cost_of(resp),
    )


def cost_of(resp: Any) -> float:
    try:
        return float(litellm.completion_cost(completion_response=resp))
    except Exception:
        return 0.0


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Turn:
    text: str
    tool_calls: tuple[ToolCall, ...]
    cost_usd: float
    raw: dict[str, Any]


def complete(
    model: str,
    messages: list[dict[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,  # start with no tools
    api_base: str | None = None,
) -> Turn:
    resp: Any = litellm.completion(model=model, messages=messages, tools=tools, api_base=api_base)
    msg = resp.choices[0].message
    text = msg.content or ""

    calls = tuple(
        ToolCall(c.id, c.function.name, _args(c.function.arguments)) for c in (msg.tool_calls or [])
    )
    if not calls and tools and (calls := _calls_from_text(text)):
        text = ""
    raw: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        raw["tool_calls"] = [
            {
                "id": c.id,
                "type": "function",
                "function": {"name": c.name, "arguments": json.dumps(c.args)},
            }
            for c in calls
        ]
    return Turn(text=text, tool_calls=calls, cost_usd=cost_of(resp), raw=raw)


def _args(arguments: Any) -> dict[str, Any]:
    parsed = arguments if isinstance(arguments, dict) else json.loads(arguments or "{}")
    return parsed if isinstance(parsed, dict) else {}


def _calls_from_text(text: str) -> tuple[ToolCall, ...]:
    decoder, calls, i = json.JSONDecoder(), list[ToolCall](), 0
    while (start := text.find("{", i)) != -1:
        try:
            obj, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            i = start + 1
            continue
        if (
            isinstance(obj, dict)
            and isinstance(obj.get("name"), str)
            and isinstance(obj.get("arguments"), dict)
        ):
            calls.append(ToolCall(f"text-call-{len(calls)}", obj["name"], obj["arguments"]))
        i = end
    return tuple(calls)
