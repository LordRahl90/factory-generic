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
