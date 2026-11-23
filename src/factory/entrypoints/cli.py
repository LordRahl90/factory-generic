import logging
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from factory.adapters.agents.loop import AgentError, run_agent
from factory.adapters.agents.workspace import Workspace
from factory.adapters.llm.client import ask
from factory.bootstrap.settings import Settings

app = typer.Typer(no_args_is_help=True)


@app.callback()
def main() -> None:
    load_dotenv()


def api_base_for(model: str, settings: Settings) -> str | None:
    return settings.ollama_base_url if model.startswith("ollama") else None


@app.command("ask")
def ask_cmd(
    prompt: str,
    model: Annotated[str | None, typer.Option(help="LiteLLM model string")] = None,
) -> None:
    """This is for developer to talk to the model directly"""
    settings = Settings()
    chosen = model or settings.default_model
    reply = ask(
        chosen, [{"role": "user", "content": prompt}], api_base=api_base_for(chosen, settings)
    )
    typer.echo(reply.text)
    typer.echo(
        f"\n{reply.model} . {reply.tokens_in} + {reply.tokens_out} tokens . ${reply.cost_usd:.4f}",
        err=True,
    )


DEFAULT_TASK = (
    "The tests are failing. Find the cause, fix the code, and run `pytest -q` until they pass."
)


@app.command()
def fix(
    repo: Path,
    task: str = DEFAULT_TASK,
    model: Annotated[str | None, typer.Option(help="LiteLLM model string")] = None,
    max_turns: int = 20,
    budget: Annotated[float, typer.Option(help="Stop once this much USD is spent")] = 0.50,
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    chosen_model = model or settings.default_model
    try:
        result = run_agent(
            chosen_model,
            task,
            Workspace(repo),
            max_turns=max_turns,
            budget_usd=budget,
            api_base=api_base_for(chosen_model, settings),
        )
    except AgentError as e:
        typer.echo(f"Agent gave up: {e}", err=True)
        raise typer.Exit(1) from e
    typer.echo(f"\n{result.summary}\n\n{result.turns} turns . ${result.cost_usd:.4f}")
