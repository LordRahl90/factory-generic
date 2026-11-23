from typing import Annotated

import typer
from dotenv import load_dotenv

from factory.adapters.llm import ask
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
