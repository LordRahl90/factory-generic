from typing import Any


def _tool(name: str, description: str, **params: str) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": {k: {"type": "string", "description": v} for k, v in params.items()},
                "required": list(params),
            },
        },
    }


SCHEMAS = [
    _tool("list_files", "Lists every file in the repository"),
    _tool("read_file", "Read a text file", path="Path relative to the repository root"),
    _tool(
        "write_file",
        "Replace a file's entire content",
        path="Path relative to the repository root",
        content="The complete new file content",
    ),
    _tool(
        "run",
        "Run a command in the repository root, without a shell, Returns exit code and output.",
        command="For example: pytest -q",
    ),
]

TOOL_NAMES = frozenset(s["function"]["name"] for s in SCHEMAS)
