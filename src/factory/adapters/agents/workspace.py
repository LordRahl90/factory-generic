import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

MAX_OUTPUT = 4000


# This is the store for the tools we are going to expose
class Workspace:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        # this is to check if the provided file belongs in the workspace
        if not p.is_relative_to(self.root):
            raise PermissionError(f"{rel} is outside the workspace")
        return p

    def read_file(self, path: str) -> str:
        return self._path(path).read_text()

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {len(content)} characters to {path}"

    def list_files(self) -> str:
        done = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
            cwd=self.root,
            capture_output=True,
            text=True,
        )
        files = sorted(set(done.stdout.splitlines()))
        return "\n".join(files[:2000]) if done.returncode == 0 else f"error: {done.stderr.strip()}"

    def run(self, command: str) -> str:
        # executes the command
        try:
            done = subprocess.run(
                shlex.split(command), cwd=self.root, capture_output=True, text=True, timeout=60
            )
        except subprocess.TimeoutExpired:
            return "error: command timed out after 60s"
        out = f"exit code {done.returncode}\n{done.stdout}{done.stderr}"
        # this executes the command and truncates the output to max output.
        return out if len(out) <= MAX_OUTPUT else out[:MAX_OUTPUT] + "\n[truncated]"

    def call(self, name: str, args: dict[str, Any], allowed: frozenset[str] | None = None) -> str:
        tools: dict[str, Callable[..., str]] = {
            "read_file": self.read_file,
            "write_file": self.write_file,
            "list_files": self.list_files,
            "run": self.run,
        }
        if name not in tools:
            return f"error: tool {name!r} is not available here"
        try:
            return tools[name](**args)
        except Exception as e:
            return f"error: {type(e).__name__}: {e}"
