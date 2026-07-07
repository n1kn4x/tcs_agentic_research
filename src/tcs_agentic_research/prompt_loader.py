"""Prompt loading utilities.

Prompts are plain Markdown files in ``src/tcs_agentic_research/prompts`` so they are easy
to inspect, version, and edit. Structured-output prompts may include a schema-named
placeholder such as ``{{InitializationBundle}}``;
:meth:`tcs_agentic_research.llm.LLMRouter.complete_structured` replaces it with the
corresponding recursive JSON Schema before calling the model.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


PROMPT_PACKAGE = "tcs_agentic_research.prompts"


def load_prompt(name: str, *, override_dir: str | Path | None = None) -> str:
    if not name.endswith(".md"):
        name = f"{name}.md"
    if override_dir is not None:
        path = Path(override_dir) / name
        if path.exists():
            return path.read_text(encoding="utf-8")
    return resources.files(PROMPT_PACKAGE).joinpath(name).read_text(encoding="utf-8")


def render_prompt(name: str, *, override_dir: str | Path | None = None, **kwargs: object) -> str:
    text = load_prompt(name, override_dir=override_dir)
    for key, value in kwargs.items():
        text = text.replace("{{" + key + "}}", str(value))
    return text
