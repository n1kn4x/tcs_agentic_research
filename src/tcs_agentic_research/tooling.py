"""Small helpers for exposing per-agent OpenAI/vLLM toolsets."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .llm import openai_tool_from_schema


@dataclass(frozen=True)
class AgentTool:
    """One model-visible tool plus an optional local executor.

    Tool access remains per-agent: each LLM call receives exactly the tools from the
    ``Toolset`` passed by that agent. A final submission tool intentionally has no executor;
    ``LLMRouter.complete_structured_with_tools`` handles it specially.
    """

    name: str
    description: str
    args_schema: type[BaseModel]
    executor: Callable[[dict[str, Any]], Any] | None = None
    strip_system_owned_fields: bool = True

    def openai_tool(self) -> dict[str, Any]:
        return openai_tool_from_schema(
            self.name,
            self.description,
            self.args_schema,
            strip_system_owned_fields=self.strip_system_owned_fields,
        )


class Toolset:
    """A request-scoped collection of tools for one agent call."""

    def __init__(self, tools: Iterable[AgentTool] = ()):
        self._tools = list(tools)
        names = [tool.name for tool in self._tools]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate tool names in toolset: {', '.join(duplicates)}")

    def __iter__(self):
        return iter(self._tools)

    def __add__(self, other: "Toolset") -> "Toolset":
        return Toolset([*self._tools, *other._tools])

    def openai_tools(self) -> list[dict[str, Any]]:
        return [tool.openai_tool() for tool in self._tools]

    def executors(self) -> dict[str, Callable[[dict[str, Any]], Any]]:
        return {tool.name: tool.executor for tool in self._tools if tool.executor is not None}


def final_submission_tool(
    name: str,
    description: str,
    schema: type[BaseModel],
) -> AgentTool:
    """Build the final tool that commits an agent's structured output."""

    return AgentTool(name=name, description=description, args_schema=schema, executor=None)
