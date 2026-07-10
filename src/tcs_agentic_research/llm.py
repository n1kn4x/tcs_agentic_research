"""vLLM/OpenAI-compatible model router with structured-output logging."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from .artifact_store import ArtifactStore, to_plain
from .schemas import AppConfig, ModelCallRecord, ModelProfile, RouterSettings

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class StructuredLLMError(RuntimeError):
    pass


class LLMRouter:
    """Route tasks to local vLLM servers and validate structured outputs."""

    def __init__(self, settings: RouterSettings, *, store: ArtifactStore | None = None, dry_run: bool = False):
        self.settings = settings
        self.store = store
        self.dry_run = dry_run

    @classmethod
    def from_config_file(
        cls, path: str | Path | None, *, store: ArtifactStore | None = None, dry_run: bool = False
    ) -> "LLMRouter":
        if path is None:
            settings = RouterSettings(
                default_task="deep",
                profiles={
                    "deep": ModelProfile(
                        model="deep-reasoner",
                        base_url="http://localhost:8000/v1",
                        api_key="EMPTY",
                        temperature=0.2,
                        max_tokens=4096,
                        task_types=[],
                    )
                },
            )
            return cls(settings, store=store, dry_run=dry_run)
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        config = AppConfig.model_validate(data)
        return cls(config.router, store=store, dry_run=dry_run)

    def select_profile(self, task_type: str) -> tuple[str, ModelProfile]:
        for name, profile in self.settings.profiles.items():
            if task_type in profile.task_types:
                return name, profile
        if self.settings.default_task in self.settings.profiles:
            return self.settings.default_task, self.settings.profiles[self.settings.default_task]
        first_name = next(iter(self.settings.profiles))
        return first_name, self.settings.profiles[first_name]

    def complete_text(
        self,
        *,
        task_type: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self.dry_run:
            raise StructuredLLMError("LLMRouter is in dry-run mode and no mock output was supplied")
        profile_name, profile = self.select_profile(task_type)
        started = time.perf_counter()
        failures: list[str] = []
        response_payload: dict[str, Any] | None = None
        try:
            response_payload = self._post_chat_completion(
                profile,
                messages,
                temperature=profile.temperature if temperature is None else temperature,
                max_tokens=profile.max_tokens if max_tokens is None else max_tokens,
            )
            content = response_payload["choices"][0]["message"]["content"]
            self._log_call(task_type, profile_name, profile, started, True, None, failures, response_payload)
            return content
        except Exception as exc:  # noqa: BLE001 - record exact model failure
            failures.append(type(exc).__name__ + ": " + str(exc))
            self._log_call(task_type, profile_name, profile, started, False, None, failures, response_payload)
            raise

    def complete_structured(
        self,
        *,
        task_type: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        mock_output: T | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Complete a structured call and validate it against ``schema``.

        ``mock_output`` is only usable in dry-run mode. In real runs, schema placeholders
        in prompts are expanded for model-visible instructions, and the output schema is
        sent through vLLM ``guided_json``/``response_format`` when available. Schema/API
        failures are logged and raised after retry/repair attempts; mock outputs are never
        returned as a recovery path.
        """
        profile_name, profile = self.select_profile(task_type)
        started = time.perf_counter()
        failures: list[str] = []
        response_payload: dict[str, Any] | None = None

        if mock_output is not None and not self.dry_run:
            failures.append("real_run: mock_output was supplied; refusing to use it")
            self._log_call(
                task_type,
                profile_name,
                profile,
                started,
                False,
                schema.__name__,
                failures,
                None,
            )
            _log_structured_failure(task_type, schema.__name__, failures)
            raise ValueError("mock_output may only be supplied when LLMRouter.dry_run is true")

        messages = _prepare_structured_messages(messages, schema)

        if self.dry_run:
            if mock_output is not None:
                failures.append("dry_run: returned supplied mock output without calling an LLM")
                self._log_call(
                    task_type,
                    profile_name,
                    profile,
                    started,
                    True,
                    schema.__name__,
                    failures,
                    None,
                    used_mock_output=True,
                )
                return mock_output
            failures.append("dry_run: no mock output supplied")
            self._log_call(
                task_type,
                profile_name,
                profile,
                started,
                False,
                schema.__name__,
                failures,
                None,
            )
            raise StructuredLLMError(
                f"dry-run router cannot synthesize {schema.__name__}; provide a mock_output"
            )

        for attempt in range(self.settings.max_retries + 1):
            try:
                response_payload = self._post_chat_completion(
                    profile,
                    messages,
                    temperature=profile.temperature if temperature is None else temperature,
                    max_tokens=profile.max_tokens if max_tokens is None else max_tokens,
                    json_schema=_llm_json_schema(schema),
                )
                content = response_payload["choices"][0]["message"]["content"]
                payload = _extract_json(content)
                _strip_system_owned_payload_fields(payload)
                result = schema.model_validate(payload)
                self._log_call(
                    task_type,
                    profile_name,
                    profile,
                    started,
                    True,
                    schema.__name__,
                    failures,
                    response_payload,
                )
                return result
            except (ValidationError, json.JSONDecodeError, KeyError) as exc:
                failures.append(f"attempt_{attempt}: structured_output_invalid: {exc}")
                logger.warning(
                    "Structured output validation failed for task_type=%s schema=%s "
                    "attempt=%s/%s; retrying with repair prompt. Error: %s",
                    task_type,
                    schema.__name__,
                    attempt + 1,
                    self.settings.max_retries + 1,
                    _truncate(str(exc), 1000),
                )
                # Ask the same endpoint to repair with the invalid response and
                # concrete validation error. The next call still sends guided_json
                # with the structured-call output schema.
                messages = messages + [
                    {
                        "role": "assistant",
                        "content": _extract_assistant_content(response_payload),
                    },
                    {
                        "role": "user",
                        "content": _structured_repair_prompt(schema, exc),
                    },
                ]
            except Exception as exc:  # noqa: BLE001 - exact failure recorded for auditability
                failures.append(f"attempt_{attempt}: {type(exc).__name__}: {exc}")
                if attempt >= self.settings.max_retries:
                    self._log_call(
                        task_type,
                        profile_name,
                        profile,
                        started,
                        False,
                        schema.__name__,
                        failures,
                        response_payload,
                    )
                    _log_structured_failure(task_type, schema.__name__, failures)
                    raise StructuredLLMError("; ".join(failures)) from exc

        self._log_call(
            task_type,
            profile_name,
            profile,
            started,
            False,
            schema.__name__,
            failures,
            response_payload,
        )
        _log_structured_failure(task_type, schema.__name__, failures)
        raise StructuredLLMError("; ".join(failures))

    def complete_structured_with_tools(
        self,
        *,
        task_type: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_executors: dict[str, Callable[[dict[str, Any]], Any]],
        schema: type[T],
        final_tool_name: str,
        mock_output: T | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[T, dict[str, Any]]:
        """Complete a structured task through the OpenAI/vLLM tool-call interface.

        The model may use ordinary tools for external observations and must finish by calling
        ``final_tool_name`` with arguments that validate against ``schema``. Raw assistant
        content/reasoning is intentionally not preserved in the returned trace; the trace contains
        only tool call IDs, tool names, arguments, observations, validation errors, and
        finalization metadata. Assistant content JSON is not accepted as a fallback here; agents
        using this method have a single finalization protocol: the final tool call.
        """
        profile_name, profile = self.select_profile(task_type)
        started = time.perf_counter()
        failures: list[str] = []
        response_payload: dict[str, Any] | None = None
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        trace: dict[str, Any] = {
            "task_type": task_type,
            "schema": schema.__name__,
            "final_tool_name": final_tool_name,
            "private_reasoning": "redacted_not_logged_or_replayed",
            "tool_calls": [],
        }

        if mock_output is not None and not self.dry_run:
            failures.append("real_run: mock_output was supplied; refusing to use it")
            self._log_call(
                task_type,
                profile_name,
                profile,
                started,
                False,
                schema.__name__,
                failures,
                None,
            )
            _log_structured_failure(task_type, schema.__name__, failures)
            raise ValueError("mock_output may only be supplied when LLMRouter.dry_run is true")

        messages = _prepare_structured_messages(messages, schema)

        if self.dry_run:
            if mock_output is not None:
                failures.append("dry_run: returned supplied mock output without calling an LLM")
                trace["finalization"] = {"mode": "dry_run_mock_output"}
                self._log_call(
                    task_type,
                    profile_name,
                    profile,
                    started,
                    True,
                    schema.__name__,
                    failures,
                    None,
                    used_mock_output=True,
                )
                return mock_output, trace
            failures.append("dry_run: no mock output supplied")
            self._log_call(
                task_type,
                profile_name,
                profile,
                started,
                False,
                schema.__name__,
                failures,
                None,
            )
            raise StructuredLLMError(
                f"dry-run router cannot synthesize {schema.__name__}; provide a mock_output"
            )

        if not profile.supports_tools:
            failures.append(f"profile `{profile_name}` does not declare supports_tools=true")
            self._log_call(
                task_type,
                profile_name,
                profile,
                started,
                False,
                schema.__name__,
                failures,
                None,
            )
            _log_structured_failure(task_type, schema.__name__, failures)
            raise StructuredLLMError("; ".join(failures))

        if _find_tool(tools, final_tool_name) is None:
            failures.append(f"final tool `{final_tool_name}` was not included in tools")
            self._log_call(
                task_type,
                profile_name,
                profile,
                started,
                False,
                schema.__name__,
                failures,
                None,
            )
            _log_structured_failure(task_type, schema.__name__, failures)
            raise StructuredLLMError("; ".join(failures))

        history = [dict(message) for message in messages]
        turn_index = 0

        while True:
            turn_index += 1
            try:
                response_payload = self._post_chat_completion(
                    profile,
                    history,
                    temperature=profile.temperature if temperature is None else temperature,
                    max_tokens=profile.max_tokens if max_tokens is None else max_tokens,
                    tools=tools,
                    tool_choice="auto",
                )
                _accumulate_usage(usage_totals, response_payload)
                message = response_payload["choices"][0]["message"]
                tool_calls = _message_tool_calls(message)

                if tool_calls:
                    history.append(_assistant_tool_call_message(tool_calls))
                    for tool_call in tool_calls:
                        call_id, tool_name, raw_arguments = _tool_call_parts(tool_call)
                        try:
                            arguments = _drop_private_tool_argument_fields(
                                _parse_tool_arguments(raw_arguments)
                            )
                        except Exception as exc:  # noqa: BLE001 - return as tool observation
                            failures.append(
                                f"turn_{turn_index}: tool_arguments_invalid for "
                                f"{tool_name or '<unknown>'}: {exc}"
                            )
                            observation = {
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            }
                            history.append(_tool_result_message(call_id, tool_name, observation))
                            trace["tool_calls"].append(
                                {
                                    "turn": turn_index,
                                    "call_id": call_id,
                                    "name": tool_name,
                                    "arguments": raw_arguments,
                                    "status": "argument_error",
                                    "observation": observation,
                                }
                            )
                            continue

                        if tool_name == final_tool_name:
                            try:
                                payload = to_plain(arguments)
                                _strip_system_owned_payload_fields(payload)
                                result = schema.model_validate(payload)
                                trace["finalization"] = {
                                    "turn": turn_index,
                                    "call_id": call_id,
                                    "mode": "final_tool_call",
                                    "tool_name": final_tool_name,
                                }
                                self._log_call(
                                    task_type,
                                    profile_name,
                                    profile,
                                    started,
                                    True,
                                    schema.__name__,
                                    failures,
                                    _usage_payload(usage_totals),
                                )
                                return result, trace
                            except (ValidationError, TypeError, ValueError) as exc:
                                failures.append(
                                    f"turn_{turn_index}: final_tool_payload_invalid: {exc}"
                                )
                                observation = {
                                    "status": "error",
                                    "error_type": type(exc).__name__,
                                    "error": _truncate(str(exc), 4000),
                                    "instruction": (
                                        f"Retry by calling `{final_tool_name}` with arguments "
                                        f"that validate as `{schema.__name__}`."
                                    ),
                                }
                                history.append(_tool_result_message(call_id, tool_name, observation))
                                trace["tool_calls"].append(
                                    {
                                        "turn": turn_index,
                                        "call_id": call_id,
                                        "name": tool_name,
                                        "arguments": to_plain(arguments),
                                        "status": "validation_error",
                                        "observation": observation,
                                    }
                                )
                                continue

                        observation = _execute_openai_tool(tool_name, arguments, tool_executors)
                        history.append(_tool_result_message(call_id, tool_name, observation))
                        trace["tool_calls"].append(
                            {
                                "turn": turn_index,
                                "call_id": call_id,
                                "name": tool_name,
                                "arguments": to_plain(arguments),
                                "status": observation.get("status", "ok")
                                if isinstance(observation, dict)
                                else "ok",
                                "observation": to_plain(observation),
                            }
                        )
                    continue

                content = _strip_reasoning_blocks(str(message.get("content") or "")).strip()
                if content:
                    failures.append(
                        f"turn_{turn_index}: assistant_content_ignored_until_final_tool"
                    )

                history.append(
                    {
                        "role": "user",
                        "content": (
                            f"Do not call more external tools unless necessary. Finish by calling "
                            f"`{final_tool_name}` with arguments that validate as "
                            f"`{schema.__name__}`. Do not reveal private reasoning."
                        ),
                    }
                )
            except Exception as exc:  # noqa: BLE001 - exact failure recorded for auditability
                failures.append(f"turn_{turn_index}: {type(exc).__name__}: {exc}")
                self._log_call(
                    task_type,
                    profile_name,
                    profile,
                    started,
                    False,
                    schema.__name__,
                    failures,
                    _usage_payload(usage_totals, fallback=response_payload),
                )
                _log_structured_failure(task_type, schema.__name__, failures)
                raise StructuredLLMError("; ".join(failures)) from exc

    def _post_chat_completion(
        self,
        profile: ModelProfile,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        body.update(profile.extra_body)
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
            # vLLM supports guided decoding through guided_json in recent versions.
            # In-prompt schema placeholders remain useful because backend support and
            # adherence vary across vLLM/model versions.
            body["guided_json"] = json_schema
        if tools is not None:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        headers = {"Authorization": f"Bearer {profile.api_key}"}
        url = profile.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.settings.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=body)
            _raise_for_status_with_body(response)
            return response.json()

    def _log_call(
        self,
        task_type: str,
        profile_name: str,
        profile: ModelProfile,
        started: float,
        structured_valid: bool,
        schema_name: str | None,
        failures: list[str],
        response_payload: dict[str, Any] | None,
        *,
        used_mock_output: bool = False,
    ) -> None:
        usage = (response_payload or {}).get("usage", {})
        record = ModelCallRecord(
            task_type=task_type,
            profile_name=profile_name,
            model=profile.model,
            latency_seconds=round(time.perf_counter() - started, 4),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            structured_schema=schema_name,
            structured_output_valid=structured_valid,
            execution_mode="dry_run" if self.dry_run else "real",
            used_mock_output=used_mock_output,
            failure_modes=failures,
        )
        if self.store is not None:
            self.store.append_model_call(record)


def openai_tool_from_schema(
    name: str,
    description: str,
    schema: type[BaseModel],
    *,
    strip_system_owned_fields: bool = True,
) -> dict[str, Any]:
    """Build an OpenAI-compatible function tool from a Pydantic schema."""
    parameters = _llm_json_schema(schema) if strip_system_owned_fields else schema.model_json_schema()
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _find_tool(tools: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in tools:
        function = tool.get("function") if isinstance(tool, dict) else None
        if isinstance(function, dict) and function.get("name") == name:
            return tool
    return None


def _message_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    tool_calls = message.get("tool_calls") or []
    if not isinstance(tool_calls, list):
        return []
    return [call for call in tool_calls if isinstance(call, dict)]


def _assistant_tool_call_message(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    return {"role": "assistant", "content": "", "tool_calls": [_sanitize_tool_call(call) for call in tool_calls]}


def _sanitize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    call_id, name, arguments = _tool_call_parts(tool_call)
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _tool_call_parts(tool_call: dict[str, Any]) -> tuple[str, str, Any]:
    function = tool_call.get("function") if isinstance(tool_call, dict) else None
    function = function if isinstance(function, dict) else {}
    call_id = str(tool_call.get("id") or f"call_{abs(hash(json.dumps(tool_call, sort_keys=True, default=str)))}")
    name = str(function.get("name") or "")
    arguments = function.get("arguments", {})
    return call_id, name, arguments


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, dict):
        return arguments
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        text = _strip_reasoning_blocks(arguments).strip()
        if not text:
            return {}
        payload = _extract_json(text)
        if isinstance(payload, dict):
            return payload
        raise TypeError("tool arguments must decode to a JSON object")
    raise TypeError(f"tool arguments must be a dict or JSON string, got {type(arguments).__name__}")


_PRIVATE_TOOL_ARGUMENT_FIELDS = {
    "analysis",
    "chain_of_thought",
    "internal_reasoning",
    "private_reasoning",
    "rationale",
    "reasoning",
    "scratchpad",
    "thought",
    "thoughts",
}


def _drop_private_tool_argument_fields(node: Any) -> Any:
    if isinstance(node, dict):
        return {
            key: _drop_private_tool_argument_fields(value)
            for key, value in node.items()
            if key not in _PRIVATE_TOOL_ARGUMENT_FIELDS
        }
    if isinstance(node, list):
        return [_drop_private_tool_argument_fields(item) for item in node]
    return node


def _execute_openai_tool(
    tool_name: str,
    arguments: dict[str, Any],
    executors: dict[str, Callable[[dict[str, Any]], Any]],
) -> dict[str, Any]:
    executor = executors.get(tool_name)
    if executor is None:
        return {
            "status": "error",
            "error_type": "UnknownTool",
            "error": f"No executor is registered for tool `{tool_name}`.",
        }
    try:
        result = to_plain(executor(arguments))
        if isinstance(result, dict):
            return result
        return {"status": "ok", "result": result}
    except Exception as exc:  # noqa: BLE001 - tool failures are observations for the model
        return {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}


def _tool_result_message(call_id: str, tool_name: str, observation: Any) -> dict[str, Any]:
    return {
        "role": "tool",
        "tool_call_id": call_id,
        "name": tool_name,
        "content": json.dumps(to_plain(observation), ensure_ascii=False, separators=(",", ":")),
    }


def _accumulate_usage(totals: dict[str, int], response_payload: dict[str, Any] | None) -> None:
    usage = (response_payload or {}).get("usage", {})
    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        value = usage.get(key)
        if isinstance(value, int):
            totals[key] = totals.get(key, 0) + value


def _usage_payload(
    totals: dict[str, int], *, fallback: dict[str, Any] | None = None
) -> dict[str, Any] | None:
    if any(totals.get(key) for key in ["prompt_tokens", "completion_tokens", "total_tokens"]):
        return {"usage": {key: value for key, value in totals.items() if value}}
    return fallback


def _raise_for_status_with_body(response: httpx.Response) -> None:
    """Raise HTTP errors with the provider's response body included."""
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        body = _format_http_error_body(response)
        message = f"{exc}\nvLLM/OpenAI-compatible server response body: {body}"
        raise httpx.HTTPStatusError(message, request=exc.request, response=exc.response) from exc


def _format_http_error_body(response: httpx.Response, *, limit: int = 5000) -> str:
    text = response.text.strip()
    if not text:
        return "<empty>"
    try:
        payload = response.json()
    except ValueError:
        return _truncate(text, limit)
    return _truncate(json.dumps(payload, ensure_ascii=False, sort_keys=True), limit)


def schema_placeholder(schema: type[BaseModel] | str) -> str:
    """Return the named prompt placeholder for a structured-output schema."""
    name = schema if isinstance(schema, str) else schema.__name__
    return "{{" + name + "}}"


def _schema_prompt(schema: type[BaseModel]) -> str:
    """Render the model-facing JSON Schema for structured output prompts."""
    schema_json = json.dumps(_llm_json_schema(schema), indent=2, sort_keys=True)
    return (
        f"Complete JSON Schema for `{schema.__name__}`.\n"
        "Return ONLY a JSON value that validates against this schema. "
        "System-owned fields such as IDs, timestamps, hashes, and artifact references "
        "are intentionally omitted; the application fills them after validation. "
        "Do not invent IDs. If a reference list requires an ID not present in the input, "
        "leave it empty. Do not include Markdown, prose, comments, an `error` object, "
        "or extra fields.\n\n"
        "```json\n"
        f"{schema_json}\n"
        "```"
    )


_SCHEMA_PLACEHOLDER_RE = re.compile(r"\{\{([A-Za-z_][A-Za-z0-9_]*)\}\}")


def _prepare_structured_messages(
    messages: list[dict[str, str]], schema: type[BaseModel]
) -> list[dict[str, str]]:
    """Replace schema placeholders in structured-call messages."""
    rendered: list[dict[str, str]] = []
    for message in messages:
        rendered_message = dict(message)
        content = rendered_message.get("content", "")
        if isinstance(content, str):
            rendered_message["content"] = _render_schema_placeholders(content, output_schema=schema)
        rendered.append(rendered_message)
    return rendered


def _render_schema_placeholders(content: str, *, output_schema: type[BaseModel]) -> str:
    def replace(match: re.Match[str]) -> str:
        schema_name = match.group(1)
        schema = _resolve_schema_placeholder(schema_name, output_schema=output_schema)
        return _schema_prompt(schema)

    return _SCHEMA_PLACEHOLDER_RE.sub(replace, content)


def _resolve_schema_placeholder(
    schema_name: str, *, output_schema: type[BaseModel]
) -> type[BaseModel]:
    candidates: list[type[BaseModel]] = []
    if output_schema.__name__ == schema_name:
        candidates.append(output_schema)
    for candidate in _iter_pydantic_model_subclasses():
        if candidate.__name__ == schema_name:
            candidates.append(candidate)

    unique_candidates = list(dict.fromkeys(candidates))
    if not unique_candidates:
        raise StructuredLLMError(
            f"Unknown schema placeholder `{{{{{schema_name}}}}}`: "
            f"no Pydantic schema named `{schema_name}` is available."
        )
    if len(unique_candidates) > 1:
        raise StructuredLLMError(
            f"Ambiguous schema placeholder `{{{{{schema_name}}}}}`: "
            f"multiple Pydantic schemas named `{schema_name}` are available."
        )
    return unique_candidates[0]


def _iter_pydantic_model_subclasses() -> list[type[BaseModel]]:
    seen: set[type[BaseModel]] = set()
    stack = list(BaseModel.__subclasses__())
    ordered: list[type[BaseModel]] = []
    while stack:
        candidate = stack.pop()
        if candidate in seen:
            continue
        seen.add(candidate)
        ordered.append(candidate)
        stack.extend(candidate.__subclasses__())
    return ordered


def _extract_assistant_content(response_payload: dict[str, Any] | None, *, limit: int = 8000) -> str:
    if response_payload is None:
        return "[No response payload was available from the previous attempt.]"
    try:
        content = response_payload["choices"][0]["message"].get("content", "")
    except Exception:  # noqa: BLE001 - best-effort diagnostic context for repair prompts
        content = json.dumps(response_payload, sort_keys=True)
    text = _strip_reasoning_blocks(str(content)).strip() or "[The previous attempt returned empty assistant content.]"
    return _truncate(text, limit)


def _structured_repair_prompt(schema: type[BaseModel], exc: Exception) -> str:
    return (
        f"The previous response did not validate for schema `{schema.__name__}`.\n\n"
        "Validation error:\n"
        f"{_truncate(str(exc), 6000)}\n\n"
        "Return ONLY corrected JSON that validates against the schema already included "
        "in the prompt and, when supported, the guided schema provided by the API. "
        "Do not include Markdown, prose, or an `error` object. Do not add extra fields."
    )


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n...[truncated {omitted} characters]"


def _log_structured_failure(task_type: str, schema_name: str, failures: list[str]) -> None:
    logger.error(
        "Structured LLM call failed permanently for task_type=%s schema=%s after retries. "
        "No mock output is allowed in real runs. Failure modes: %s",
        task_type,
        schema_name,
        _truncate("; ".join(failures), 4000),
    )


SYSTEM_OWNED_SCHEMA_FIELDS = {
    "answer_id",
    "artifact_refs",
    "call_id",
    "claim_id",
    "candidate_id",
    "created_at",
    "dag_id",
    "duplicate_id",
    "event_id",
    "evidence_id",
    "extract_id",
    "goal_id",
    "imported_at",
    "metadata_path",
    "obligation_id",
    "paper_id",
    "proposal_id",
    "quote_id",
    "related_proposal_ids",
    "related_report_ids",
    "depends_on_claim_ids",
    "supersedes_claim_ids",
    "report_id",
    "result_id",
    "run_id",
    "sha256",
    "source_refs",
    "statement_id",
    "task_id",
    "text_artifact_ref",
    "updated_at",
    "verdict_id",
}


def _llm_json_schema(schema: type[BaseModel]) -> dict[str, Any]:
    payload = schema.model_json_schema()
    _strip_system_owned_schema_fields(payload)
    return payload


def _strip_system_owned_schema_fields(node: Any) -> None:
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            for field_name in SYSTEM_OWNED_SCHEMA_FIELDS:
                properties.pop(field_name, None)
            required = node.get("required")
            if isinstance(required, list):
                node["required"] = [
                    field_name
                    for field_name in required
                    if field_name not in SYSTEM_OWNED_SCHEMA_FIELDS
                ]
        for value in node.values():
            _strip_system_owned_schema_fields(value)
    elif isinstance(node, list):
        for item in node:
            _strip_system_owned_schema_fields(item)


def _strip_system_owned_payload_fields(node: Any) -> None:
    if isinstance(node, dict):
        for field_name in SYSTEM_OWNED_SCHEMA_FIELDS:
            node.pop(field_name, None)
        for value in node.values():
            _strip_system_owned_payload_fields(value)
    elif isinstance(node, list):
        for item in node:
            _strip_system_owned_payload_fields(item)


def _extract_json(content: str) -> Any:
    text = _strip_reasoning_blocks(content).strip()
    if text.startswith("```"):
        # Strip common fenced JSON blocks.
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as original_error:
        decoder = json.JSONDecoder()
        for start, ch in enumerate(text):
            if ch not in "[{":
                continue
            try:
                payload, _end = decoder.raw_decode(text[start:])
                return payload
            except json.JSONDecodeError:
                continue
        raise original_error


def _strip_reasoning_blocks(content: str) -> str:
    """Remove common model-internal reasoning wrappers before JSON extraction."""
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text
