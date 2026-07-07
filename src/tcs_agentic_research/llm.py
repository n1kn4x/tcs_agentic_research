"""vLLM/OpenAI-compatible model router with structured-output logging."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Any, TypeVar

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from .artifact_store import ArtifactStore
from .schemas import AppConfig, ModelCallRecord, ModelProfile, RouterSettings

def schema_placeholder(schema: type[BaseModel]) -> str:
    return "{{" + schema.__name__ + "}}"


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
        messages: list[dict[str, str]],
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
        messages: list[dict[str, str]],
        schema: type[T],
        mock_output: T | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Complete a structured call and validate it against ``schema``.

        ``mock_output`` is only usable in dry-run mode. In real runs, schema/API
        failures are logged and raised after retry/repair attempts; mock outputs
        are never returned as a recovery path.
        """
        profile_name, profile = self.select_profile(task_type)
        started = time.perf_counter()
        failures: list[str] = []
        response_payload: dict[str, Any] | None = None

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
        if mock_output is not None:
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
        for attempt in range(self.settings.max_retries + 1):
            try:
                response_payload = self._post_chat_completion(
                    profile,
                    messages,
                    temperature=profile.temperature if temperature is None else temperature,
                    max_tokens=profile.max_tokens if max_tokens is None else max_tokens,
                    json_schema=schema.model_json_schema(),
                )
                content = response_payload["choices"][0]["message"]["content"]
                payload = _extract_json(content)
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
                # Ask the same endpoint to repair, but include the invalid response,
                # the concrete validation error, and the schema. Without this context
                # the next call cannot actually know what to fix if guided decoding is
                # unsupported or ignored by the backend.
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

    def _post_chat_completion(
        self,
        profile: ModelProfile,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        json_schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": profile.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_schema is not None:
            body["response_format"] = {"type": "json_object"}
            # vLLM supports guided decoding through guided_json in recent versions.
            body["guided_json"] = json_schema
        headers = {"Authorization": f"Bearer {profile.api_key}"}
        url = profile.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.settings.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
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


def _prepare_structured_messages(
    messages: list[dict[str, str]], schema: type[BaseModel]
) -> list[dict[str, str]]:
    """Inject schema documentation into prompt placeholders or append it as a fallback.

    Prompt files may contain a schema-specific placeholder such as
    ``{{InitializationBundle}}`` where the complete output contract should appear.
    The replacement includes the full Pydantic JSON Schema, including recursively
    referenced ``$defs`` and enum values.  If a prompt has no matching placeholder,
    keep the historical behavior of appending a schema contract as an extra message.
    """
    schema_text = _schema_prompt(schema)
    placeholder = schema_placeholder(schema)
    rendered: list[dict[str, str]] = []
    found_placeholder = False
    for message in messages:
        content = message.get("content", "")
        if placeholder in content:
            found_placeholder = True
            content = content.replace(placeholder, schema_text)
        rendered.append({**message, "content": content})
    if found_placeholder:
        return rendered
    return _with_structured_output_instruction(rendered, schema)


def _schema_prompt(schema: type[BaseModel]) -> str:
    """Render a complete, prompt-friendly schema contract for a Pydantic model."""
    schema_dict = schema.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2, sort_keys=True)
    required = schema_dict.get("required", [])
    properties = schema_dict.get("properties", {})
    field_lines = []
    for name, spec in properties.items():
        marker = "required" if name in required else "optional"
        field_lines.append(f"- {name}: {marker}; {_describe_schema_fragment(spec)}")
    if not field_lines:
        field_lines.append("- [schema has no declared top-level fields]")
    return (
        f"Structured-output contract for schema `{schema.__name__}`.\n"
        "Return exactly one JSON object. Do not wrap it in another key, Markdown, prose, "
        "an error object, or a reasoning transcript. Do not add fields not present in "
        "the schema.\n"
        "Fields marked with `additionalProperties: false` reject any extra keys.\n\n"
        "Top-level fields:\n"
        f"{chr(10).join(field_lines)}\n\n"
        "Complete JSON Schema, including recursive `$defs` sub-schemas and enum values:\n"
        f"{schema_json}"
    )


def _describe_schema_fragment(fragment: dict[str, Any]) -> str:
    """Return a compact description for one JSON-schema fragment."""
    if "enum" in fragment:
        return "one of " + ", ".join(json.dumps(value) for value in fragment["enum"])
    if "const" in fragment:
        return "constant " + json.dumps(fragment["const"])
    if "$ref" in fragment:
        return "object " + fragment["$ref"].rsplit("/", maxsplit=1)[-1]
    if "anyOf" in fragment:
        return " or ".join(_describe_schema_fragment(item) for item in fragment["anyOf"])
    if "allOf" in fragment:
        return " and ".join(_describe_schema_fragment(item) for item in fragment["allOf"])
    fragment_type = fragment.get("type")
    if fragment_type == "array":
        items = fragment.get("items", {})
        if isinstance(items, dict):
            return "array of " + _describe_schema_fragment(items)
        return "array"
    if isinstance(fragment_type, list):
        return " or ".join(str(item) for item in fragment_type)
    if fragment_type:
        return str(fragment_type)
    return fragment.get("title", "object")


def _extract_assistant_content(response_payload: dict[str, Any] | None, *, limit: int = 8000) -> str:
    if response_payload is None:
        return "[No response payload was available from the previous attempt.]"
    try:
        content = response_payload["choices"][0]["message"].get("content", "")
    except Exception:  # noqa: BLE001 - best-effort diagnostic context for repair prompts
        content = json.dumps(response_payload, sort_keys=True)
    text = str(content).strip() or "[The previous attempt returned empty assistant content.]"
    return _truncate(text, limit)


def _with_structured_output_instruction(
    messages: list[dict[str, str]], schema: type[BaseModel]
) -> list[dict[str, str]]:
    """Append a concrete schema instruction for backends that ignore guided_json."""
    return [*messages, {"role": "user", "content": _schema_prompt(schema)}]


def _structured_repair_prompt(schema: type[BaseModel], exc: Exception) -> str:
    return (
        f"The previous response did not validate for schema `{schema.__name__}`.\n\n"
        "Validation error:\n"
        f"{_truncate(str(exc), 6000)}\n\n"
        "Return ONLY corrected JSON. Do not include Markdown, prose, or an `error` object. "
        "Use exactly the field names, required fields, and enum values in this schema. "
        "Do not add extra fields.\n\n"
        f"{_schema_prompt(schema)}"
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
