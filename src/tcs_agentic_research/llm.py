"""Bounded OpenAI-compatible model calls.

There is deliberately no model-driven tool loop in this module.  A model receives one bounded
request and returns either text or one JSON value.  The deterministic engine executes actions.
"""

from __future__ import annotations

import json
import re
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TypeVar

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from .artifact_store import ArtifactStore
from .schemas import (
    AppConfig,
    CoreSettings,
    ExperimenterSettings,
    LeapSettings,
    ModelCallRecord,
    ModelProfile,
    RouterSettings,
)

T = TypeVar("T", bound=BaseModel)


class StructuredLLMError(RuntimeError):
    pass


class ModelBudgetExceeded(StructuredLLMError):
    pass


class InputBudgetExceeded(StructuredLLMError):
    pass


class _ResponseValidationError(StructuredLLMError):
    def __init__(self, content: str, error: Exception):
        super().__init__(str(error))
        self.content = content
        self.validation_error = error


class LLMRouter:
    """Route fresh, bounded calls and validate structured responses."""

    def __init__(
        self,
        settings: RouterSettings,
        *,
        store: ArtifactStore | None = None,
        dry_run: bool = False,
        experimenter: ExperimenterSettings | None = None,
        core: CoreSettings | None = None,
        leap: LeapSettings | None = None,
    ):
        self.settings = settings
        self.store = store
        self.dry_run = dry_run
        self.experimenter = experimenter
        self.core = core or CoreSettings()
        self.leap = leap or LeapSettings()
        self._step_id = ""
        self._remaining_calls: int | None = None

    @classmethod
    def from_config_file(
        cls,
        path: str | Path | None,
        *,
        store: ArtifactStore | None = None,
        dry_run: bool = False,
    ) -> "LLMRouter":
        if path is None:
            config = AppConfig(
                router=RouterSettings(
                    default_profile="reasoning",
                    repair_profile="reasoning",
                    profiles={
                        "reasoning": ModelProfile(
                            model="qwen-research",
                            task_types=[],
                            extra_body={
                                "top_p": 0.95,
                                "top_k": 20,
                                "presence_penalty": 0.0,
                                "chat_template_kwargs": {
                                    "enable_thinking": True,
                                    "preserve_thinking": False,
                                },
                            },
                        )
                    },
                )
            )
        else:
            raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
            config = AppConfig.model_validate(raw)
        return cls(
            config.router,
            store=store,
            dry_run=dry_run,
            experimenter=config.experimenter,
            core=config.core,
            leap=config.leap,
        )

    def select_profile(self, task_type: str) -> tuple[str, ModelProfile]:
        for name, profile in self.settings.profiles.items():
            if task_type in profile.task_types:
                return name, profile
        name = self.settings.default_profile
        if name not in self.settings.profiles:
            raise StructuredLLMError(f"Unknown default model profile `{name}`")
        return name, self.settings.profiles[name]

    @property
    def operation_budget_active(self) -> bool:
        return self._remaining_calls is not None

    @contextmanager
    def step_budget(self, step_id: str, *, max_calls: int | None = None) -> Iterator[None]:
        """Apply a hard model-call cap to one engine operation.

        Budgets may not be nested because nested accounting is easy to get subtly wrong.
        """
        if self._remaining_calls is not None:
            raise RuntimeError("model call budgets may not be nested")
        self._step_id = step_id
        self._remaining_calls = max_calls or self.core.max_model_calls_per_step
        try:
            yield
        finally:
            self._remaining_calls = None
            self._step_id = ""

    def complete_text(
        self,
        *,
        task_type: str,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        if self.dry_run:
            raise StructuredLLMError("dry-run text calls require a typed mock and are disabled")
        profile_name, profile = self.select_profile(task_type)
        response = self._call(
            task_type=task_type,
            profile_name=profile_name,
            profile=profile,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        try:
            return str(response["choices"][0]["message"].get("content") or "")
        except (KeyError, IndexError, TypeError) as exc:
            raise StructuredLLMError("provider response has no assistant content") from exc

    def complete_structured(
        self,
        *,
        task_type: str,
        messages: list[dict[str, Any]],
        schema: type[T],
        mock_output: T | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        allow_repair: bool = True,
    ) -> T:
        """Return one validated JSON object, with at most one fresh formatter repair call."""
        if self.dry_run:
            if mock_output is None:
                raise StructuredLLMError(
                    f"dry-run cannot synthesize {schema.__name__}; supply a deterministic mock"
                )
            return schema.model_validate(mock_output)
        if mock_output is not None:
            raise ValueError("mock_output is only allowed in dry-run mode")

        profile_name, profile = self.select_profile(task_type)
        try:
            response = self._call(
                task_type=task_type,
                profile_name=profile_name,
                profile=profile,
                messages=messages,
                schema=schema,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            result = response.pop("_validated_model")
            return schema.model_validate(result)
        except _ResponseValidationError as first_error:
            if self.settings.repair_attempts == 0 or not allow_repair:
                raise StructuredLLMError(
                    f"{schema.__name__} validation failed: "
                    f"{_truncate(str(first_error.validation_error), 2000)}"
                ) from first_error
            return self._repair(
                schema=schema,
                invalid_content=first_error.content,
                validation_error=first_error.validation_error,
                parent_task_type=task_type,
            )

    def _repair(
        self,
        *,
        schema: type[T],
        invalid_content: str,
        validation_error: Exception,
        parent_task_type: str,
    ) -> T:
        profile_name = self.settings.repair_profile
        if profile_name not in self.settings.profiles:
            raise StructuredLLMError(f"Unknown repair model profile `{profile_name}`")
        profile = self.settings.profiles[profile_name]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a JSON formatter. Return one JSON object matching the supplied "
                    "response_format schema. Preserve recoverable content, use conservative empty "
                    "defaults, and invent no scientific facts. Return no Markdown or commentary."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "schema": schema.__name__,
                        "validation_error": _truncate(str(validation_error), 3000),
                        "invalid_output": _truncate(invalid_content, 6000),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        try:
            response = self._call(
                task_type=f"{parent_task_type}_repair",
                profile_name=profile_name,
                profile=profile,
                messages=messages,
                schema=schema,
                temperature=0.0,
            )
            result = response.pop("_validated_model")
            return schema.model_validate(result)
        except _ResponseValidationError as exc:
            raise StructuredLLMError(
                f"{schema.__name__} remained invalid after one repair: "
                f"{_truncate(str(exc.validation_error), 2000)}"
            ) from exc

    def _call(
        self,
        *,
        task_type: str,
        profile_name: str,
        profile: ModelProfile,
        messages: list[dict[str, Any]],
        schema: type[BaseModel] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        self._consume_call_budget()
        input_chars = len(json.dumps(messages, ensure_ascii=False, default=str))
        if input_chars > self.settings.max_input_chars:
            raise InputBudgetExceeded(
                f"model input is {input_chars} characters; budget is "
                f"{self.settings.max_input_chars}. Compact the deterministic context first."
            )
        output_tokens = min(
            max_tokens or profile.max_tokens,
            self.settings.max_output_tokens,
        )
        body: dict[str, Any] = dict(profile.extra_body)
        # Configuration may tune sampling/chat-template kwargs, but it cannot smuggle in tools
        # or override hard request budgets.
        for forbidden in ["model", "messages", "max_tokens", "tools", "tool_choice", "response_format"]:
            body.pop(forbidden, None)
        body.update(
            {
                "model": profile.model,
                "messages": messages,
                "temperature": profile.temperature if temperature is None else temperature,
                "max_tokens": output_tokens,
            }
        )
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": _safe_schema_name(schema.__name__),
                    "strict": True,
                    "schema": schema.model_json_schema(),
                },
            }

        started = time.perf_counter()
        response_payload: dict[str, Any] | None = None
        failure = ""
        valid = False
        try:
            response_payload = self._post(profile, body)
            if schema is not None:
                content = _assistant_content(response_payload)
                try:
                    response_payload["_validated_model"] = schema.model_validate(
                        _extract_json(content)
                    )
                except (ValidationError, json.JSONDecodeError, TypeError) as exc:
                    failure = (
                        f"structured_output_invalid: {type(exc).__name__}: "
                        f"{_truncate(str(exc), 2500)}"
                    )
                    raise _ResponseValidationError(content, exc) from exc
            valid = True
            return response_payload
        except Exception as exc:
            if not failure:
                failure = f"{type(exc).__name__}: {_truncate(str(exc), 3000)}"
            raise
        finally:
            usage = (response_payload or {}).get("usage", {})
            if self.store is not None:
                self.store.append_model_call(
                    ModelCallRecord(
                        step_id=self._step_id,
                        task_type=task_type,
                        profile_name=profile_name,
                        model=profile.model,
                        input_chars=input_chars,
                        latency_seconds=round(time.perf_counter() - started, 4),
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                        total_tokens=usage.get("total_tokens"),
                        structured_schema=schema.__name__ if schema else None,
                        valid=valid,
                        execution_mode="real",
                        failure=failure,
                    )
                )

    def _consume_call_budget(self) -> None:
        if self._remaining_calls is None:
            # Standalone subsystem CLI calls still get a one-call operation budget. A structured
            # repair may consume a second call, hence two is the safe standalone cap.
            return
        if self._remaining_calls <= 0:
            raise ModelBudgetExceeded(
                f"model-call budget exhausted for step `{self._step_id}`"
            )
        self._remaining_calls -= 1

    def _post(self, profile: ModelProfile, body: dict[str, Any]) -> dict[str, Any]:
        headers = {"Authorization": f"Bearer {profile.api_key}"}
        url = profile.base_url.rstrip("/") + "/chat/completions"
        with httpx.Client(timeout=self.settings.timeout_seconds) as client:
            response = client.post(url, headers=headers, json=body)
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = _truncate(response.text.strip() or "<empty response>", 5000)
                raise httpx.HTTPStatusError(
                    f"{exc}\nOpenAI-compatible server response: {detail}",
                    request=exc.request,
                    response=exc.response,
                ) from exc
            payload = response.json()
            if not isinstance(payload, dict):
                raise StructuredLLMError("provider returned a non-object response")
            return payload


def _assistant_content(response: dict[str, Any]) -> str:
    try:
        value = response["choices"][0]["message"].get("content")
    except (KeyError, IndexError, TypeError) as exc:
        raise StructuredLLMError("provider response has no assistant message") from exc
    return _strip_reasoning_blocks(str(value or "")).strip()


def _extract_json(content: str) -> Any:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines.pop()
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as original:
        decoder = json.JSONDecoder()
        for start, character in enumerate(text):
            if character not in "[{":
                continue
            try:
                value, _ = decoder.raw_decode(text[start:])
                return value
            except json.JSONDecodeError:
                continue
        raise original


def _strip_reasoning_blocks(content: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
    return re.sub(r"<reasoning>.*?</reasoning>", "", text, flags=re.IGNORECASE | re.DOTALL)


def _safe_schema_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "Output"
    return ("Schema_" + safe if safe[0].isdigit() else safe)[:64]


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"
