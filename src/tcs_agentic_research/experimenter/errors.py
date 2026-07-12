"""Experimenter-specific errors."""

from __future__ import annotations


class ExperimenterError(RuntimeError):
    """Base class for experimenter failures."""

    fatal_tool_error = True


class ExperimenterConfigurationError(ExperimenterError):
    """Raised when the experimenter is requested without valid configuration."""


class ExperimenterRuntimeError(ExperimenterError):
    """Raised when Docker, pi, or artifact validation fails during an experiment."""
