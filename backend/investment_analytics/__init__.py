"""Regulatory-aware investment analytics core."""

from .compiler import compile_insight, compile_insights
from .errors import PolicyError

__all__ = ["PolicyError", "compile_insight", "compile_insights"]

