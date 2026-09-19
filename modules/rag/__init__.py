"""RAG package exports without eagerly importing the complete online pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modules.rag.pipeline import RagPipeline

__all__ = ["RagPipeline"]


def __getattr__(name: str) -> Any:
    if name == "RagPipeline":
        from modules.rag.pipeline import RagPipeline

        return RagPipeline
    raise AttributeError(name)
