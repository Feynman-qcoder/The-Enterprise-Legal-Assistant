"""Stable internal contract shared by Legacy and V2 legal retrieval paths."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias


RetrievalContractVersion: TypeAlias = str
ChunkReference: TypeAlias = int | str
ParentReference: TypeAlias = int | str

LEGACY_RETRIEVAL_CONTRACT = "legacy_v1"
V2_RETRIEVAL_CONTRACT = "v2"
SUPPORTED_RETRIEVAL_CONTRACTS = frozenset(
    {LEGACY_RETRIEVAL_CONTRACT, V2_RETRIEVAL_CONTRACT},
)


class RetrievalContractError(RuntimeError):
    """Raised when a retrieval record violates its frozen data contract."""


@dataclass(frozen=True, slots=True)
class LegalRetrievalHit:
    """Contract-neutral child hit consumed by BM25, RRF and parent recall."""

    chunk_id: ChunkReference
    content: str
    parent_reference: ParentReference
    metadata: dict[str, Any]
    similarity: float

    @property
    def source_file(self) -> str:
        value = self.metadata.get("source_file")
        return value if isinstance(value, str) else ""
