from typing import Protocol, Any
from dataclasses import dataclass


class EmbedderProtocol(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


@dataclass(frozen=True)
class StoreSearchResult:
    id: int | str
    score: float
    payload: dict[str, Any] | None


class StoreProtocol(Protocol):
    def upload(self, documents: list[dict[str, Any]], vector_filed: str) -> None: ...

    def search(self, query: str, limit: int = 3) -> list[StoreSearchResult]: ...