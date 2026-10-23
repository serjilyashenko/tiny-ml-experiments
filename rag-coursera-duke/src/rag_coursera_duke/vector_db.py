from typing import Any
from .protocols import EmbedderProtocol, StoreSearchResult
from qdrant_client import models, QdrantClient


class Store:
    def __init__(self, embedder: EmbedderProtocol) -> None:
        self.embedder = embedder
        self.client = QdrantClient(":memory:")
        self.client.recreate_collection(
            collection_name="top_wines",
            vectors_config=models.VectorParams(
                size=embedder.dimension,
                distance=models.Distance.COSINE
            )
        )

    def upload(self, documents: list[dict[str, Any]], vector_filed: str) -> None:
        points = [
            models.PointStruct(
                id=idx,
                vector=self.embedder.embed(doc[vector_filed]),
                payload=doc
            ) for idx, doc in enumerate(documents)  # data is the variable holding all the wines
        ]
        self.client.upload_points(
            collection_name="top_wines",
            points=points,
        )
        print(f">> Uploaded {len(points)} points")

    def search(self, query: str, limit: int = 3) -> list[StoreSearchResult]:
        hits = self.client.search(
            collection_name="top_wines",
            query_vector=self.embedder.embed(query),
            limit=limit
        )
        return [
            StoreSearchResult(
                id=hit.id,
                score=hit.score,
                payload=hit.payload,
            )
            for hit in hits
        ]

    def close(self) -> None:
        self.client.close()
