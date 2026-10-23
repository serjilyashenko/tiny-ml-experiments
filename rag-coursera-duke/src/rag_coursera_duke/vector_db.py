from qdrant_client import models, QdrantClient


class Store:
    def __init__(self, embedder):
        self.qdrant = QdrantClient(":memory:")
        self.qdrant.recreate_collection(
            collection_name="top_wines",
            vectors_config=models.VectorParams(
                size=embedder.dimension,
                distance=models.Distance.COSINE
            )
        )

    def getTest(self):
        return self.qdrant
