from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self):
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

    @property
    def dim(self) -> int:
        return self.encoder.get_sentence_embedding_dimension()

    def embed(self, text: str):
        return self.encoder.encode(text).tolist()
