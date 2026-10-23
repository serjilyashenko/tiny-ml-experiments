from pathlib import Path
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"


class Embedder:
    def __init__(self):
        if not (MODEL_PATH / "modules.json").is_file():
            raise FileNotFoundError(
                f"Model files are missing in {MODEL_PATH}. "
                "Run 'uv run prepare-models'."
            )
        self.embedder = SentenceTransformer(str(MODEL_PATH))

    @property
    def dimension(self) -> int:
        return self.embedder.get_sentence_embedding_dimension()

    def embed(self, text: str):
        return self.embedder.encode(text).tolist()
