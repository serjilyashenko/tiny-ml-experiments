from .documents import load_documents
from .embedder import Embedder

def run() -> None:
    embedder = Embedder()

    # documents = load_documents()
    documents = load_documents(count=10)

    print(documents)

    vector = embedder.embed("A fruity red wine")
    print(vector)

    # q = input("You: ").strip()
    # print(q)
    # print("Hello from rag-coursera-duke!")
