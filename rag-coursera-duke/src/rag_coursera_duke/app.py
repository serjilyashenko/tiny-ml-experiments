from .documents import load_documents
from .embedder import Embedder

def run() -> None:
    # documents = load_documents()
    documents = load_documents(count=10)

    print(documents)

    embedder = Embedder()
    vector = embedder.embed("A fruity red wine")
    print(vector)

    # q = input("You: ").strip()
    # print(q)
    # print("Hello from rag-coursera-duke!")
