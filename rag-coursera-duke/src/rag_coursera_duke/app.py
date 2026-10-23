from .documents import load_documents
from .embedder import Embedder
from .vector_db import Store

def run() -> None:
    embedder = Embedder()
    store = Store(embedder=embedder)

    # documents = load_documents()
    documents = load_documents(count=10)

    print(documents)

    vector = embedder.embed("A fruity red wine")
    print(vector)

    print(store.getTest())


    # q = input("You: ").strip()
    # print(q)
    # print("Hello from rag-coursera-duke!")
