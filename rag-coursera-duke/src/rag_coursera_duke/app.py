from .protocols import EmbedderProtocol, StoreProtocol
from .embedder import Embedder
from .documents import load_documents
from .vector_db import Store

def run() -> None:
    embedder: EmbedderProtocol = Embedder()
    store: StoreProtocol = Store(embedder=embedder)

    # documents = load_documents()
    documents = load_documents(count=700)

    store.upload(documents, 'notes')

    print(store.search("A wine from Mendoza Argentina"))

    # q = input("You: ").strip()
    # print(q)
    # print("Hello from rag-coursera-duke!")
