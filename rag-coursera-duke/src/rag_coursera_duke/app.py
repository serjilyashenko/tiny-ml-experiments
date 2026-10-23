from .protocols import EmbedderProtocol, StoreProtocol
from .embedder import Embedder
from .documents import load_documents
from .vector_db import Store

def run() -> None:
    embedder: EmbedderProtocol = Embedder()
    store: StoreProtocol = Store(embedder=embedder)

    try:
        # documents = load_documents()
        documents = load_documents(count=700)

        user_prompt = "A bold red wine with blackberry, dark chocolate, vanilla and a long finish"

        store.upload(documents=documents, vector_filed="notes")
        search_results = store.search(user_prompt)
        print(search_results)
    finally:
        store.close()

    # q = input("You: ").strip()
    # print(q)
    # print("Hello from rag-coursera-duke!")
