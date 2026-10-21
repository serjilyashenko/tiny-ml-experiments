from .documents import load_documents

def run() -> None:
    # documents = load_documents()
    documents = load_documents(count=10)

    print(documents)

    # q = input("You: ").strip()
    # print(q)
    # print("Hello from rag-coursera-duke!")
