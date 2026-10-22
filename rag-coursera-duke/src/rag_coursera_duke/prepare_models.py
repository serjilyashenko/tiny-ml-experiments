from huggingface_hub import snapshot_download

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = "models/all-MiniLM-L6-v2"


def main():
    print(f">> Preparing model: {MODEL_ID}")

    path = snapshot_download(
        repo_id=MODEL_ID,
        local_dir=MODEL_DIR,
    )

    print(f">> Model is ready: {path}")


if __name__ == "__main__":
    main()
