from pathlib import Path
import pandas as pd


def load_documents(count: int = None) -> list[dict]:
    csv_path = Path(__file__).parent / "top_rated_wines.csv"
    df = pd.read_csv(csv_path)

    if count is not None:
        df = df.sample(n=count)

    data = df.to_dict(orient="records")
    print(f">> Loaded {len(data)} documents")

    return data
