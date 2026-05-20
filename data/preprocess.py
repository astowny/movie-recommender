import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "ml-latest-small"

if __name__ == "__main__":
    ratings_path = RAW_DIR / "ratings.csv"
    df = pd.read_csv(ratings_path)
    # Filtrer utilisateurs avec au moins 5 notes
    counts = df['userId'].value_counts()
    keep_users = counts[counts >= 5].index
    df = df[df['userId'].isin(keep_users)]
    out = DATA_DIR / "ratings_processed.csv"
    df.to_csv(out, index=False)
    print("Wrote", out)
