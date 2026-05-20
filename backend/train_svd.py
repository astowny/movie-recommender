"""Train a simple SVD-based recommender on the processed MovieLens ratings.

Creates `data/model_svd.npz` containing user_factors, sigma, item_factors,
and mappings between user ids and item indices.
"""
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED = DATA_DIR / "ratings_processed.csv"
MOVIES = DATA_DIR / "ml-latest-small" / "movies.csv"
MODEL_OUT = DATA_DIR / "model_svd.npz"


def load_ratings():
    if PROCESSED.exists():
        df = pd.read_csv(PROCESSED)
    else:
        raw = DATA_DIR / "ml-latest-small" / "ratings.csv"
        df = pd.read_csv(raw)
    return df


def train(k=50):
    df = load_ratings()
    # pivot to user x item matrix
    users = sorted(df['userId'].unique())
    items = sorted(df['movieId'].unique())
    user_to_idx = {u: i for i, u in enumerate(users)}
    item_to_idx = {m: i for i, m in enumerate(items)}

    R = np.zeros((len(users), len(items)), dtype=float)
    for _, row in df.iterrows():
        R[user_to_idx[row['userId']], item_to_idx[row['movieId']]] = row['rating']

    # subtract user means to center
    user_means = np.true_divide(R.sum(axis=1), (R != 0).sum(axis=1))
    user_means = np.nan_to_num(user_means)
    R_centered = R - user_means.reshape(-1, 1)

    # use numpy SVD (dense) — fine for ml-latest-small
    # compute full SVD then truncate
    U, s, Vt = np.linalg.svd(R_centered, full_matrices=False)
    k = min(k, U.shape[1])
    U_k = U[:, :k]
    s_k = s[:k]
    Vt_k = Vt[:k, :]

    # save model and mappings
    np.savez(MODEL_OUT,
             user_factors=U_k,
             sigma=s_k,
             item_factors=Vt_k,
             user_ids=np.array(users),
             item_ids=np.array(items),
             user_means=user_means)
    print("Saved model to", MODEL_OUT)


if __name__ == "__main__":
    train(k=50)
