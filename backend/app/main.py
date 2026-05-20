from fastapi import FastAPI
from pathlib import Path
import numpy as np
import pandas as pd

app = FastAPI()

MODEL = None
MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "model_svd.npz"
MOVIES_CSV = Path(__file__).resolve().parents[1] / "data" / "ml-latest-small" / "movies.csv"


def load_model():
    global MODEL
    if MODEL_PATH.exists():
        npz = np.load(MODEL_PATH, allow_pickle=True)
        MODEL = {
            'user_factors': npz['user_factors'],
            'sigma': npz['sigma'],
            'item_factors': npz['item_factors'],
            'user_ids': npz['user_ids'].tolist(),
            'item_ids': npz['item_ids'].tolist(),
            'user_means': npz['user_means']
        }
        # load movie titles if available
        if MOVIES_CSV.exists():
            movies = pd.read_csv(MOVIES_CSV)
            MODEL['movie_map'] = {int(r.movieId): r.title for _, r in movies.iterrows()}
        else:
            MODEL['movie_map'] = {}


@app.on_event("startup")
def startup_event():
    load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL is not None}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, n: int = 10):
    if MODEL is None:
        return {"error": "model not trained"}

    try:
        idx = MODEL['user_ids'].index(user_id)
    except ValueError:
        return {"user_id": user_id, "recommendations": []}

    U = MODEL['user_factors']
    S = MODEL['sigma']
    Vt = MODEL['item_factors']
    user_mean = MODEL['user_means'][idx]

    # reconstruct approximate scores for the user
    user_vec = U[idx, :]
    scores = (user_vec * S) @ Vt
    # map item indices to movie ids and titles
    item_ids = MODEL['item_ids']

    # For simplicity, do not filter seen items here
    top_idx = np.argsort(-scores)[:n]
    recs = []
    for i in top_idx:
        mid = int(item_ids[i])
        recs.append({"movieId": mid, "title": MODEL['movie_map'].get(mid, ""), "score": float(scores[i] + user_mean)})

    return {"user_id": user_id, "recommendations": recs}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
