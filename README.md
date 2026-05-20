# Movie Recommender

Prototype de moteur de recommandation de films.

Structure:

- `backend`: API FastAPI + pipeline SVD
- `data`: datasets, prétraitement et modèle sauvegardé
- `frontend`: interface minimale React
- `notebooks`: exploration et expériences
- `tests`: tests unitaires

## Prérequis

- Python 3.10+
- Node.js 18+ / npm

## Installation backend

```bash
cd /root/code/movie-recommender
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Prétraitement et entraînement

Si MovieLens est disponible en local, le script utilisera `data/ml-latest-small`.
Sinon, un petit jeu de données de démonstration sera utilisé pour afficher un pipeline fonctionnel.

```bash
python data/preprocess.py
python backend/train_svd.py
```

## Lancer l'API

```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Interface React

```bash
cd frontend
npm install
npm run dev
```

## Tests

```bash
pytest
```
