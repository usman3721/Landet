# Language Detector

Detects which of 133 languages a piece of text is written in. A scikit-learn model
(TF-IDF + Naive Bayes) reaches about 98% accuracy, and a small FastAPI app serves it.

This project is deliberately simple. The aim is to understand every step of building
and serving a model, without extra tools hiding what is going on.

## Setup

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync
```

This creates `.venv` with everything the project needs. Prefix commands with `uv run`
so they use that environment.

## Train the model

The dataset is too big for Git, so put it here first:

```
data/raw/new_merged_languages.csv
```

Then run the four steps in order, from the project root:

```bash
uv run python -m src.data_model_pipeline.data_ingestion
uv run python -m src.data_model_pipeline.data_processing
uv run python -m src.data_model_pipeline.model_training
uv run python -m src.data_model_pipeline.model_evaluation
```

| Step | What it does | Output |
| --- | --- | --- |
| `data_ingestion` | Drops unused columns, splits the data 80/10/10 | `data/preprocessed/` |
| `data_processing` | Learns the vocabulary (TF-IDF) and the label names from the training data | `models/vectorizer.joblib`, `models/label_encoder.joblib` |
| `model_training` | Trains Naive Bayes on the training data | `models/language_detector.joblib` |
| `model_evaluation` | Scores the model on unseen data | `reports/report.json`, `reports/confusion_matrix.png` |

`notebooks/sample.ipynb` walks through the same steps one cell at a time. Open it in
VS Code and pick the `.venv` kernel. Run the pipeline above first, because the notebook
loads the saved model.

## Run the API

```bash
uv run uvicorn src.app.main:app --reload
```

Interactive docs are at http://127.0.0.1:8000/docs. Or use the API directly:

```bash
# 1. Log in to get a token
curl -X POST http://127.0.0.1:8000/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# 2. Predict, using that token
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -H "api-key: demo-key" \
  -H "token: <paste the access_token here>" \
  -d '{"text": "Bonjour tout le monde"}'
```

The API works without Redis. If Redis is running, repeated texts are answered from the cache.

The login and the default secrets in `config.py` are demo values. Change them before
putting this anywhere public.

## Run with Docker

Train the model first. The container reads it from your local `models/` folder.

```bash
docker compose up --build
```

This starts the API and Redis.

## Test

```bash
uv run pytest
uv run ruff check .
uv run ruff format .
```

The tests build a tiny dataset of their own, so they run without the real data, the
trained model or Redis. GitHub Actions runs the same checks on every pull request to `main`.

## Project structure

```
src/
├── data_model_pipeline/     # Training: raw data -> saved model
│   ├── data_ingestion.py
│   ├── data_processing.py   # clean_text() lives here, shared with the API
│   ├── model_training.py
│   └── model_evaluation.py
└── app/                     # Serving: saved model -> HTTP API
    ├── main.py              # Creates the app and connects everything
    ├── api/                 # Endpoints: /login, /predict
    ├── core/                # Settings, JWT and API key checks
    ├── services/            # Loads the model and predicts
    ├── cache/               # Redis cache
    └── middleware/          # Request logging
tests/                       # Mirrors the parts above
notebooks/sample.ipynb       # Exploration
data/, models/, reports/     # Generated locally, not in Git
```

## How it works

**Computers need numbers, not text.** TF-IDF turns each text into a list of numbers.
Here it counts short character patterns such as `the`, `sch` or `ção`. These work better
than whole words because every language has its own letter patterns, even for words
the model has never seen.

**The model learns which patterns belong to which language.** Naive Bayes works out how
likely each pattern is in each language, then picks the language that best explains the
text. It is fast, and it is a good baseline.

**Keep data apart to measure honestly.** The model trains on the train set. You compare
ideas on the validation set. You use the test set once, at the end. A model scored on
data it has already seen looks better than it really is.

**Fit on train, transform everywhere else.** The vectorizer learns its vocabulary from the
training data only. Letting it see validation data would leak information and inflate your score.

**Training and serving must match.** The API runs the same `clean_text()` that training
used. If they differed, the model would get input unlike anything it learned from, and
accuracy would drop without any error appearing.

## Things to try

Learn by changing one thing, predicting what will happen, then measuring it:

1. Switch `analyzer` to `"word"` in `data_processing.py`. Does accuracy drop? Why?
2. Reduce `max_features` from 50,000 to 5,000. How much accuracy do you lose, and how much smaller is the model?
3. Open `reports/report.json` and find the worst languages. Which ones get confused, and are they related?
4. Replace `MultinomialNB` with `LogisticRegression`. Compare accuracy, training time and model size.
5. Write Naive Bayes yourself in NumPy and compare it with scikit-learn's result.
