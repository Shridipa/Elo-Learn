# Elo Learn

A research-oriented adaptive learning platform built with FastAPI and Streamlit.

Elo Learn combines student interaction modeling, concept mastery tracking, knowledge graph reasoning, explainable recommendations, and spaced-repetition scheduling into a coherent research prototype.

## What it does

- Estimates per-concept mastery using Bayesian Knowledge Tracing (BKT).
- Builds student and topic embeddings for similarity and evidence-based recommendation.
- Uses a knowledge graph to reason about prerequisites, readiness, and remediation.
- Generates explainable learning recommendations with peer evidence and mastery signals.
- Schedules reviews with SM2 spacing and forecasts retention.
- Surfaces instructor-facing cohort analytics and at-risk learners.

## Key features

- Knowledge Tracing: student mastery over topics from interaction history.
- Explainable Recommendations: hybrid recommender with explicit reasoning.
- Knowledge Graph Reasoning: prerequisite chains and remediation plans.
- Spaced Repetition: SM2 review scheduling for retained learning.
- Instructor Analytics: cohort summaries and at-risk detection.
- Benchmark Lab: offline model comparison with precision, recall, NDCG, and MRR.

## Architecture

The project is organized as follows:

- `backend/`: FastAPI application, recommendation endpoints, knowledge tracing, instructor analytics, and spaced repetition.
- `frontend/`: Streamlit dashboard for student and instructor views.
- `recommendation_engine/`: baseline and explainable recommendation models.
- `knowledge_graph/`: concept graph reasoning utilities.
- `ml_models/`: student embedding generation and feature engineering.
- `datasets/`: data artifacts and interaction records.
- `tests/`: unit and integration tests.

## Installation

Requirements: Python 3.10+.

Windows:

```powershell
python -m venv venv
& .\\venv\\Scripts\\Activate.ps1
python -m pip install -r requirements.txt
```

macOS / Linux:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running locally

Start the backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the frontend:

```bash
python -m streamlit run frontend/dashboard.py --server.port 8500
```

Open `http://localhost:8500` to view the dashboard.

## Selected API endpoints

- `GET /students/{student_id}/mastery`
- `GET /recommendations/{student_id}?model=hybrid&top_k=5`
- `POST /recommend/explain_v2`
- `GET /kg/path?topic={topic}`
- `GET /instructor/cohort_overview?weak_threshold=0.65`
- `GET /instructor/at_risk?mastery_threshold=0.6&max_results=50`
- `GET /reviews/due/{student_id}`
- `GET /recommend/benchmark?top_k=5&evaluation=full`

## Testing

Run the test suite with:

```bash
pytest -q
```
