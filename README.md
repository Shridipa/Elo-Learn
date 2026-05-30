# Elo Learn

Elo Learn is an adaptive learning prototype built with FastAPI and Streamlit. The code is meant to show how student mastery, concept reasoning, recommendation logic and review scheduling can work together in one package.

I built this project to explore a practical student modeling workflow: track performance, surface the right next topics, explain recommendations, and schedule review tasks in a compact dashboard.

## What it does

- Tracks student progress across topics using a Bayesian Knowledge Tracing implementation.
- Builds student and concept embeddings to compare mastery patterns and peer behavior.
- Uses a concept graph to identify prerequisites, readiness and remediation steps.
- Produces explainable recommendations with topic evidence and confidence signals.
- Schedules review sessions using an SM2-inspired spaced repetition routine.
- Shows cohort metrics and identifies students who need instructor attention.

## Why it matters

This project is not just a demo. It is a working research prototype for adaptive learning that combines multiple educational signals in one place. It is designed to be easy to run locally and to demonstrate how analytics can support both individual learners and instructors.

## Project structure

- `backend/`: FastAPI application, endpoints for recommendations, mastery, coach analytics, and review scheduling.
- `frontend/`: Streamlit dashboard with student and instructor views.
- `recommendation_engine/`: baseline and explainable recommendation models.
- `knowledge_graph/`: concept dependency reasoning and readiness calculations.
- `ml_models/`: embedding generation and feature engineering for students and topics.
- `datasets/`: static artifacts and example student interactions.
- `tests/`: automated checks for algorithms and backend endpoints.

## Setup

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

## Run locally

Start the backend:

```bash
uvicorn backend.main:app --reload --port 8000
```

Start the frontend:

```bash
python -m streamlit run frontend/dashboard.py --server.port 8500
```

Open http://localhost:8500 in your browser to use the dashboard.

## Important endpoints

- `GET /students/{student_id}/mastery`: student mastery map
- `GET /recommendations/{student_id}?model=hybrid&top_k=5`: explainable recommendations
- `POST /recommend/explain_v2`: recommendation explanations
- `GET /kg/path?topic={topic}`: concept prerequisite path
- `GET /instructor/cohort_overview?weak_threshold=0.65`: cohort analytics summary
- `GET /instructor/at_risk?mastery_threshold=0.6&max_results=50`: at-risk student list
- `GET /reviews/due/{student_id}`: review topics due now
- `GET /recommend/benchmark?top_k=5&evaluation=full`: recommendation benchmark comparison

## Tests

Run the full suite with:

```bash
pytest -q
```
