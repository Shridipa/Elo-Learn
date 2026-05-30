# Architecture

## Problem
Traditional learning platforms provide static content. Students have different mastery levels, prior knowledge gaps, and learning speeds. Static content delivery makes it hard for instructors to see who is falling behind, and it does not adapt to the student’s current state.

## Solution
Elo Learn combines:

- Recommendation Systems
- Bayesian Knowledge Tracing
- Knowledge Graph Reasoning
- Spaced Repetition Scheduling

This combination enables an adaptive learning platform that personalizes topic sequencing, recommends remediation, and surfaces instructor analytics.

## Architecture

### High-level architecture

```text
Student Interactions
          │
          ▼
Student Embeddings
          │
          ▼
Knowledge Tracing
          │
          ▼
Knowledge Graph
          │
          ▼
Recommendations
          │
          ▼
Spaced Repetition
```

### Core components

- **Frontend**: `frontend/dashboard.py` is a Streamlit interface for students and instructors.
- **Backend**: `backend/main.py` exposes FastAPI endpoints for recommendations, knowledge tracing, knowledge graph reasoning, spaced repetition, student embeddings, and instructor analytics.
- **Recommendation engine**: `recommendation_engine/` contains multiple baselines, hybrid ranking, and explainability methods.
- **Knowledge Graph**: `knowledge_graph/graph.py` models concept relationships, prerequisites, and readiness reasoning.
- **ML Models**: `ml_models/` computes student embeddings, similarity search, and feature engineering.
- **Datasets**: `datasets/` stores runtime data, embeddings, and evaluation artifacts.

## Data flow

1. A student interaction is recorded and stored in datasets.
2. Knowledge tracing updates the mastery estimate for each concept.
3. Student embeddings are computed for similarity-based reasoning.
4. The recommendation engine ranks next topics using mastery, readiness, and peer evidence.
5. Knowledge graph reasoning refines candidate topics based on prerequisites.
6. Spaced repetition schedules review topics for long-term retention.
7. The dashboard visualizes recommendations, mastery, KG reasoning, and instructor alerts.

## Design principles

- **Modular by design**: Each ML component is independently benchmarked.
- **Explainability**: Recommendations include reasons and evidence.
- **Research-ready**: Supports temporal evaluation, benchmark exports, and cohort analytics.
- **Production-focused**: Includes Docker deployment, frontend and backend separation, and documentation for deployment.

## System diagram

```mermaid
flowchart TB
    A[Student Interaction] --> B[FastAPI Backend]
    B --> C[Knowledge Tracing (BKT)]
    B --> D[Student Embeddings]
    D --> E[Similarity Search]
    B --> F[Recommendation Engine]
    F --> G[Explainability Layer]
    G --> H[Knowledge Graph Reasoning]
    F --> I[Spaced Repetition Scheduler]
    B --> J[Streamlit Dashboard]
```

## Deployment

- Backend service: `uvicorn backend.main:app --host 0.0.0.0 --port 8000`
- Frontend service: `python -m streamlit run frontend/dashboard.py --server.port 8500`
- Docker orchestration: `deployment/docker-compose.yml`

## Notes

This document is intended to support portfolio-level storytelling and demonstrate a clear system design for a deployable adaptive learning platform.