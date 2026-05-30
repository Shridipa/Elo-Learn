# Deployment Guide for Elo Learn

This guide explains how to deploy Elo Learn locally and on hosted platforms.

## Local Docker deployment

1. Build and start services:

```bash
docker-compose -f deployment/docker-compose.yml up --build
```

2. Access the backend at `http://localhost:8000`.
3. Access the frontend at `http://localhost:8500`.

## Render deployment

1. Add `deployment/render.yaml` to your repository.
2. Connect the repo in Render.
3. Use two services:
   - Backend service: `uvicorn backend.main:app --host 0.0.0.0 --port 10000`
   - Frontend service: `python -m streamlit run frontend/dashboard.py --server.port 10001 --server.headless true`

## Railway deployment

1. Add `deployment/railway.json` to the repository.
2. Connect the repository in Railway.
3. Configure the backend and frontend services as specified.

## Production considerations

- Use environment variables to configure backend host and dataset paths.
- Secure the dashboard behind a reverse proxy if exposing publicly.
- Use a proper storage strategy for dataset artifacts and model checkpoints.
- Add TLS via the hosting provider or reverse proxy.

## Notes

- The `backend` service exposes FastAPI on port `8000`.
- The `frontend` service uses Streamlit on port `8500`.
- `deployment/nginx.conf` can be used when deploying behind NGINX or a reverse proxy.
