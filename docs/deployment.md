# Deployment

This document describes how to deploy Elo Learn using Docker and Docker Compose.

## Build and run with Docker

The recommended deployment path is through `deployment/docker-compose.yml`.

### Start the stack

```bash
cd deployment
docker-compose up --build
```

This brings up:

- `backend` on `http://localhost:8000`
- `frontend` on `http://localhost:8500`

## deployment/Dockerfile

The backend image is defined in `deployment/Dockerfile` and installs the Python dependencies from `requirements.txt`.

## deployment/docker-compose.yml

The Compose file launches:

- `backend`: FastAPI service using the same project sources
- `frontend`: Streamlit dashboard in a Python 3.11 container

## nginx

`deployment/nginx.conf` is available as a reverse proxy template if you want to front the backend and frontend with a single HTTP entry point.

## Production checklist

- Confirm `requirements.txt` is up to date.
- Keep dataset files in `datasets/` for runtime.
- Expose backend port `8000` and frontend port `8500`.
- Use `docker-compose down` to stop the stack.

## Notes

For a polished production deployment, pair `deployment/docker-compose.yml` with a reverse proxy, TLS certificate management, and persistent storage for dataset and result files.