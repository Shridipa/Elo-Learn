# Contributing to Elo Learn

Thank you for your interest in contributing to Elo Learn. This project is designed to be research-friendly, clean, and easy to extend.

## Contribution guidelines

- Fork the repository and create a feature branch from `main`.
- Keep changes small and focused.
- Write clear commit messages and issue descriptions.
- Follow PEP 8 formatting and Python best practices.
- Include tests for any bug fixes or new functionality.

## Running locally

```bash
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
```

## Code reviews

- Provide a short summary of the change.
- List any files or tests that should be reviewed carefully.
- Document any limitations or follow-up work.

## Reporting issues

- Use the GitHub issue tracker.
- Provide reproducible steps and expected vs actual behavior.
- Tag the issue clearly (bug, enhancement, docs, question).

## Structure of this repository

- `backend/`: FastAPI server and core APIs
- `frontend/`: Streamlit dashboard
- `datasets/`: sample data and artifacts
- `docs/`: project documentation
- `tests/`: automated tests
- `deployment/`: deployment manifests and guides
