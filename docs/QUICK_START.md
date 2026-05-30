## Quick Start Guide

Get Elo Learn up and running in 5 minutes.

### Prerequisites
- Python 3.9+
- PostgreSQL 12+ (or Docker)
- 4GB RAM recommended

### Option 1: Local Development (Docker Compose)

```bash
# Clone repository
git clone <repo-url>
cd elo-learn

# Start all services
docker-compose up -d

# Wait for services to start (~30 seconds)
docker-compose logs -f

# API available at: http://localhost:8000
# Dashboard available at: http://localhost:8501
```

### Option 2: Local Development (Manual)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Setup database (ensure PostgreSQL is running)
python backend/database/init_db.py

# Generate synthetic dataset
python datasets/generate_synthetic_interactions.py

# Train models
python training/train_main.py

# Start backend
uvicorn backend.main:app --reload

# In another terminal: Start dashboard
streamlit run frontend/dashboard.py
```

### First Steps

1. **Generate Dataset** (~2 minutes)
   ```bash
   python datasets/generate_synthetic_interactions.py
   ```
   Creates synthetic interactions for 500 students across 17 concepts.

2. **Engineer Features** (automatic in training script)
   - Extracts student embeddings
   - Creates topic embeddings
   - Computes interaction features

3. **Train Recommendation Models** (~5 minutes)
   ```bash
   python training/train_main.py
   ```
   Trains CF, content-based, and hybrid recommenders.

4. **Evaluate Models**
   Automatic evaluation included in training script.
   Check results in `results/recommendation_results.json`

5. **Access Dashboard**
   Open http://localhost:8501

### File Structure After Setup

```
elo-learn/
├── datasets/
│   ├── student_interactions.csv      # Synthetic data (500 students)
│   ├── concept_graph.json            # Concept dependencies
│   └── embeddings.json               # Generated embeddings
├── results/
│   ├── recommendation_results.json   # Model evaluation metrics
│   └── rl_training.json              # RL agent performance
├── checkpoints/                      # Saved model weights
└── logs/                             # Application logs
```

### API Examples

**Get Recommendations**
```bash
curl http://localhost:8000/recommendations/1
```

**Record Student Interaction**
```bash
curl -X POST http://localhost:8000/interactions \
  -H "Content-Type: application/json" \
  -d '{
    "student_id": 1,
    "topic_id": 1,
    "is_correct": true,
    "score": 0.95,
    "time_spent": 300,
    "difficulty_presented": 0.5,
    "session_id": "session_1"
  }'
```

**Get Student Analytics**
```bash
curl http://localhost:8000/analytics/student/1
```

### Troubleshooting

**PostgreSQL connection error:**
```bash
# Start PostgreSQL
docker-compose up -d db

# Or connect to existing PostgreSQL
# Update DATABASE_URL in .env
```

**Port already in use:**
```bash
# Change ports in .env and docker-compose.yml
API_PORT=8001
# Streamlit automatically finds next available port
```

**Out of memory:**
```bash
# Reduce dataset size
python datasets/generate_synthetic_interactions.py --n_students 100

# Reduce model complexity
python training/train_main.py --embedding_dim 32
```

### Next Steps

1. **Explore Data**: Check `notebooks/01_eda.ipynb`
2. **Understand Models**: Read `docs/architecture.md`
3. **Run Full Pipeline**: Execute `training/train_main.py`
4. **Deploy**: Follow `docs/deployment.md`
5. **Extend**: Add custom recommendation models in `recommendation_engine/`

### Configuration

Key settings in `.env`:
```bash
# Model
EMBEDDING_DIM=64
REC_TOP_K=5

# RL
RL_LEARNING_RATE=0.001
RL_GAMMA=0.99

# System
LOG_LEVEL=INFO
CACHE_TTL=3600
```

### Performance Targets

On default settings (500 students):
- Dataset generation: ~5 seconds
- Feature engineering: ~3 seconds
- Model training: ~10 seconds
- API startup: ~5 seconds
- Recommendation latency: <100ms

### Support & Documentation

- 📖 [Architecture Guide](docs/architecture.md)
- 🔌 [API Reference](docs/api.md)
- 🔬 [Research Findings](research/findings.md)
- 📊 [Development Notes](research/development_notes.md)

---

**Happy Learning! 🚀**
