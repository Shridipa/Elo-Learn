"""Main FastAPI Application for Elo Learn

Initializes the API server, sets up routes, middleware, and CORS configuration.
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from backend.config import settings
from backend.schemas import (
    HealthResponse,
    RecommendationResultsResponse,
    RLTrainingResultsResponse,
    ExplainableRecommendationResponse,
    RecommendationEvidenceResponse,
    RecommendationConfidenceResponse,
    RecommendationBenchmarkResponse,
    RecommendationReadinessResponse,
    DueReviewsResponse,
    ReviewCompleteResponse,
    ReviewCompleteRequest,
    ReviewScheduleResponse,
    RetentionForecastResponse,
    RetentionStatusResponse,
    ReviewStatisticsResponse,
)
from backend.knowledge_tracing.service import KnowledgeTracingService
from backend.spaced_repetition.scheduler import SpacedRepetitionScheduler
from knowledge_graph.graph import KnowledgeGraph
from ml_models.student_embeddings import StudentEmbeddingEngine
from recommendation_engine.explainable import ExplainableRecommender
from recommendation_engine import evaluator
from recommendation_engine.temporal_split import temporal_train_test_split
from sklearn.neighbors import NearestNeighbors
import numpy as np
import joblib
import time
from sklearn.exceptions import InconsistentVersionWarning

# Configure logging
logging.basicConfig(
    level=settings.log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("Starting Elo Learn API...")
    logger.info(f"Debug mode: {settings.debug}")
    logger.info(f"API running on {settings.api_host}:{settings.api_port}")
    
    # Initialize optional runtime resources
    app.state.db_connection = None
    app.state.model_checkpoints_loaded = False
    app.state.metrics = {
        "request_count": 0,
        "total_response_time": 0.0,
        "error_count": 0,
    }
    app.state.cache_initialized = False

    # Load knowledge graph into app state
    try:
        kg = KnowledgeGraph()
        kg_path = settings.data_path / "concept_graph.json"
        if kg_path.exists():
            kg.load_from_file(str(kg_path))
            app.state.kg = kg
            logger.info(f"Knowledge graph loaded with {kg.graph.number_of_nodes()} nodes")
        else:
            logger.warning(f"Concept graph not found at {kg_path}")
            # Fallback to a built-in sample knowledge graph for tests/environments
            try:
                from knowledge_graph.graph import create_sample_knowledge_graph
                app.state.kg = create_sample_knowledge_graph()
                logger.info('Initialized sample knowledge graph (fallback)')
            except Exception:
                app.state.kg = kg
    except Exception as exc:
        logger.error(f"Failed to load knowledge graph: {exc}")

    # Pre-warm the embedding cache if artifacts exist
    try:
        _ensure_embedding_cache(app)
    except Exception as exc:
        logger.warning(f"Failed to pre-warm embedding cache: {exc}")

    # Initialize knowledge tracing service
    try:
        interactions_file = settings.data_path / 'student_interactions.csv'
        if interactions_file.exists():
            kt_service = KnowledgeTracingService(interactions_file, concept_graph_path=settings.data_path / 'concept_graph.json')
            app.state.knowledge_tracing = kt_service
            logger.info('Knowledge tracing service initialized with %d students', len(kt_service.store.mastery_by_student))
        else:
            app.state.knowledge_tracing = None
            logger.warning('Student interactions file not found for knowledge tracing: %s', interactions_file)
    except Exception as exc:
        logger.error(f'Failed to initialize knowledge tracing service: {exc}')
        app.state.knowledge_tracing = None
    # Initialize explainable recommender and precompute topic embeddings
    try:
        explainer = ExplainableRecommender(neighbor_count=10)
        app.state.explainable_recommender = explainer
        # precompute topic embeddings if KG present
        if getattr(app.state, 'kg', None) is not None:
            try:
                app.state.topic_embeddings = _get_topic_embeddings(app.state.kg)
                logger.info('Precomputed topic embeddings for explainable recommender')
            except Exception as exc:
                logger.warning(f'Failed to precompute topic embeddings: {exc}')
        else:
            app.state.topic_embeddings = {}
    except Exception as exc:
        logger.warning(f'Failed to initialize ExplainableRecommender: {exc}')
    
    # Initialize spaced repetition scheduler
    try:
        sr_scheduler = SpacedRepetitionScheduler()
        app.state.sr_scheduler = sr_scheduler
        logger.info('Spaced repetition scheduler initialized')
    except Exception as exc:
        logger.warning(f'Failed to initialize spaced repetition scheduler: {exc}')
    
    yield
    
    # Shutdown
    logger.info("Shutting down Elo Learn API...")
    if getattr(app.state, "db_connection", None) is not None:
        try:
            app.state.db_connection.close()
            logger.info("Closed database connection")
        except Exception as exc:
            logger.warning(f"Failed to close database connection: {exc}")

    logger.info("Cleanup complete")

# ==================== FastAPI App ====================

app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan,
    debug=settings.debug
)

# ==================== CORS Configuration ====================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.debug else ["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Request Metrics Middleware ====================

@app.middleware("http")
async def request_metrics_middleware(request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = time.time() - start
    metrics = getattr(app.state, "metrics", None)
    if metrics is not None:
        metrics["request_count"] += 1
        metrics["total_response_time"] += elapsed
        if response.status_code >= 500:
            metrics["error_count"] += 1
    return response

# ==================== Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.utcnow().isoformat()},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        },
    )

# ==================== Health Check Endpoints ====================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    db_available = getattr(app.state, "db_connection", None) is not None
    cache_available = getattr(app.state, "embedding_cache", None) is not None
    kg = getattr(app.state, "kg", None)
    kg_available = bool(kg and getattr(kg, "graph", None) is not None and kg.graph.number_of_nodes() > 0)
    services = {
        "database": db_available,
        "cache": cache_available,
        "knowledge_graph": kg_available,
        "ml_models": cache_available and kg_available,
    }
    overall_status = "healthy" if all(services.values()) else "degraded"
    return HealthResponse(
        status=overall_status,
        version=settings.api_version,
        timestamp=datetime.utcnow(),
        services=services
    )

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "name": settings.api_title,
        "version": settings.api_version,
        "status": "running"
    }

@app.get("/metrics", tags=["Metrics"])
async def metrics():
    """System metrics endpoint"""
    metrics = getattr(app.state, "metrics", {"request_count": 0, "total_response_time": 0.0, "error_count": 0})
    total_requests = metrics.get("request_count", 0)
    average_response_time = (
        metrics["total_response_time"] / total_requests if total_requests > 0 else 0.0
    )
    error_rate = (
        metrics["error_count"] / total_requests if total_requests > 0 else 0.0
    )
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.api_version,
        "request_count": total_requests,
        "average_response_time": average_response_time,
        "error_rate": error_rate,
        "cache_loaded": getattr(app.state, "embedding_cache", None) is not None,
    }

# ==================== Version Info ====================

@app.get("/version", tags=["Info"])
async def version_info():
    """Get version and build information"""
    return {
        "version": settings.api_version,
        "api_title": settings.api_title,
        "debug": settings.debug,
        "timestamp": datetime.utcnow().isoformat()
    }

# ==================== Result Endpoints ====================

def _load_json_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Result file not found: {path}")
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        logger.error(f"Failed to load JSON from {path}: {exc}")
        raise HTTPException(status_code=500, detail="Failed to read result file")


def _ensure_embedding_cache(app: FastAPI, force_refresh: bool = False):
    """Ensure app.state.embedding_cache exists and is populated.

    Cache structure:
    {
        'embeddings': {id: list},
        'features': {id: {...}},
        'ids': [id,...],
        'X': np.array([...]),
        'nn': NearestNeighbors fitted,
        'timestamp': float
    }
    """
    if not force_refresh:
        cache = getattr(app.state, 'embedding_cache', None)
        if cache:
            age = time.time() - cache.get('timestamp', 0)
            if age < settings.cache_ttl:
                return cache
            logger.info('In-memory embedding cache stale (age %.0fs), refreshing', age)

    embeddings_file = settings.data_path / 'student_embeddings.json'
    if not embeddings_file.exists():
        app.state.embedding_cache = None
        return None

    try:
        raw = json.loads(embeddings_file.read_text())
        emb = raw.get('embeddings', {})
        features = raw.get('features', {})
        emb_int = {int(k): v for k, v in emb.items()}
        ids = sorted(list(emb_int.keys()))
        X = np.array([emb_int[i] for i in ids])
        # Check for persisted NN index
        nn = None
        cache_file = settings.model_checkpoint_path / 'nn_index.joblib'

        # If persisted index exists and is recent enough, load it
        if cache_file.exists() and not force_refresh:
            try:
                mtime = cache_file.stat().st_mtime
                age = time.time() - mtime
                if age < settings.cache_ttl:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('error', category=InconsistentVersionWarning)
                        nn = joblib.load(cache_file)
                    logger.info('Loaded persisted NN index from %s', cache_file)
                else:
                    logger.info('Persisted NN index is stale (age %.0fs), will rebuild', age)
            except InconsistentVersionWarning as exc:
                logger.warning('Persisted NN index version mismatch: %s. Rebuilding from scratch.', exc)
                nn = None
            except Exception as exc:
                logger.warning(f'Failed to load persisted NN index: {exc}')

        # If no persisted NN or it's stale, build from scratch and persist
        if nn is None and X.shape[0] >= 1:
            nn = NearestNeighbors(n_neighbors=min(20, X.shape[0]), algorithm='auto').fit(X)
            try:
                joblib.dump(nn, cache_file)
                logger.info('Persisted NN index to %s', cache_file)
            except Exception as exc:
                logger.warning(f'Failed to persist NN index: {exc}')

        cluster_labels = raw.get('cluster_labels', [])
        cluster_metrics = raw.get('cluster_metrics', {})
        experiment_metrics = raw.get('experiment_metrics', {})
        coords_pca = raw.get('coords_pca', {})
        coords_tsne = raw.get('coords_tsne', {})

        cache = {
            'embeddings': emb_int,
            'features': {int(k): v for k, v in features.items()} if isinstance(features, dict) else {},
            'ids': ids,
            'X': X,
            'nn': nn,
            'cluster_labels': cluster_labels,
            'cluster_metrics': cluster_metrics,
            'experiment_metrics': experiment_metrics,
            'coords_pca': coords_pca,
            'coords_tsne': coords_tsne,
            'timestamp': time.time()
        }
        app.state.embedding_cache = cache
        logger.info('Loaded embedding cache with %d students', len(ids))
        return cache
    except Exception as exc:
        logger.error(f'Failed to build embedding cache: {exc}')
        app.state.embedding_cache = None
        return None


def _get_embedding_cache(app: FastAPI):
    cache = _ensure_embedding_cache(app)
    if not cache:
        raise HTTPException(status_code=500, detail='Embedding cache unavailable')
    return cache


def _get_knowledge_tracing_service():
    service = getattr(app.state, 'knowledge_tracing', None)
    interactions_file = settings.data_path / 'student_interactions.csv'
    if service is None or getattr(service, 'interactions_path', None) != interactions_file:
        if not interactions_file.exists():
            raise HTTPException(status_code=503, detail='Knowledge tracing data unavailable')
        try:
            service = KnowledgeTracingService(interactions_file, concept_graph_path=settings.data_path / 'concept_graph.json')
            app.state.knowledge_tracing = service
        except Exception as exc:
            logger.error(f'Failed to initialize knowledge tracing service: {exc}')
            raise HTTPException(status_code=500, detail='Failed to initialize knowledge tracing service')
    return service


def _get_student_neighbors(cache: Dict[str, Any], student_id: int, top_k: int = 5):
    ids = cache.get('ids', [])
    if student_id not in ids:
        raise HTTPException(status_code=404, detail=f'Student {student_id} not found in embedding cache')
    idx_map = {sid: i for i, sid in enumerate(ids)}
    idx = idx_map[student_id]
    X = cache.get('X')
    if X is None or cache.get('nn') is None:
        raise HTTPException(status_code=500, detail='Nearest neighbor index unavailable')
    n = min(top_k + 1, X.shape[0])
    distances, indices = cache['nn'].kneighbors([X[idx]], n_neighbors=n)
    neighbors = []
    for dist, i in zip(distances[0], indices[0]):
        sid = ids[i]
        if sid == student_id:
            continue
        neighbors.append({'student_id': int(sid), 'distance': float(dist)})
        if len(neighbors) >= top_k:
            break
    return neighbors


def _load_interactions_df() -> Any:
    interactions_file = settings.data_path / 'student_interactions.csv'
    if not interactions_file.exists():
        raise HTTPException(status_code=404, detail='Interactions CSV not found')
    import pandas as pd
    return pd.read_csv(interactions_file)


def _get_topic_embeddings(kg: KnowledgeGraph) -> Dict[str, Any]:
    embeddings_file = settings.data_path / 'embeddings.json'
    if embeddings_file.exists():
        try:
            raw = json.loads(embeddings_file.read_text())
            topic_emb = raw.get('topic_embeddings', {})
            if isinstance(topic_emb, dict) and topic_emb:
                return topic_emb
        except Exception:
            pass
    return kg.compute_concept_embeddings()


def _get_cluster_labels(cache: Dict[str, Any]) -> Dict[int, int]:
    labels = cache.get('cluster_labels', [])
    ids = cache.get('ids', [])
    return {int(ids[i]): int(labels[i]) for i in range(min(len(ids), len(labels)))} if labels else {}


@app.get("/results/recommendations", response_model=RecommendationResultsResponse, tags=["Results"])
async def get_recommendation_results():
    """Get saved recommendation evaluation results."""
    results_path = settings.results_path / "recommendation_results.json"
    return {"results": _load_json_file(results_path)}

@app.get("/results/rl", response_model=RLTrainingResultsResponse, tags=["Results"])
async def get_rl_results():
    """Get saved RL training results."""
    results_path = settings.results_path / "rl_training.json"
    return {"results": _load_json_file(results_path)}


@app.get('/cache/status', tags=['Cache'])
async def cache_status():
    """Return current embedding cache status and age."""
    cache = getattr(app.state, 'embedding_cache', None)
    if cache is None:
        return {'cached': False, 'age_seconds': None, 'topic': 'student_embeddings'}
    age = time.time() - cache.get('timestamp', 0)
    persisted_index_age = None
    cache_file = settings.model_checkpoint_path / 'nn_index.joblib'
    if cache_file.exists():
        persisted_index_age = time.time() - cache_file.stat().st_mtime

    return {
        'cached': True,
        'age_seconds': age,
        'student_count': len(cache.get('ids', [])),
        'cache_ttl': settings.cache_ttl,
        'persisted_nn_index_age_seconds': persisted_index_age,
    }


@app.post('/cache/refresh', tags=['Cache'])
async def cache_refresh():
    """Force refresh the student embedding cache and persisted neighbor index."""
    cache = _ensure_embedding_cache(app, force_refresh=True)
    if cache is None:
        raise HTTPException(status_code=500, detail='Failed to refresh cache')
    return {
        'cached': True,
        'student_count': len(cache.get('ids', [])),
        'timestamp': cache.get('timestamp'),
    }


@app.get('/recommendations/{student_id}', tags=['Recommendations'])
async def get_recommendations(student_id: int, model: str = 'hybrid', top_k: int = 5):
    """Return field-ready recommendations for a student with inline rationale."""
    interactions_file = settings.data_path / 'student_interactions.csv'
    if not interactions_file.exists():
        raise HTTPException(status_code=404, detail='Interactions CSV not found')

    import pandas as pd
    from recommendation_engine.baselines import CollaborativeFilteringBaseline, ContentBasedFilteringBaseline, HybridRecommender, explain_recommendation

    df = pd.read_csv(interactions_file)
    if df.empty:
        raise HTTPException(status_code=404, detail='No interaction data available')

    topics = sorted(df['topic'].dropna().unique().tolist())
    if not topics:
        raise HTTPException(status_code=404, detail='No topics found in interactions')

    # Build baseline models on demand
    recommendations = []
    student_ids = df['student_id'].unique().tolist()
    if student_id not in student_ids:
        # cold start: top popular topics
        popular = df['topic'].value_counts().head(top_k).index.tolist()
        for topic in popular:
            recommendations.append({
                'topic_id': topic,
                'topic_name': topic,
                'predicted_score': 0.0,
                'reason': 'Cold start: popular topic among all students',
                'recommended_difficulty': 0.5,
            })
        return {'student_id': student_id, 'recommendations': recommendations, 'recommendation_type': 'cold_start'}

    try:
        if model == 'cf':
            rec_model = CollaborativeFilteringBaseline()
            rec_model.fit(df)
            recs = rec_model.recommend(student_id, topics, top_k=top_k)
        elif model == 'content':
            # fallback: use simple topic features from the concept graph
            topic_embeddings = {}
            try:
                # try to use saved topic embeddings if present
                embeddings_file = settings.data_path / 'embeddings.json'
                if embeddings_file.exists():
                    raw = json.loads(embeddings_file.read_text())
                    topic_embeddings = {k: np.array(v) for k, v in raw.get('topic_embeddings', {}).items()}
            except Exception:
                topic_embeddings = {}
            if not topic_embeddings:
                # use knowledge graph structure as fallback
                kg = app.state.kg
                topic_embeddings = {topic: kg.compute_concept_embeddings().get(topic, np.zeros(8)) for topic in topics}
            rec_model = ContentBasedFilteringBaseline()
            rec_model.fit(df, topic_embeddings)
            recs = rec_model.recommend(student_id, topics, top_k=top_k)
        else:
            # hybrid recommendation
            topic_embeddings = {}
            try:
                embeddings_file = settings.data_path / 'embeddings.json'
                if embeddings_file.exists():
                    raw = json.loads(embeddings_file.read_text())
                    topic_embeddings = {k: np.array(v) for k, v in raw.get('topic_embeddings', {}).items()}
            except Exception:
                topic_embeddings = {}
            if not topic_embeddings:
                kg = app.state.kg
                topic_embeddings = {topic: kg.compute_concept_embeddings().get(topic, np.zeros(8)) for topic in topics}
            rec_model = HybridRecommender()
            rec_model.fit(df, topic_embeddings)
            recs = rec_model.recommend(student_id, topics, top_k=top_k)
    except Exception as exc:
        logger.error(f'Failed to build recommendation model: {exc}')
        raise HTTPException(status_code=500, detail='Recommendation generation failed')

    # Prepare explanations for each recommended topic
    cache = _ensure_embedding_cache(app)
    features = cache.get('features', {}) if cache else {}
    recs_with_reason = []
    for rec in recs:
        topic = rec.get('topic')
        score = rec.get('score', 0.0)
        explain_payload = {'student_id': student_id, 'topic': topic, 'n_neighbors': 3}
        try:
            explanation = await recommend_explain(explain_payload)
            reason = explanation['explanation'].get('reason', '')
        except Exception as exc:
            reason = f'Unable to compute explanation: {exc}'
        recs_with_reason.append({
            'topic_id': topic,
            'topic_name': topic,
            'predicted_score': float(score),
            'reason': reason,
            'recommended_difficulty': 0.5,
        })

    return {
        'student_id': student_id,
        'recommendations': recs_with_reason,
        'recommendation_type': model,
    }


# ==================== Knowledge Graph Endpoints ====================

@app.get("/kg/subgraph", tags=["KnowledgeGraph"])
async def kg_subgraph(concepts: str = ""):
    """Return visualization data for a comma-separated list of concepts"""
    kg: KnowledgeGraph = app.state.kg
    # If no concepts provided, return the full graph summary for convenience
    if not concepts:
        all_nodes = list(kg.graph.nodes())
        return kg.visualize_subgraph(all_nodes)

    concept_list = [c.strip() for c in concepts.split(',') if c.strip()]
    if not concept_list:
        # empty after stripping - return full graph
        return kg.visualize_subgraph(list(kg.graph.nodes()))
    return kg.visualize_subgraph(concept_list)


def _get_student_mastery(student_id: int) -> Dict[str, float]:
    service = _get_knowledge_tracing_service()
    mastery = service.get_student_mastery(student_id)
    if not mastery:
        raise HTTPException(status_code=404, detail=f'Student {student_id} mastery not found')
    return mastery


@app.get('/kg/prerequisites/{topic}', tags=['KnowledgeGraph'])
async def kg_prerequisites(topic: str):
    """Return all prerequisite concepts required for a topic."""
    kg: KnowledgeGraph = app.state.kg
    if not topic:
        raise HTTPException(status_code=400, detail='Topic path parameter is required')
    prerequisites = sorted(kg.get_prerequisites(topic))
    return {'topic': topic, 'prerequisites': prerequisites}


@app.get('/kg/path', tags=['KnowledgeGraph'])
async def kg_path(topic: str = None):
    """Return a concept learning path for a topic based on prerequisites."""
    if not topic:
        raise HTTPException(status_code=400, detail='Query parameter topic is required')
    kg: KnowledgeGraph = app.state.kg
    path = kg.get_prerequisite_chain(topic)
    return {'topic': topic, 'path': path}


@app.get('/kg/readiness/{student_id}', tags=['KnowledgeGraph'])
async def kg_readiness(student_id: int, topic: str = None):
    """Return readiness scores for a student and optional target topic."""
    mastery = _get_student_mastery(student_id)
    kg: KnowledgeGraph = app.state.kg
    if topic:
        readiness = kg.get_readiness(topic, mastery)
        return {'student_id': student_id, 'topic': topic, 'readiness': readiness}
    scores = kg.get_readiness_scores(mastery)
    return {'student_id': student_id, 'readiness': scores}


@app.get('/kg/root_cause/{student_id}', tags=['KnowledgeGraph'])
async def kg_root_cause(student_id: int, topic: str = None):
    """Return root cause analysis for student struggles with a topic."""
    mastery = _get_student_mastery(student_id)
    kg: KnowledgeGraph = app.state.kg
    if not topic:
        topic = min(mastery, key=mastery.get)
    result = kg.get_root_cause(topic, mastery)
    return {'student_id': student_id, 'root_cause': result}


@app.get('/kg/remediation/{student_id}', tags=['KnowledgeGraph'])
async def kg_remediation(student_id: int, topic: str = None):
    """Return a remediation plan for a student and target topic."""
    mastery = _get_student_mastery(student_id)
    kg: KnowledgeGraph = app.state.kg
    if not topic:
        topic = min(mastery, key=mastery.get)
    result = kg.get_remediation_plan(topic, mastery)
    return {'student_id': student_id, 'remediation': result}


@app.post("/students/embeddings", tags=["Students"])
async def compute_student_embeddings():
    """Compute student embeddings from the saved interactions CSV and return summary."""
    interactions_file = settings.data_path / 'student_interactions.csv'
    if not interactions_file.exists():
        raise HTTPException(status_code=404, detail="Interactions CSV not found")
    try:
        import pandas as pd
        df = pd.read_csv(interactions_file)
        engine = StudentEmbeddingEngine(embedding_dim=32, results_path=settings.data_path)
        embeddings, metadata = engine.fit_transform(df)
        # Return small summary
        sample_id = next(iter(embeddings.keys()))
        # Refresh embedding cache in app state
        try:
            _ensure_embedding_cache(app)
        except Exception:
            pass
        return {
            'n_students': len(embeddings),
            'sample_student': int(sample_id),
            'neighbors_sample': engine.nearest_neighbors(embeddings, sample_id, n=5)
        }
    except Exception as exc:
        logger.error(f"Failed to compute embeddings: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/students/embeddings_data", tags=["Students"])
async def get_student_embeddings_data():
    """Return the saved student embeddings JSON produced by the embedding engine."""
    embeddings_file = settings.data_path / 'student_embeddings.json'
    return {"results": _load_json_file(embeddings_file)}


@app.get("/students/profile/{student_id}", tags=["Students"])
async def get_student_profile(student_id: int, top_k: int = 5):
    """Return a research profile for a student with embedding metadata and neighbors."""
    cache = _get_embedding_cache(app)
    if student_id not in cache['embeddings']:
        raise HTTPException(status_code=404, detail=f"Student {student_id} not found")

    ids = cache['ids']
    idx = ids.index(student_id)
    cluster_label = None
    if idx < len(cache.get('cluster_labels', [])):
        cluster_label = cache['cluster_labels'][idx]

    coords_pca = cache.get('coords_pca', {}).get(str(student_id)) or cache.get('coords_pca', {}).get(student_id)
    coords_tsne = cache.get('coords_tsne', {}).get(str(student_id)) or cache.get('coords_tsne', {}).get(student_id)

    profile = {
        'student_id': student_id,
        'cluster_label': cluster_label,
        'features': cache['features'].get(student_id, {}),
        'embedding': cache['embeddings'][student_id],
        'coords_pca': coords_pca,
        'coords_tsne': coords_tsne,
        'similar_students': _get_student_neighbors(cache, student_id, top_k=top_k)
    }
    return profile


@app.get("/students/similar/{student_id}", tags=["Students"])
async def get_similar_students(student_id: int, top_k: int = 5):
    """Return nearest neighbor students for a given student."""
    cache = _get_embedding_cache(app)
    neighbors = _get_student_neighbors(cache, student_id, top_k=top_k)
    return {'student_id': student_id, 'neighbors': neighbors}


@app.get("/students/clusters", tags=["Students"])
async def get_student_clusters():
    """Return cluster summary statistics and assignments for student embeddings."""
    cache = _get_embedding_cache(app)
    cluster_labels = cache.get('cluster_labels', [])
    ids = cache.get('ids', [])
    cluster_sizes = {}
    for label in cluster_labels:
        cluster_sizes[str(label)] = cluster_sizes.get(str(label), 0) + 1

    cluster_assignments = {
        str(student_id): int(cluster_labels[i])
        for i, student_id in enumerate(ids)
        if i < len(cluster_labels)
    }

    return {
        'student_count': len(ids),
        'cluster_sizes': cluster_sizes,
        'cluster_metrics': cache.get('cluster_metrics', {}),
        'experiment_metrics': cache.get('experiment_metrics', {}),
        'cluster_assignments': cluster_assignments
    }


@app.get("/students/embeddings/experiments", tags=["Students"])
async def get_student_embedding_experiments():
    """Return embedding experiment metrics across dimensionalities."""
    cache = _get_embedding_cache(app)
    ids = cache.get('ids', [])
    return {
        'student_count': len(ids),
        'cluster_metrics': cache.get('cluster_metrics', {}),
        'experiment_metrics': cache.get('experiment_metrics', {}),
        'feature_count': len(next(iter(cache.get('features', {}).values())) if cache.get('features') else 0)
    }


@app.get('/students/{student_id}/mastery', tags=['KnowledgeTracing'])
async def get_student_mastery(student_id: int):
    """Return concept-level mastery estimates for a student."""
    service = _get_knowledge_tracing_service()
    mastery = service.get_student_mastery(student_id)
    if not mastery:
        raise HTTPException(status_code=404, detail=f'Student {student_id} mastery not found')
    return {'student_id': student_id, 'mastery': mastery}


@app.get('/students/{student_id}/weak_concepts', tags=['KnowledgeTracing'])
async def get_student_weak_concepts(student_id: int, threshold: float = 0.65, top_k: int = 5):
    """Return the weakest concepts for a student based on mastery."""
    service = _get_knowledge_tracing_service()
    weak_concepts = service.get_weak_concepts(student_id, threshold=threshold, top_k=top_k)
    return {'student_id': student_id, 'weak_concepts': weak_concepts}


@app.get('/students/{student_id}/trajectory', tags=['KnowledgeTracing'])
async def get_student_trajectory(student_id: int, concept: str):
    """Return a concept mastery trajectory for a student."""
    if not concept:
        raise HTTPException(status_code=400, detail='Concept query parameter is required')
    service = _get_knowledge_tracing_service()
    trajectory = service.get_trajectory(student_id, concept)
    if not trajectory:
        raise HTTPException(status_code=404, detail=f'No trajectory found for concept {concept}')
    return {'student_id': student_id, 'concept': concept, 'trajectory': trajectory}


@app.get('/knowledge-tracing/metrics', tags=['KnowledgeTracing'])
async def get_knowledge_tracing_metrics():
    """Return evaluation metrics for knowledge tracing predictions."""
    service = _get_knowledge_tracing_service()
    return {'metrics': service.metrics}


@app.get('/knowledge-tracing/heatmap', tags=['KnowledgeTracing'])
async def get_knowledge_tracing_heatmap(max_students: int = 20):
    """Return a mastery heatmap for the top students."""
    service = _get_knowledge_tracing_service()
    return service.get_heatmap(max_students=max_students)


@app.post("/kg/recommend_next", tags=["KnowledgeGraph"])
async def kg_recommend_next(payload: Dict[str, float]):
    """Recommend next concepts given a mastery level dict (concept -> mastery 0..1)"""
    kg: KnowledgeGraph = app.state.kg
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expecting JSON dict of mastery levels")
    recommendations = kg.get_recommended_next_concepts(payload, top_k=10)
    return {"recommendations": recommendations}


@app.post("/kg/infer_weaknesses", tags=["KnowledgeGraph"])
async def kg_infer_weaknesses(payload: Dict[str, float]):
    """Infer likely prerequisite weaknesses from a mastery dict"""
    kg: KnowledgeGraph = app.state.kg
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Expecting JSON dict of mastery levels")
    inferred = kg.infer_weaknesses(payload)
    return {"weaknesses": inferred}


@app.post('/recommend/explain', tags=['Recommendations'])
async def recommend_explain(payload: Dict):
    """Explain recommendation using student embeddings and interaction history.

    Expects JSON: {student_id: int, topic: str, n_neighbors: int (optional)}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail='Expecting JSON payload')

    student_id = int(payload.get('student_id')) if payload.get('student_id') is not None else None
    topic = payload.get('topic')
    n_neighbors = int(payload.get('n_neighbors', 5))

    if student_id is None or not topic:
        raise HTTPException(status_code=400, detail='Provide student_id and topic')

    # Load embeddings and interactions
    embeddings_file = settings.data_path / 'student_embeddings.json'
    interactions_file = settings.data_path / 'student_interactions.csv'
    if not embeddings_file.exists() or not interactions_file.exists():
        raise HTTPException(status_code=404, detail='Required artifacts not found')

    import pandas as pd
    from recommendation_engine.baselines import explain_recommendation

    # Prefer in-memory cache for speed
    cache = _ensure_embedding_cache(app)
    interactions_df = pd.read_csv(interactions_file)

    if cache:
        try:
            embeddings = cache['embeddings']
            features = cache.get('features', {})
            ids = cache['ids']
            X = cache['X']
            nn = cache.get('nn')

            # If nn is available and student in ids, use it to fetch neighbors quickly
            if nn is not None and student_id in ids:
                idx_map = {sid: i for i, sid in enumerate(ids)}
                tidx = idx_map[student_id]
                n = min(n_neighbors + 1, X.shape[0])
                distances, indices = nn.kneighbors([X[tidx]], n_neighbors=n)
                # convert indices to ids and compute neighbor info via explain_recommendation logic
                neighbor_ids = [ids[i] for i in indices[0] if ids[i] != student_id]
                # Build a reduced embeddings dict for explain_recommendation
                reduced_emb = {int(k): embeddings[int(k)] for k in neighbor_ids + [student_id]}
                explanation = explain_recommendation(student_id, topic, reduced_emb, interactions_df, n_neighbors=n_neighbors, features=features)
                return {'explanation': explanation}
        except Exception as exc:
            logger.warning(f'Cache-based explanation failed: {exc}')

    # Fallback: load from files and compute
    raw = json.loads(embeddings_file.read_text())
    embeddings = raw.get('embeddings', {})
    features = raw.get('features', {})
    # normalize keys to ints
    embeddings = {int(k): v for k, v in embeddings.items()}
    features = {int(k): v for k, v in features.items()} if isinstance(features, dict) else {}

    try:
        explanation = explain_recommendation(student_id, topic, embeddings, interactions_df, n_neighbors=n_neighbors, features=features)
        return {'explanation': explanation}
    except Exception as exc:
        logger.error(f'Explain failed: {exc}')
        raise HTTPException(status_code=500, detail=str(exc))


@app.post('/recommend/explain_v2', response_model=ExplainableRecommendationResponse, tags=['Recommendations'])
async def recommend_explain_v2(payload: Dict[str, Any]):
    """Return explainable recommendations using embeddings, tracing, and graph readiness."""
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail='Expecting JSON payload')

    student_id = int(payload.get('student_id')) if payload.get('student_id') is not None else None
    top_k = int(payload.get('top_k', 5))
    n_neighbors = int(payload.get('n_neighbors', 10))
    if student_id is None:
        raise HTTPException(status_code=400, detail='Provide student_id')

    kg: KnowledgeGraph = app.state.kg
    interactions_df = _load_interactions_df()
    mastery = _get_student_mastery(student_id)

    # Load topic embeddings and student embeddings
    topic_embeddings = _get_topic_embeddings(kg)
    cache = _ensure_embedding_cache(app)
    if cache is None:
        raise HTTPException(status_code=503, detail='Embedding cache unavailable')

    student_embeddings = cache.get('embeddings', {})
    cluster_labels = _get_cluster_labels(cache)

    weights = {
        'similarity': payload.get('weights', {}).get('similarity', settings.rec_similarity_weight),
        'mastery': payload.get('weights', {}).get('mastery', settings.rec_mastery_weight),
        'readiness': payload.get('weights', {}).get('readiness', settings.rec_readiness_weight),
        'cluster_success': payload.get('weights', {}).get('cluster_success', settings.rec_cluster_weight),
    }

    explainer = ExplainableRecommender(neighbor_count=n_neighbors)
    recommendations = explainer.explain_v2(
        student_id,
        interactions_df,
        mastery,
        kg,
        student_embeddings,
        topic_embeddings=topic_embeddings,
        top_k=top_k,
        weights=weights,
        cluster_labels=cluster_labels,
    )

    return {
        'student_id': student_id,
        'recommendations': recommendations,
        'top_k': top_k,
        'weights': weights,
        'timestamp': datetime.utcnow(),
    }


@app.get('/recommend/evidence/{student_id}', response_model=RecommendationEvidenceResponse, tags=['Recommendations'])
async def recommend_evidence(student_id: int, top_k: int = 5):
    """Return peer evidence for explainable recommendations."""
    kg: KnowledgeGraph = app.state.kg
    interactions_df = _load_interactions_df()
    mastery = _get_student_mastery(student_id)
    topic_embeddings = _get_topic_embeddings(kg)
    cache = _ensure_embedding_cache(app)
    if cache is None:
        raise HTTPException(status_code=503, detail='Embedding cache unavailable')

    student_embeddings = cache.get('embeddings', {})
    cluster_labels = _get_cluster_labels(cache)

    explainer = ExplainableRecommender(neighbor_count=10)
    recommendations = explainer.explain_v2(
        student_id,
        interactions_df,
        mastery,
        kg,
        student_embeddings,
        topic_embeddings=topic_embeddings,
        top_k=top_k,
        cluster_labels=cluster_labels,
    )
    evidence = explainer.evidence_summary(student_id, interactions_df, student_embeddings, top_k=top_k)

    return {
        'student_id': student_id,
        'evidence': evidence,
        'recommendations': recommendations,
        'timestamp': datetime.utcnow(),
    }


@app.get('/recommend/confidence/{student_id}', response_model=RecommendationConfidenceResponse, tags=['Recommendations'])
async def recommend_confidence(student_id: int, top_k: int = 5):
    """Return confidence summary for explainable recommendations."""
    kg: KnowledgeGraph = app.state.kg
    interactions_df = _load_interactions_df()
    mastery = _get_student_mastery(student_id)
    topic_embeddings = _get_topic_embeddings(kg)
    cache = _ensure_embedding_cache(app)
    if cache is None:
        raise HTTPException(status_code=503, detail='Embedding cache unavailable')

    student_embeddings = cache.get('embeddings', {})
    cluster_labels = _get_cluster_labels(cache)

    explainer = ExplainableRecommender(neighbor_count=10)
    summary = explainer.confidence_summary(
        student_id,
        interactions_df,
        mastery,
        kg,
        student_embeddings,
        topic_embeddings=topic_embeddings,
        top_k=top_k,
    )

    return {
        'student_id': student_id,
        'average_confidence': summary['average_confidence'],
        'recommendations': summary['recommendations'],
        'timestamp': datetime.utcnow(),
    }


@app.get('/recommend/readiness/{student_id}', response_model=RecommendationReadinessResponse, tags=['Recommendations'])
async def recommend_readiness(student_id: int, topic: Optional[str] = None):
    """Return readiness scores for a student across all topics or a target topic."""
    kg: KnowledgeGraph = app.state.kg
    mastery = _get_student_mastery(student_id)
    explainer = ExplainableRecommender()
    summary = explainer.readiness_summary(student_id, mastery, kg, topic=topic)

    if topic:
        readiness_map = {topic: summary['readiness']}
    else:
        readiness_map = summary['readiness']

    return {
        'student_id': student_id,
        'readiness': readiness_map,
        'timestamp': datetime.utcnow(),
    }


@app.get('/recommend/benchmark', response_model=RecommendationBenchmarkResponse, tags=['Recommendations'])
async def recommend_benchmark(top_k: int = 5, sample_fraction: float = 0.2, evaluation: str = 'full'):
    """Compute benchmark metrics for popularity, CF, sequential, and explainable models."""
    if evaluation not in {'full', 'temporal'}:
        raise HTTPException(status_code=400, detail="evaluation must be 'full' or 'temporal'")

    kg: KnowledgeGraph = app.state.kg
    interactions_df = _load_interactions_df()
    topic_embeddings = getattr(app.state, 'topic_embeddings', {}) or _get_topic_embeddings(kg)
    cache = _ensure_embedding_cache(app)
    student_embeddings = cache.get('embeddings', {}) if cache else {}

    if evaluation == 'temporal':
        train_df, test_df = temporal_train_test_split(interactions_df, test_fraction=sample_fraction)
        student_ids = sorted(test_df['student_id'].dropna().unique().tolist())
        if not student_ids:
            raise HTTPException(status_code=404, detail='No students available for temporal evaluation')
    else:
        train_df = interactions_df
        test_df = interactions_df
        student_ids = sorted(interactions_df['student_id'].dropna().unique().tolist())
        if sample_fraction < 1.0:
            count = max(1, int(len(student_ids) * float(sample_fraction)))
            student_ids = student_ids[:count]

    from recommendation_engine.baselines import PopularityBaseline, CollaborativeFilteringBaseline, SequentialTransitionBaseline

    popularity = PopularityBaseline()
    popularity.fit(train_df)

    collaborative = CollaborativeFilteringBaseline()
    collaborative.fit(train_df)

    sequential = SequentialTransitionBaseline()
    sequential.fit(train_df)

    explainer = getattr(app.state, 'explainable_recommender', None)
    if explainer is None:
        explainer = ExplainableRecommender(neighbor_count=10)

    def _compute_metrics(model):
        precisions = []
        recalls = []
        ndcgs = []
        mrrs = []

        for sid in student_ids:
            if evaluation == 'temporal':
                y_true = set(test_df[test_df['student_id'] == sid]['topic'].dropna().unique().tolist())
                if not y_true:
                    continue
                recs = model.explain_v2(
                    sid,
                    train_df,
                    {},
                    kg,
                    student_embeddings,
                    topic_embeddings=topic_embeddings,
                    top_k=top_k,
                )
            else:
                y_true = set(interactions_df[interactions_df['student_id'] == sid]['topic'].dropna().unique().tolist())
                recs = model.explain_v2(
                    sid,
                    interactions_df,
                    {},
                    kg,
                    student_embeddings,
                    topic_embeddings=topic_embeddings,
                    top_k=top_k,
                )

            y_pred = [r['topic'] for r in recs]
            precisions.append(evaluator.precision_at_k(y_true, y_pred, top_k))
            recalls.append(evaluator.recall_at_k(y_true, y_pred, top_k))
            ndcgs.append(evaluator.ndcg_at_k(y_true, y_pred, top_k))
            mrrs.append(evaluator.mrr(y_true, y_pred))

        count = len(precisions) or 1
        return {
            f'precision_at_{top_k}': float(np.mean(precisions)) if precisions else 0.0,
            f'recall_at_{top_k}': float(np.mean(recalls)) if recalls else 0.0,
            f'ndcg_at_{top_k}': float(np.mean(ndcgs)) if ndcgs else 0.0,
            'mrr': float(np.mean(mrrs)) if mrrs else 0.0,
        }

    return {
        'popularity': _compute_metrics(popularity),
        'collaborative_filtering': _compute_metrics(collaborative),
        'sequential': _compute_metrics(sequential),
        'explainable': _compute_metrics(explainer),
    }


@app.get('/recommend/metrics', tags=['Recommendations'])
async def recommend_metrics(top_k: int = 5, sample_fraction: float = 1.0, evaluation: str = 'full'):
    """Compute evaluation metrics for the explainable recommender across students.

    Note: simple offline evaluation using interactions as ground truth. This is
    a lightweight research metric endpoint for quick snapshots.
    """
    kg: KnowledgeGraph = app.state.kg
    interactions_df = _load_interactions_df()

    # explainer instance
    explainer = getattr(app.state, 'explainable_recommender', None)
    if explainer is None:
        explainer = ExplainableRecommender(neighbor_count=10)

    topic_embeddings = getattr(app.state, 'topic_embeddings', {}) or _get_topic_embeddings(kg)

    if evaluation == 'temporal':
        # perform temporal per-student split
        train_df, test_df = temporal_train_test_split(interactions_df, test_fraction=sample_fraction)
        # evaluate on users that have test items
        student_ids = sorted(test_df['student_id'].dropna().unique().tolist())
        metrics = evaluator.evaluate_with_holdout(
            explainer,
            train_df,
            test_df,
            student_ids,
            topic_embeddings=topic_embeddings,
            top_k=top_k,
        )
    else:
        # full evaluation on available interactions
        student_ids = sorted(interactions_df['student_id'].dropna().unique().tolist())
        metrics = evaluator.evaluate_all(
            explainer,
            interactions_df,
            student_ids,
            topic_embeddings=topic_embeddings,
            top_k=top_k,
            sample_fraction=sample_fraction,
        )

    return {
        'top_k': top_k,
        'sample_fraction': sample_fraction,
        'metrics': metrics,
        'timestamp': datetime.utcnow(),
    }


@app.get('/topics', tags=['Topics'])
async def list_topics():
    """Return list of unique topics from interactions CSV"""
    interactions_file = settings.data_path / 'student_interactions.csv'
    if not interactions_file.exists():
        raise HTTPException(status_code=404, detail='Interactions CSV not found')
    import pandas as pd
    try:
        df = pd.read_csv(interactions_file)
        topics = sorted(df['topic'].dropna().unique().tolist())
        return {'topics': topics}
    except Exception as exc:
        logger.error(f'Failed to list topics: {exc}')
        raise HTTPException(status_code=500, detail=str(exc))


# ==================== Instructor Analytics ====================
from backend.instructor_analytics import compute_cohort_overview, detect_at_risk_students


@app.get('/instructor/cohort_overview', tags=['Instructor'])
async def instructor_cohort_overview(weak_threshold: float = 0.65):
    """Return cohort-level summary metrics for instructors."""
    interactions_file = settings.data_path / 'student_interactions.csv'
    try:
        overview = compute_cohort_overview(interactions_file, weak_threshold=weak_threshold)
        return overview
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Interactions CSV not found')
    except Exception as exc:
        logger.error(f'Failed to compute cohort overview: {exc}')
        raise HTTPException(status_code=500, detail='Cohort analytics failed')


@app.get('/instructor/at_risk', tags=['Instructor'])
async def instructor_at_risk(mastery_threshold: float = 0.6, max_results: int = 50):
    """Return list of at-risk students for instructors to review."""
    interactions_file = settings.data_path / 'student_interactions.csv'
    try:
        rows = detect_at_risk_students(interactions_file, mastery_threshold=mastery_threshold, top_k=max_results)
        return {'at_risk': rows, 'count': len(rows)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail='Interactions CSV not found')
    except Exception as exc:
        logger.error(f'Failed to detect at-risk students: {exc}')
        raise HTTPException(status_code=500, detail='At-risk detection failed')

# ==================== Spaced Repetition Endpoints ====================

@app.get('/reviews/due/{student_id}', response_model=DueReviewsResponse, tags=['Spaced Repetition'])
async def get_due_reviews(student_id: int):
    """Get topics due for review (review overdue or due today)."""
    scheduler = getattr(app.state, 'sr_scheduler', None)
    if not scheduler:
        raise HTTPException(status_code=503, detail='Spaced repetition scheduler unavailable')
    
    due = scheduler.get_due_reviews(student_id)
    return DueReviewsResponse(
        student_id=student_id,
        due_reviews=due,
        total_due=len(due),
        timestamp=datetime.utcnow()
    )


@app.post('/reviews/complete', response_model=ReviewCompleteResponse, tags=['Spaced Repetition'])
async def complete_review(request: ReviewCompleteRequest):
    """Record a completed review and update spaced repetition state."""
    scheduler = getattr(app.state, 'sr_scheduler', None)
    if not scheduler:
        raise HTTPException(status_code=503, detail='Spaced repetition scheduler unavailable')
    
    result = scheduler.record_review(
        student_id=request.student_id,
        topic=request.topic,
        quality=request.quality,
        mastery=request.mastery
    )
    
    return ReviewCompleteResponse(
        topic=result['topic'],
        quality=result['quality'],
        ease_factor=result['ease_factor'],
        repetition=result['repetition'],
        interval=result['interval'],
        next_review=result['next_review'],
        timestamp=datetime.utcnow()
    )


@app.get('/reviews/schedule/{student_id}', response_model=ReviewScheduleResponse, tags=['Spaced Repetition'])
async def get_review_schedule(student_id: int, days_ahead: int = 7):
    """Get review schedule for next N days."""
    scheduler = getattr(app.state, 'sr_scheduler', None)
    if not scheduler:
        raise HTTPException(status_code=503, detail='Spaced repetition scheduler unavailable')
    
    schedule = scheduler.get_schedule(student_id, days_ahead=days_ahead)
    
    return ReviewScheduleResponse(
        student_id=student_id,
        schedule_window_days=days_ahead,
        total_reviews=schedule['total_reviews'],
        by_date=schedule['by_date'],
        timestamp=datetime.utcnow()
    )


@app.get('/reviews/retention/{student_id}/{topic}', response_model=RetentionForecastResponse, tags=['Spaced Repetition'])
async def get_retention_forecast(student_id: int, topic: str, forecast_days: int = 30):
    """Forecast retention curve for a topic over time."""
    scheduler = getattr(app.state, 'sr_scheduler', None)
    if not scheduler:
        raise HTTPException(status_code=503, detail='Spaced repetition scheduler unavailable')
    
    forecast = scheduler.get_retention_forecast(student_id, topic, forecast_days=forecast_days)
    
    return RetentionForecastResponse(
        student_id=student_id,
        topic=forecast['topic'],
        ease_factor=forecast['ease_factor'],
        strength=forecast['strength'],
        current_retention=forecast['current_retention'],
        retention_curve=[
            {'days': day, 'retention': retention}
            for day, retention in forecast['retention_curve']
        ],
        days_until_90_percent=forecast['days_until_90_percent'],
        forecast_window_days=forecast_days,
        timestamp=datetime.utcnow()
    )


@app.get('/reviews/status/{student_id}', response_model=RetentionStatusResponse, tags=['Spaced Repetition'])
async def get_retention_status(student_id: int):
    """Get overall retention status across all topics."""
    scheduler = getattr(app.state, 'sr_scheduler', None)
    if not scheduler:
        raise HTTPException(status_code=503, detail='Spaced repetition scheduler unavailable')
    
    kt_service = _get_knowledge_tracing_service()
    mastery_levels = kt_service.get_student_mastery(student_id) if kt_service else {}
    
    status = scheduler.get_student_retention_status(student_id, mastery_levels)
    
    return RetentionStatusResponse(
        student_id=status['student_id'],
        overall_retention=status['overall_retention'],
        due_count=status['due_count'],
        topics=status['topics'],
        timestamp=datetime.utcnow()
    )


@app.get('/reviews/statistics/{student_id}', response_model=ReviewStatisticsResponse, tags=['Spaced Repetition'])
async def get_review_statistics(student_id: int):
    """Get review statistics for a student."""
    scheduler = getattr(app.state, 'sr_scheduler', None)
    if not scheduler:
        raise HTTPException(status_code=503, detail='Spaced repetition scheduler unavailable')
    
    stats = scheduler.get_statistics(student_id)
    
    return ReviewStatisticsResponse(
        student_id=student_id,
        statistics=stats,
        timestamp=datetime.utcnow()
    )

if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.max_workers,
        log_level=settings.log_level.lower(),
    )
