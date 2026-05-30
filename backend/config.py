"""Backend Configuration

Core settings for the Elo Learn platform, including database,
ML models, and API configuration.
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    # API Configuration
    api_title: str = "Elo Learn API"
    api_description: str = "Adaptive Learning Platform with RL and Recommendations"
    api_version: str = "1.0.0-alpha"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = True
    
    # Database
    database_url: str = "postgresql://user:password@localhost:5432/elo_learn"
    redis_url: str = "redis://localhost:6379/0"
    
    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_path: Path = project_root / "datasets"
    model_checkpoint_path: Path = project_root / "checkpoints"
    results_path: Path = project_root / "results"
    
    # ML Configuration
    embedding_dim: int = 64
    student_embedding_dim: int = 32
    concept_embedding_dim: int = 32
    batch_size: int = 32
    
    # RL Configuration
    rl_learning_rate: float = 0.001
    rl_gamma: float = 0.99
    rl_epsilon_decay: float = 0.995
    rl_buffer_size: int = 100000
    rl_episode_steps: int = 1000
    
    # Recommendation Configuration
    rec_top_k: int = 5
    rec_batch_size: int = 32
    rec_similarity_threshold: float = 0.5
    rec_similarity_weight: float = 0.35
    rec_mastery_weight: float = 0.25
    rec_readiness_weight: float = 0.25
    rec_cluster_weight: float = 0.15

    # LLM Configuration
    openai_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 3600
    
    # Performance
    max_workers: int = 4
    cache_ttl: int = 3600
    
    # Logging
    log_level: str = "INFO"
    log_file: Path = project_root / "logs" / "app.log"
    
    model_config = {
        'env_file': '.env',
        'case_sensitive': False,
        'protected_namespaces': ('settings_',),
    }

# Global settings instance
settings = Settings()

# Ensure necessary directories exist
settings.data_path.mkdir(parents=True, exist_ok=True)
settings.model_checkpoint_path.mkdir(parents=True, exist_ok=True)
settings.results_path.mkdir(parents=True, exist_ok=True)
settings.log_file.parent.mkdir(parents=True, exist_ok=True)
