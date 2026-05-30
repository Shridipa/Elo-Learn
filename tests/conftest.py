"""Test Configuration and Fixtures

Provides reusable test fixtures and configurations for unit and integration tests.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Generator, Dict, List
import json

# ==================== Sample Data Fixtures ====================

@pytest.fixture
def sample_interactions_data() -> pd.DataFrame:
    """Sample student interaction data for testing"""
    data = {
        'student_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
        'topic': ['Vectors', 'Vectors', 'Matrices', 'Vectors', 'Matrices', 
                 'Matrices', 'Python Fundamentals', 'Python Data Structures', 'NumPy Basics'],
        'is_correct': [True, True, False, True, True, False, True, True, True],
        'score': [0.9, 0.85, 0.3, 0.95, 0.88, 0.4, 0.92, 0.87, 0.91],
        'time_spent': [300, 250, 450, 280, 320, 400, 200, 250, 300],
        'difficulty_presented': [0.5, 0.55, 0.6, 0.5, 0.6, 0.65, 0.3, 0.5, 0.5],
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_student_profiles() -> Dict:
    """Sample student profiles for testing"""
    return {
        1: {
            "student_id": 1,
            "proficiency": "intermediate",
            "learning_style": "visual",
            "base_ability": 0.65,
            "learning_speed": 1.0,
            "retention": 0.75,
        },
        2: {
            "student_id": 2,
            "proficiency": "beginner",
            "learning_style": "auditory",
            "base_ability": 0.45,
            "learning_speed": 0.8,
            "retention": 0.60,
        },
        3: {
            "student_id": 3,
            "proficiency": "advanced",
            "learning_style": "kinesthetic",
            "base_ability": 0.85,
            "learning_speed": 1.3,
            "retention": 0.88,
        },
    }

@pytest.fixture
def sample_concepts() -> Dict:
    """Sample concept hierarchy for testing"""
    return {
        "Vectors": {
            "difficulty": 0.5,
            "prerequisites": [],
            "category": "Mathematics"
        },
        "Matrices": {
            "difficulty": 0.6,
            "prerequisites": ["Vectors"],
            "category": "Mathematics"
        },
        "Linear Algebra Basics": {
            "difficulty": 0.4,
            "prerequisites": [],
            "category": "Mathematics"
        },
        "Python Fundamentals": {
            "difficulty": 0.3,
            "prerequisites": [],
            "category": "Programming"
        },
        "NumPy Basics": {
            "difficulty": 0.5,
            "prerequisites": ["Python Fundamentals"],
            "category": "Programming"
        },
    }

# ==================== Configuration Fixtures ====================

@pytest.fixture
def test_config() -> Dict:
    """Test configuration"""
    return {
        "embedding_dim": 32,
        "batch_size": 16,
        "learning_rate": 0.001,
        "epochs": 10,
        "random_seed": 42,
    }

# ==================== Mock Services ====================

class MockRecommendationEngine:
    """Mock recommendation engine for testing"""
    
    def __init__(self, concepts: Dict):
        self.concepts = concepts
        self.recommendations = {}
    
    def recommend(self, student_id: int, top_k: int = 5) -> List[Dict]:
        """Return mock recommendations"""
        topics = list(self.concepts.keys())
        return [
            {
                "topic_id": topics[i % len(topics)],
                "score": 0.9 - (i * 0.1),
                "reason": f"Mock recommendation {i+1}"
            }
            for i in range(min(top_k, len(topics)))
        ]

class MockRLAgent:
    """Mock RL tutor agent for testing"""
    
    def __init__(self):
        self.difficulty = 0.5
    
    def suggest_difficulty(self, student_id: int) -> float:
        """Return mock difficulty suggestion"""
        return self.difficulty
    
    def update(self, reward: float) -> None:
        """Mock update"""
        self.difficulty = np.clip(self.difficulty + reward * 0.1, 0, 1)

@pytest.fixture
def mock_recommendation_engine(sample_concepts) -> MockRecommendationEngine:
    """Fixture for mock recommendation engine"""
    return MockRecommendationEngine(sample_concepts)

@pytest.fixture
def mock_rl_agent() -> MockRLAgent:
    """Fixture for mock RL agent"""
    return MockRLAgent()

# ==================== Markers ====================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "ml: mark test as ML-related"
    )
