"""Database models for Elo Learn platform

Defines core data models for:
- Student profiles and interactions
- Learning topics and concepts
- Quiz questions and attempts
- Performance metrics
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

Base = declarative_base()

class Student(Base):
    """Student profile and engagement tracking"""
    __tablename__ = "students"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    username = Column(String(255), unique=True, index=True)
    email = Column(String(255), unique=True, index=True)
    
    # Learning Profile
    total_interactions = Column(Integer, default=0)
    total_correct = Column(Integer, default=0)
    current_skill_level = Column(Float, default=0.0)  # 0-1
    overall_retention = Column(Float, default=0.0)  # 0-1
    average_time_spent = Column(Float, default=0.0)  # seconds
    
    # Embeddings (for recommendations)
    embedding = Column(JSON, nullable=True)  # Student feature vector
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    last_active = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    interactions = relationship("StudentInteraction", back_populates="student", cascade="all, delete-orphan")
    quiz_attempts = relationship("QuizAttempt", back_populates="student", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Student {self.username}>"


class Topic(Base):
    """Learning topics/subjects"""
    __tablename__ = "topics"
    
    id = Column(Integer, primary_key=True, index=True)
    topic_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), index=True)
    description = Column(Text, nullable=True)
    category = Column(String(100), index=True)  # e.g., "Math", "CS", "Science"
    difficulty = Column(Float, default=0.5)  # 0-1 (normalized difficulty)
    
    # Learning Objective
    bloom_level = Column(String(50), default="understand")  # remember, understand, apply, analyze, evaluate, create
    prerequisites = Column(JSON, nullable=True)  # List of prerequisite topic IDs
    
    # Metrics
    average_mastery = Column(Float, default=0.0)
    student_count = Column(Integer, default=0)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def __repr__(self):
        return f"<Topic {self.name}>"


class StudentInteraction(Base):
    """Individual learning interactions (quiz attempts, exercises)"""
    __tablename__ = "student_interactions"
    
    id = Column(Integer, primary_key=True, index=True)
    interaction_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    student_id = Column(Integer, ForeignKey("students.id"), index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), index=True)
    
    # Performance
    is_correct = Column(Boolean, index=True)
    score = Column(Float)  # 0-1
    time_spent = Column(Float)  # seconds
    attempts = Column(Integer, default=1)  # number of attempts before success
    
    # Difficulty tracking
    difficulty_presented = Column(Float)  # Difficulty level shown to student
    estimated_ability = Column(Float, nullable=True)  # IRT ability estimate
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    session_id = Column(String(36), index=True)  # Group interactions by session
    is_review = Column(Boolean, default=False)  # Is this a review/spaced repetition?
    
    # Relationships
    student = relationship("Student", back_populates="interactions")
    
    def __repr__(self):
        return f"<Interaction {self.student_id} -> {self.topic_id}>"


class QuizAttempt(Base):
    """Quiz and test attempt tracking"""
    __tablename__ = "quiz_attempts"
    
    id = Column(Integer, primary_key=True, index=True)
    attempt_id = Column(String(36), unique=True, index=True, default=lambda: str(uuid.uuid4()))
    
    # Foreign keys
    student_id = Column(Integer, ForeignKey("students.id"), index=True)
    quiz_id = Column(String(36), index=True)
    
    # Results
    total_questions = Column(Integer)
    correct_answers = Column(Integer)
    score = Column(Float)  # 0-1
    time_taken = Column(Float)  # seconds
    
    # Difficulty
    difficulty_level = Column(Float)  # 0-1
    adaptive_difficulty = Column(Boolean, default=False)  # Was difficulty adaptive?
    
    # Metadata
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    quiz_type = Column(String(50))  # "diagnostic", "practice", "assessment", "review"
    
    # Relationships
    student = relationship("Student", back_populates="quiz_attempts")
    
    def __repr__(self):
        return f"<QuizAttempt {self.student_id}>"


class StudentEmbedding(Base):
    """Stored student embeddings for recommendation similarity search"""
    __tablename__ = "student_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), unique=True, index=True)
    
    # Embedding
    embedding = Column(JSON)  # Vector representation
    embedding_dim = Column(Integer)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    embedding_model = Column(String(100))  # e.g., "collaborative_filtering_v1"
    
    def __repr__(self):
        return f"<StudentEmbedding {self.student_id}>"


class PerformancePrediction(Base):
    """Cached performance predictions"""
    __tablename__ = "performance_predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"), index=True)
    topic_id = Column(Integer, ForeignKey("topics.id"), index=True)
    
    # Predictions
    predicted_score = Column(Float)  # Predicted performance on this topic
    confidence = Column(Float)  # Confidence in prediction
    predicted_time = Column(Float)  # Predicted time needed
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String(100))  # Which model made prediction
    
    def __repr__(self):
        return f"<Prediction {self.student_id} -> {self.topic_id}>"
