"""Core Schemas for API requests/responses"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# ==================== Student Schemas ====================

class StudentBase(BaseModel):
    username: str
    email: str

class StudentCreate(StudentBase):
    pass

class StudentUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    current_skill_level: Optional[float] = None

class StudentResponse(StudentBase):
    id: int
    student_id: str
    total_interactions: int
    total_correct: int
    current_skill_level: float
    overall_retention: float
    average_time_spent: float
    created_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True

# ==================== Topic Schemas ====================

class TopicBase(BaseModel):
    name: str
    category: str
    difficulty: float = Field(default=0.5, ge=0.0, le=1.0)

class TopicCreate(TopicBase):
    description: Optional[str] = None
    bloom_level: Optional[str] = None
    prerequisites: Optional[List[str]] = None

class TopicResponse(TopicBase):
    id: int
    topic_id: str
    description: Optional[str]
    bloom_level: str
    average_mastery: float
    student_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# ==================== Interaction Schemas ====================

class InteractionCreate(BaseModel):
    student_id: int
    topic_id: int
    is_correct: bool
    score: float = Field(ge=0.0, le=1.0)
    time_spent: float = Field(ge=0.0)
    attempts: int = Field(default=1, ge=1)
    difficulty_presented: float = Field(ge=0.0, le=1.0)
    session_id: str
    is_review: bool = False

class InteractionResponse(InteractionCreate):
    id: int
    interaction_id: str
    estimated_ability: Optional[float]
    timestamp: datetime
    
    class Config:
        from_attributes = True

# ==================== Quiz Schemas ====================

class QuizAttemptCreate(BaseModel):
    student_id: int
    quiz_id: str
    total_questions: int
    correct_answers: int
    score: float = Field(ge=0.0, le=1.0)
    time_taken: float = Field(ge=0.0)
    difficulty_level: float = Field(ge=0.0, le=1.0)
    quiz_type: str = "practice"
    adaptive_difficulty: bool = False

class QuizAttemptResponse(QuizAttemptCreate):
    id: int
    attempt_id: str
    timestamp: datetime
    
    class Config:
        from_attributes = True

# ==================== Recommendation Schemas ====================

class TopicRecommendation(BaseModel):
    topic_id: str
    topic_name: str
    predicted_score: float
    reason: str
    recommended_difficulty: float

class RecommendationResponse(BaseModel):
    student_id: str
    recommendations: List[TopicRecommendation]
    recommendation_type: str  # "next_topic", "review", "challenge"
    timestamp: datetime

class PeerEvidence(BaseModel):
    similar_students: int
    average_mastery: float
    success_rate: float
    neighbor_ids: List[int]
    topic: str

class ExplainableTopicRecommendation(BaseModel):
    topic: str
    recommendation_score: float
    confidence: float
    readiness: float
    mastery: float
    similarity_score: float
    cluster_success_score: float
    reasons: List[str]
    peer_evidence: PeerEvidence

class ExplainableRecommendationResponse(BaseModel):
    student_id: int
    recommendations: List[ExplainableTopicRecommendation]
    top_k: int
    weights: Dict[str, float]
    timestamp: datetime

class RecommendationEvidenceResponse(BaseModel):
    student_id: int
    evidence: Dict[str, Any]
    recommendations: List[ExplainableTopicRecommendation]
    timestamp: datetime

class RecommendationConfidenceResponse(BaseModel):
    student_id: int
    average_confidence: float
    recommendations: List[Dict[str, Any]]
    timestamp: datetime

class RecommendationBenchmarkResponse(BaseModel):
    popularity: Dict[str, float]
    collaborative_filtering: Dict[str, float]
    sequential: Dict[str, float]
    explainable: Dict[str, float]

class RecommendationReadinessResponse(BaseModel):
    student_id: int
    readiness: Dict[str, float]
    timestamp: datetime

# ==================== RL Tutor Schemas ====================

class DifficultyAdjustment(BaseModel):
    current_difficulty: float
    recommended_difficulty: float
    adjustment_reason: str
    confidence: float

class RLTutorAction(BaseModel):
    student_id: str
    action: str  # "increase_difficulty", "maintain", "decrease_difficulty", "review"
    next_topic_id: str
    difficulty_level: float

# ==================== Analytics Schemas ====================

class StudentAnalytics(BaseModel):
    student_id: str
    total_interactions: int
    correct_rate: float
    average_time: float
    topics_mastered: List[str]
    topics_struggling: List[str]
    retention_rate: float
    learning_trajectory: List[Dict[str, Any]]

class SystemAnalytics(BaseModel):
    total_students: int
    total_interactions: int
    average_mastery: float
    recommendation_accuracy: float
    rl_avg_reward: float
    retention_improvement: float

class ResultPayload(BaseModel):
    results: Dict[str, Any]

class RecommendationResultsResponse(ResultPayload):
    pass

class RLTrainingResultsResponse(ResultPayload):
    pass

# ==================== Spaced Repetition Schemas ====================

class ReviewState(BaseModel):
    topic: str
    ease_factor: float
    repetition: int
    interval: int
    last_review: Optional[str]
    next_review: str

class ReviewRecord(BaseModel):
    student_id: int
    topic: str
    quality: int
    timestamp: str
    ease_factor: float
    repetition: int
    interval: int

class DueReviewResponse(BaseModel):
    topic: str
    ease_factor: float
    repetition: int
    status: str

class DueReviewsResponse(BaseModel):
    student_id: int
    due_reviews: List[DueReviewResponse]
    total_due: int
    timestamp: datetime

class ReviewCompleteRequest(BaseModel):
    student_id: int
    topic: str
    quality: int = Field(ge=0, le=5)
    mastery: Optional[float] = None

class ReviewCompleteResponse(BaseModel):
    topic: str
    quality: int
    ease_factor: float
    repetition: int
    interval: int
    next_review: str
    timestamp: datetime

class ScheduledReview(BaseModel):
    topic: str
    due_date: str
    ease_factor: float
    repetition: int
    days_until_due: int

class ReviewScheduleResponse(BaseModel):
    student_id: int
    schedule_window_days: int
    total_reviews: int
    by_date: Dict[str, List[ScheduledReview]]
    timestamp: datetime

class RetentionCurvePoint(BaseModel):
    days: float
    retention: float

class RetentionForecastResponse(BaseModel):
    student_id: int
    topic: str
    ease_factor: float
    strength: float
    current_retention: float
    retention_curve: List[RetentionCurvePoint]
    days_until_90_percent: float
    forecast_window_days: int
    timestamp: datetime

class TopicRetentionStatus(BaseModel):
    topic: str
    retention: float
    ease_factor: float
    repetition: int
    mastery: float
    due: bool
    review_count: int

class RetentionStatusResponse(BaseModel):
    student_id: int
    overall_retention: float
    due_count: int
    topics: List[TopicRetentionStatus]
    timestamp: datetime

class ReviewStatistics(BaseModel):
    total_topics: int
    due_reviews: int
    avg_ease_factor: float
    avg_repetitions: float
    topics_list: List[str]

class ReviewStatisticsResponse(BaseModel):
    student_id: int
    statistics: ReviewStatistics
    timestamp: datetime

# ==================== Health Check ====================

class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, bool]
