"""API Reference Documentation

Complete API endpoints for the Elo Learn platform.
"""

# ==================== Authentication ====================

## POST /auth/register
"""
Register a new student account

Request:
{
  "username": "student1",
  "email": "student1@example.com",
  "password": "secure_password"
}

Response:
{
  "student_id": "uuid",
  "username": "student1",
  "email": "student1@example.com",
  "created_at": "2024-05-29T10:00:00Z"
}
"""

## POST /auth/login
"""
Login and get JWT token

Request:
{
  "email": "student1@example.com",
  "password": "secure_password"
}

Response:
{
  "access_token": "jwt_token",
  "token_type": "bearer",
  "expires_in": 3600
}
"""

## POST /auth/logout
"""
Logout and invalidate token
"""

# ==================== Student Profile ====================

## GET /students/{student_id}
"""
Get student profile

Response:
{
  "student_id": "uuid",
  "username": "student1",
  "email": "student1@example.com",
  "total_interactions": 42,
  "total_correct": 35,
  "current_skill_level": 0.65,
  "overall_retention": 0.72,
  "average_time_spent": 245.5,
  "created_at": "2024-05-01T00:00:00Z",
  "is_active": true
}
"""

## POST /students
"""
Create new student (admin only)

Request:
{
  "username": "student1",
  "email": "student1@example.com"
}

Response:
{
  "id": 1,
  "student_id": "uuid",
  "username": "student1",
  ...
}
"""

## PUT /students/{student_id}
"""
Update student profile

Request:
{
  "username": "updated_name",
  "current_skill_level": 0.75
}

Response:
{
  "student_id": "uuid",
  ...
}
"""

## GET /students/{student_id}/analytics
"""
Get detailed student analytics

Response:
{
  "student_id": "uuid",
  "total_interactions": 42,
  "correct_rate": 0.833,
  "average_time": 245.5,
  "topics_mastered": ["Vectors", "Matrices"],
  "topics_struggling": ["Eigenvalues"],
  "retention_rate": 0.72,
  "learning_trajectory": [
    {"date": "2024-05-01", "score": 0.5},
    {"date": "2024-05-02", "score": 0.65},
    ...
  ]
}
"""

# ==================== Topics ====================

## GET /topics
"""
Get all available topics

Query parameters:
- category: Filter by category
- difficulty_min: Minimum difficulty
- difficulty_max: Maximum difficulty
- limit: Number of results

Response:
[
  {
    "id": 1,
    "topic_id": "uuid",
    "name": "Vectors",
    "category": "Mathematics",
    "difficulty": 0.5,
    "bloom_level": "understand",
    "average_mastery": 0.65,
    "student_count": 150
  },
  ...
]
"""

## GET /topics/{topic_id}
"""
Get single topic details

Response:
{
  "id": 1,
  "topic_id": "uuid",
  "name": "Vectors",
  "description": "Introduction to vector mathematics",
  "category": "Mathematics",
  "difficulty": 0.5,
  "bloom_level": "understand",
  "prerequisites": ["Linear Algebra Basics"],
  "average_mastery": 0.65,
  "student_count": 150
}
"""

## GET /topics/{topic_id}/prerequisites
"""
Get prerequisites for a topic

Response:
{
  "topic_id": "uuid",
  "topic_name": "Matrices",
  "prerequisites": [
    {
      "topic_id": "uuid",
      "topic_name": "Linear Algebra Basics",
      "required": true,
      "recommended": false
    }
  ]
}
"""

# ==================== Interactions (Tracking) ====================

## POST /interactions
"""
Record student interaction (quiz attempt)

Request:
{
  "student_id": 1,
  "topic_id": 1,
  "is_correct": true,
  "score": 0.95,
  "time_spent": 300,
  "difficulty_presented": 0.5,
  "session_id": "session_uuid",
  "is_review": false
}

Response:
{
  "id": 123,
  "interaction_id": "uuid",
  "student_id": 1,
  "topic_id": 1,
  "is_correct": true,
  "score": 0.95,
  "time_spent": 300,
  "timestamp": "2024-05-29T10:15:00Z"
}
"""

## GET /interactions/{student_id}
"""
Get all interactions for a student

Query parameters:
- topic_id: Filter by topic
- from_date: Start date
- to_date: End date
- limit: Number of results

Response:
[
  {
    "id": 123,
    "interaction_id": "uuid",
    "student_id": 1,
    "topic_id": 1,
    "is_correct": true,
    "score": 0.95,
    "time_spent": 300,
    "timestamp": "2024-05-29T10:15:00Z"
  },
  ...
]
"""

# ==================== Recommendations ====================

## GET /recommendations/{student_id}
"""
Get personalized topic recommendations

Query parameters:
- top_k: Number of recommendations (default: 5)
- type: "next_topic" | "review" | "challenge"

Response:
{
  "student_id": "uuid",
  "recommendations": [
    {
      "topic_id": "uuid",
      "topic_name": "Matrices",
      "predicted_score": 0.82,
      "reason": "You've mastered prerequisites. Difficulty matches your level.",
      "recommended_difficulty": 0.6
    },
    ...
  ],
  "recommendation_type": "next_topic",
  "timestamp": "2024-05-29T10:15:00Z"
}
"""

## GET /recommendations/{student_id}/explain
"""
Get explanation for a specific recommendation

Query parameters:
- topic_id: Topic to explain

Response:
{
  "student_id": "uuid",
  "topic_id": "uuid",
  "topic_name": "Matrices",
  "reasons": [
    {
      "reason": "You've mastered the prerequisite Linear Algebra Basics",
      "weight": 0.4
    },
    {
      "reason": "Similar students enjoyed this topic after Vectors",
      "weight": 0.35
    },
    ...
  ],
  "confidence": 0.85
}
"""

# ==================== RL Tutor / Adaptive ====================

## GET /tutor/{student_id}/difficulty
"""
Get adaptive difficulty suggestion

Response:
{
  "student_id": "uuid",
  "current_difficulty": 0.5,
  "recommended_difficulty": 0.6,
  "adjustment_reason": "You've achieved 90%+ on recent attempts",
  "confidence": 0.92
}
"""

## POST /tutor/{student_id}/action
"""
Record RL agent action (difficulty adjustment, etc)

Request:
{
  "action": "increase_difficulty",
  "topic_id": 1,
  "difficulty_level": 0.6,
  "next_topic_id": 2
}

Response:
{
  "student_id": "uuid",
  "action": "increase_difficulty",
  "difficulty_level": 0.6,
  "next_topic_id": 2,
  "timestamp": "2024-05-29T10:15:00Z"
}
"""

## GET /tutor/stats
"""
Get RL agent training statistics

Response:
{
  "avg_episode_reward": 42.3,
  "convergence_steps": 50000,
  "policy_stability": 0.92,
  "engagement_improvement": 0.15,
  "retention_improvement": 0.18
}
"""

# ==================== Analytics & Metrics ====================

## GET /analytics/student/{student_id}
"""
Get comprehensive student analytics

Response:
{
  "student_id": "uuid",
  "total_interactions": 42,
  "correct_rate": 0.833,
  "average_time": 245.5,
  "topics_mastered": ["Vectors", "Matrices"],
  "topics_struggling": ["Eigenvalues"],
  "retention_rate": 0.72,
  "engagement_score": 0.8,
  "learning_trajectory": [...],
  "predicted_next_performance": 0.75
}
"""

## GET /analytics/system
"""
Get system-wide analytics

Response:
{
  "total_students": 500,
  "total_interactions": 42000,
  "average_mastery": 0.65,
  "recommendation_accuracy": 0.78,
  "rl_avg_reward": 42.3,
  "retention_improvement": 0.12,
  "topics_per_student_avg": 15.4
}
"""

## GET /analytics/recommendations/quality
"""
Get recommendation system quality metrics

Response:
{
  "precision_at_5": 0.82,
  "recall_at_5": 0.74,
  "ndcg_at_5": 0.78,
  "precision_at_10": 0.78,
  "recall_at_10": 0.85,
  "ndcg_at_10": 0.81,
  "coverage": 0.92,
  "diversity": 0.65
}
"""

## GET /analytics/rl/rewards
"""
Get RL agent reward curves

Response:
{
  "episodes": 100,
  "avg_episode_reward": 42.3,
  "max_reward": 87.5,
  "min_reward": -12.3,
  "reward_trend": 0.15,
  "convergence_achieved": true,
  "episode_rewards": [10.2, 15.3, ...]
}
"""

# ==================== Health & System ====================

## GET /health
"""
Health check endpoint

Response:
{
  "status": "healthy",
  "version": "1.0.0-alpha",
  "timestamp": "2024-05-29T10:15:00Z",
  "services": {
    "database": true,
    "cache": true,
    "ml_models": true
  }
}
"""

## GET /metrics
"""
System performance metrics

Response:
{
  "timestamp": "2024-05-29T10:15:00Z",
  "version": "1.0.0-alpha",
  "request_count": 1000000,
  "average_response_time": 125,
  "error_rate": 0.001,
  "p99_latency": 500
}
"""

## GET /version
"""
Get version information

Response:
{
  "version": "1.0.0-alpha",
  "api_title": "Elo Learn API",
  "debug": false,
  "timestamp": "2024-05-29T10:15:00Z"
}
"""

---

# Error Responses

All endpoints can return error responses:

```json
{
  "detail": "Error message",
  "status_code": 400,
  "timestamp": "2024-05-29T10:15:00Z"
}
```

Common status codes:
- 200: Success
- 201: Created
- 400: Bad Request
- 401: Unauthorized
- 403: Forbidden
- 404: Not Found
- 500: Internal Server Error
