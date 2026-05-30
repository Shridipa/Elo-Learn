import pandas as pd
from pathlib import Path
from typing import Dict, Any, List


def _load_interactions(interactions_path: Path) -> pd.DataFrame:
    if not interactions_path.exists():
        raise FileNotFoundError(f"Interactions CSV not found: {interactions_path}")
    df = pd.read_csv(interactions_path)
    # Expect columns: student_id, topic, score
    return df


def compute_cohort_overview(interactions_path: Path, weak_threshold: float = 0.65) -> Dict[str, Any]:
    df = _load_interactions(interactions_path)
    # Per-student average mastery
    student_avg = df.groupby('student_id')['score'].mean()
    overall_avg_mastery = float(student_avg.mean()) if not student_avg.empty else 0.0

    # Per-topic averages and weak concept detection
    topic_stats = df.groupby('topic')['score'].agg(['mean', 'count']).reset_index()
    topic_stats = topic_stats.rename(columns={'mean': 'average_score', 'count': 'interactions'})
    # Weak concepts: topics with average score below threshold
    weak_concepts = topic_stats[topic_stats['average_score'] < weak_threshold].sort_values('average_score').to_dict(orient='records')

    # Engagement: interactions per student normalized
    interactions_per_student = df.groupby('student_id').size()
    avg_interactions = float(interactions_per_student.mean()) if not interactions_per_student.empty else 0.0
    max_interactions = int(interactions_per_student.max()) if not interactions_per_student.empty else 0
    engagement_score = float(avg_interactions / max_interactions) if max_interactions > 0 else 0.0

    # Completion percentage: percent of students with avg mastery >= 0.75
    completion_threshold = 0.75
    completed = (student_avg >= completion_threshold).sum()
    total_students = len(student_avg)
    completion_pct = float(completed) / total_students if total_students > 0 else 0.0

    # Student mastery list for distribution visualizations
    student_masteries = []
    if not student_avg.empty:
        # convert to plain python floats
        student_masteries = [float(x) for x in student_avg.tolist()]

    # Mastery bins (counts) for simple histograms
    bins = [0.0, 0.4, 0.6, 0.75, 1.0]
    mastery_bins = {}
    if student_masteries:
        cat = pd.cut(student_masteries, bins=bins, include_lowest=True)
        counts = cat.value_counts().sort_index()
        mastery_bins = {str(interval): int(count) for interval, count in counts.items()}

    # Top weak topics (names only)
    top_weak_topics = [t.get('topic') if isinstance(t, dict) and 'topic' in t else t for t in weak_concepts][:10]

    return {
        'student_count': total_students,
        'average_mastery': overall_avg_mastery,
        'avg_interactions_per_student': avg_interactions,
        'engagement_score': engagement_score,
        'completion_pct': completion_pct,
        'weak_concepts': weak_concepts,
        'top_weak_topics': top_weak_topics,
        'topic_stats': topic_stats.to_dict(orient='records'),
        'student_masteries': student_masteries,
        'mastery_bins': mastery_bins,
    }



def detect_at_risk_students(interactions_path: Path, mastery_threshold: float = 0.6, top_k: int = 50) -> List[Dict[str, Any]]:
    df = _load_interactions(interactions_path)
    student_topic = df.groupby(['student_id', 'topic'])['score'].mean().reset_index()
    # Student average mastery
    student_avg = student_topic.groupby('student_id')['score'].mean()
    # Weak topic count per student
    weak_counts = student_topic[student_topic['score'] < mastery_threshold].groupby('student_id').size()

    rows = []
    for sid, avg in student_avg.items():
        weak = int(weak_counts.get(sid, 0))
        rows.append({'student_id': int(sid), 'average_mastery': float(avg), 'weak_topic_count': weak})

    # Sort by weakest (low avg mastery then many weak topics)
    rows_sorted = sorted(rows, key=lambda r: (r['average_mastery'], -r['weak_topic_count']))
    return rows_sorted[:top_k]
