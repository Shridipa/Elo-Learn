"""Student Embedding Engine

Compute dense student representations from interaction history.
Produces rich learner vectors, research-grade cluster metrics, and similarity search.
"""
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class StudentEmbeddingEngine:
    def __init__(self,
                 embedding_dim: int = 32,
                 results_path: Path = Path("datasets"),
                 concept_graph_path: Optional[Path] = None):
        self.embedding_dim = embedding_dim
        self.results_path = results_path
        self.concept_graph_path = concept_graph_path or self.results_path / "concept_graph.json"
        self.topic_categories = self._load_topic_categories()
        self.topic_difficulties = self._load_topic_difficulties()

    def _load_topic_categories(self) -> Dict[str, str]:
        if not self.concept_graph_path.exists():
            return {}
        try:
            with open(self.concept_graph_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            return {node.get('id'): node.get('category', 'unknown') for node in graph.get('nodes', [])}
        except Exception:
            return {}

    def _load_topic_difficulties(self) -> Dict[str, float]:
        if not self.concept_graph_path.exists():
            return {}
        try:
            with open(self.concept_graph_path, 'r', encoding='utf-8') as f:
                graph = json.load(f)
            return {node.get('id'): float(node.get('difficulty', 0.5)) for node in graph.get('nodes', [])}
        except Exception:
            return {}

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if 'is_correct' not in df.columns:
            df['is_correct'] = df['score'] >= 0.7
        if 'time_spent' not in df.columns:
            df['time_spent'] = np.nan
        if 'difficulty_presented' not in df.columns:
            df['difficulty_presented'] = df['topic'].map(self.topic_difficulties).fillna(0.5)
        if 'session_id' not in df.columns:
            df['session_id'] = df['student_id'].astype(str)
        if 'is_review' not in df.columns:
            df['is_review'] = False
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        else:
            df['timestamp'] = pd.NaT

        df['success'] = df['is_correct'].astype(float)
        df['topic_difficulty'] = df['topic'].map(self.topic_difficulties).fillna(0.5)
        return df

    def _aggregate_student_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df = self._ensure_columns(df.copy())
        total_topics = df['topic'].nunique() if 'topic' in df.columns else 0

        student = df.groupby('student_id').agg(
            total_interactions=('topic', 'count'),
            unique_topics=('topic', 'nunique'),
            mean_score=('score', 'mean'),
            success_rate=('success', 'mean'),
            score_std=('score', 'std'),
            avg_time_spent=('time_spent', 'mean'),
            avg_difficulty_presented=('difficulty_presented', 'mean'),
            avg_topic_difficulty=('topic_difficulty', 'mean'),
            sessions=('session_id', 'nunique'),
            review_count=('is_review', 'sum'),
            total_reviews=('is_review', 'count'),
            num_topics_revisited=('topic', lambda x: (x.value_counts() > 1).sum())
        )

        student['total_interactions'] = student['total_interactions'].fillna(0).astype(int)
        student['unique_topics'] = student['unique_topics'].fillna(0).astype(int)
        student['score_std'] = student['score_std'].fillna(0.0)
        student['avg_time_spent'] = student['avg_time_spent'].fillna(0.0)
        student['avg_difficulty_presented'] = student['avg_difficulty_presented'].fillna(0.5)
        student['avg_topic_difficulty'] = student['avg_topic_difficulty'].fillna(0.5)
        student['sessions'] = student['sessions'].fillna(1).astype(int)
        student['review_ratio'] = student['review_count'] / student['total_interactions'].replace(0, 1)
        student['revision_frequency'] = student['num_topics_revisited'] / student['total_interactions'].replace(0, 1)
        student['attempts_per_topic'] = student['total_interactions'] / student['unique_topics'].replace(0, 1)
        student['topic_completion'] = student['unique_topics'] / max(total_topics, 1)
        student['difficulty_gap'] = student['avg_difficulty_presented'] - student['avg_topic_difficulty']
        student['engagement_score'] = student['total_interactions'] / (student['total_interactions'].max() + 1e-6)

        if df['timestamp'].notna().any():
            sorted_df = df.sort_values(['student_id', 'timestamp'])
            intervals = sorted_df.groupby('student_id')['timestamp'].diff().dt.total_seconds() / 86400.0
            interval_avg = intervals.groupby(sorted_df['student_id']).mean()
            days_active = sorted_df.groupby('student_id')['timestamp'].agg(lambda x: (x.max() - x.min()).days if len(x) > 1 else 0)
            student['avg_interaction_interval'] = interval_avg
            student['days_active'] = days_active
        else:
            student['avg_interaction_interval'] = 0.0
            student['days_active'] = 0.0

        student['avg_interaction_interval'] = student['avg_interaction_interval'].fillna(0.0)
        student['days_active'] = student['days_active'].fillna(0.0)
        student['retention_score'] = np.clip(1.0 / (1.0 + student['avg_interaction_interval']), 0.0, 1.0)

        # Category mastery features from concept graph
        category_features = {}
        if self.topic_categories:
            category_scores = (
                df.groupby(['student_id', df['topic'].map(self.topic_categories)])['score']
                .mean()
                .unstack(fill_value=np.nan)
            )
            category_features = {
                f'category_{col}': category_scores[col].fillna(0.0)
                for col in category_scores.columns
            }
        if category_features:
            category_df = pd.DataFrame(category_features)
            student = student.join(category_df)

        student = student.fillna(0.0)
        student = student.replace([np.inf, -np.inf], 0.0)

        return student, df

    def _build_feature_groups(self, features_df: pd.DataFrame) -> Dict[str, List[str]]:
        mastery_cols = ['mean_score', 'success_rate', 'score_std', 'topic_completion']
        engagement_cols = ['total_interactions', 'sessions', 'engagement_score', 'avg_time_spent', 'attempts_per_topic']
        retention_cols = ['revision_frequency', 'review_ratio', 'num_topics_revisited', 'retention_score', 'avg_interaction_interval', 'days_active']
        difficulty_cols = ['avg_difficulty_presented', 'difficulty_gap', 'avg_topic_difficulty', 'score_std']

        category_cols = [c for c in features_df.columns if c.startswith('category_')]
        mastery_cols += category_cols

        groups = {
            'mastery': [c for c in mastery_cols if c in features_df.columns],
            'engagement': [c for c in engagement_cols if c in features_df.columns],
            'retention': [c for c in retention_cols if c in features_df.columns],
            'difficulty': [c for c in difficulty_cols if c in features_df.columns],
        }
        return groups

    def _allocate_group_dims(self) -> Dict[str, int]:
        base = self.embedding_dim // 4
        remainder = self.embedding_dim % 4
        dims = {
            'mastery': base,
            'engagement': base,
            'retention': base,
            'difficulty': base,
        }
        for i, group in enumerate(['mastery', 'engagement', 'retention', 'difficulty']):
            if i < remainder:
                dims[group] += 1
        return dims

    def _reduce_group(self, X: np.ndarray, dim: int) -> np.ndarray:
        if X.shape[1] == 0:
            return np.zeros((X.shape[0], dim), dtype=np.float32)
        # Choose n_components safely: must be between 1 and min(n_samples, n_features)
        n_samples, n_features = X.shape
        if dim > 0:
            n_components = max(1, min(dim, n_samples, n_features))
        else:
            n_components = 0

        if n_components > 0 and n_components <= min(n_samples, n_features):
            try:
                pca = PCA(n_components=n_components, random_state=42)
                reduced = pca.fit_transform(X)
                # If requested dim larger than reduced, pad
                if reduced.shape[1] < dim:
                    pad = np.zeros((reduced.shape[0], dim - reduced.shape[1]), dtype=np.float32)
                    return np.hstack([reduced.astype(np.float32), pad])
                return reduced.astype(np.float32)
            except Exception:
                # fallback to scaled approach
                pass
        scaled = StandardScaler().fit_transform(X)
        if scaled.shape[1] >= dim and dim > 0:
            n_samples, n_features = scaled.shape
            n_components = max(1, min(dim, n_samples, n_features))
            try:
                pca = PCA(n_components=n_components, random_state=42)
                reduced = pca.fit_transform(scaled)
                if reduced.shape[1] < dim:
                    pad = np.zeros((reduced.shape[0], dim - reduced.shape[1]), dtype=np.float32)
                    return np.hstack([reduced.astype(np.float32), pad])
                return reduced.astype(np.float32)
            except Exception:
                pass
        pad = np.zeros((scaled.shape[0], max(0, dim - scaled.shape[1])), dtype=np.float32)
        return np.hstack([scaled.astype(np.float32), pad])

    def _compute_embedding_matrix(self, features_df: pd.DataFrame, embedding_dim: int) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        groups = self._build_feature_groups(features_df)
        group_dims = self._allocate_group_dims() if embedding_dim == self.embedding_dim else self._allocate_group_dims_for_dim(embedding_dim)
        vectors = []
        group_projection = {}
        for group_name, cols in groups.items():
            dim = group_dims[group_name]
            X = features_df[cols].astype(float).fillna(0.0).values
            vec = self._reduce_group(X, dim)
            vectors.append(vec)
            group_projection[group_name] = vec
        X_embed = np.concatenate(vectors, axis=1)
        if X_embed.shape[1] != embedding_dim:
            if X_embed.shape[1] > embedding_dim:
                X_embed = X_embed[:, :embedding_dim]
            else:
                pad = np.zeros((X_embed.shape[0], embedding_dim - X_embed.shape[1]), dtype=np.float32)
                X_embed = np.hstack([X_embed, pad])
        return X_embed, group_projection

    def _allocate_group_dims_for_dim(self, embedding_dim: int) -> Dict[str, int]:
        base = embedding_dim // 4
        remainder = embedding_dim % 4
        dims = {'mastery': base, 'engagement': base, 'retention': base, 'difficulty': base}
        for i, group in enumerate(['mastery', 'engagement', 'retention', 'difficulty']):
            if i < remainder:
                dims[group] += 1
        return dims

    def _cluster_purity(self, labels: np.ndarray, features_df: pd.DataFrame) -> float:
        category_cols = [c for c in features_df.columns if c.startswith('category_')]
        if not category_cols:
            return 0.0
        category_labels = features_df[category_cols].idxmax(axis=1)
        df = pd.DataFrame({'cluster': labels, 'category': category_labels})
        purity = df.groupby('cluster').apply(lambda group: group['category'].value_counts(normalize=True).max()).mean()
        return float(purity)

    def _neighbor_similarity_score(self, X: np.ndarray, top_k: int = 5) -> float:
        if X.shape[0] <= 1:
            return 0.0
        nn = NearestNeighbors(n_neighbors=min(top_k + 1, X.shape[0]), algorithm='auto').fit(X)
        distances, _ = nn.kneighbors(X)
        if distances.shape[1] <= 1:
            return 0.0
        score = np.mean(1.0 / (1.0 + distances[:, 1:]))
        return float(score)

    def _experiment_metrics(self, features_df: pd.DataFrame, dims: List[int]) -> Dict[str, Any]:
        metrics = {}
        for dim in dims:
            X_embed, _ = self._compute_embedding_matrix(features_df, dim)
            n_samples = X_embed.shape[0]
            n_clusters = min(8, max(2, int(np.sqrt(n_samples))))
            labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_embed)
            sil = silhouette_score(X_embed, labels) if len(set(labels)) > 1 and n_samples > 1 else 0.0
            purity = self._cluster_purity(labels, features_df)
            neighbor_score = self._neighbor_similarity_score(X_embed)
            metrics[str(dim)] = {
                'dimension': dim,
                'cluster_count': int(n_clusters),
                'silhouette_score': float(sil),
                'cluster_purity': float(purity),
                'neighbor_similarity_score': float(neighbor_score)
            }
        return metrics

    def fit_transform(self, interactions_df: pd.DataFrame) -> Tuple[Dict[int, List[float]], Dict[str, Any]]:
        features_df, df = self._aggregate_student_features(interactions_df)
        X_embed, group_projection = self._compute_embedding_matrix(features_df, self.embedding_dim)

        k = min(8, max(2, int(np.sqrt(X_embed.shape[0]))))
        kmeans = KMeans(n_clusters=k, random_state=42).fit(X_embed)
        labels = kmeans.labels_
        nn = NearestNeighbors(n_neighbors=min(10, X_embed.shape[0]), algorithm='auto').fit(X_embed)

        student_ids = list(features_df.index.astype(int))
        embeddings = {int(sid): X_embed[i].tolist() for i, sid in enumerate(student_ids)}

        coords_pca = None
        coords_tsne = None
        try:
            viz_pca = PCA(n_components=2, random_state=42).fit_transform(X_embed)
            coords_pca = {int(sid): [float(viz_pca[i, 0]), float(viz_pca[i, 1])] for i, sid in enumerate(student_ids)}
        except Exception:
            coords_pca = {int(sid): [0.0, 0.0] for sid in student_ids}

        try:
            tsne = TSNE(n_components=2, perplexity=min(30, max(5, X_embed.shape[0] // 3)), random_state=42, init='pca')
            viz_tsne = tsne.fit_transform(X_embed)
            coords_tsne = {int(sid): [float(viz_tsne[i, 0]), float(viz_tsne[i, 1])] for i, sid in enumerate(student_ids)}
        except Exception:
            coords_tsne = None

        experiments = self._experiment_metrics(features_df, [16, 32, 64])

        features_mapping = {}
        for sid, row in features_df.iterrows():
            features_mapping[int(sid)] = {col: _safe_float(row[col]) for col in features_df.columns}

        results = {
            'student_ids': student_ids,
            'embeddings': embeddings,
            'cluster_labels': labels.tolist(),
            'kmeans_inertia': float(kmeans.inertia_),
            'coords_pca': coords_pca,
            'coords_tsne': coords_tsne,
            'features': features_mapping,
            'experiment_metrics': experiments,
            'cluster_metrics': {
                'silhouette_score': float(silhouette_score(X_embed, labels)) if len(set(labels)) > 1 else 0.0,
                'cluster_purity': self._cluster_purity(labels, features_df),
                'neighbor_similarity_score': self._neighbor_similarity_score(X_embed),
                'cluster_count': int(k)
            }
        }

        try:
            out_file = self.results_path / 'student_embeddings.json'
            out_file.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file, 'w', encoding='utf-8') as f:
                json.dump(results, f)
            logger.info(f"Saved student embeddings to {out_file}")
        except Exception as exc:
            logger.warning(f"Could not save embeddings: {exc}")

        metadata = {
            'features_df': features_df,
            'kmeans': kmeans,
            'nn': nn,
            'group_projection': group_projection
        }

        return embeddings, metadata

    def nearest_neighbors(self, embeddings: Dict[int, List[float]], student_id: int, n: int = 5) -> List[int]:
        ids = list(embeddings.keys())
        X = np.array([embeddings[i] for i in ids])
        if student_id not in ids:
            raise ValueError('student_id not found')
        idx = ids.index(student_id)
        nbrs = NearestNeighbors(n_neighbors=min(n + 1, X.shape[0]), algorithm='auto').fit(X)
        distances, indices = nbrs.kneighbors([X[idx]])
        neighbors = [ids[i] for i in indices[0] if ids[i] != student_id]
        return neighbors

    def profile_student(self, student_id: int, interactions_df: pd.DataFrame) -> Dict[str, Any]:
        features_df, _ = self._aggregate_student_features(interactions_df)
        if student_id not in features_df.index:
            raise ValueError('student_id not found')
        return {col: _safe_float(features_df.loc[student_id, col]) for col in features_df.columns}


def quick_run_from_csv(csv_path: str = 'datasets/student_interactions.csv') -> Dict:
    df = pd.read_csv(csv_path)
    engine = StudentEmbeddingEngine(embedding_dim=32, results_path=Path('datasets'))
    embeddings, metadata = engine.fit_transform(df)
    return {'n_students': len(embeddings), 'example_student': next(iter(embeddings.keys()))}


if __name__ == '__main__':
    res = quick_run_from_csv()
    print(res)
