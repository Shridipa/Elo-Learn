"""Main Training Pipeline

Orchestrates training of recommendation systems and RL agents.
This script brings together all PHASE 2-3 components.
"""

import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from local modules
from datasets.generate_synthetic_interactions import generate_dataset, generate_concept_graph
from ml_models.feature_engineering import engineer_features
from recommendation_engine.baselines import create_all_baselines
from recommendation_engine.evaluation import OfflineEvaluator
from rl_engine.learning_environment import train_rl_agent

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training pipeline"""
    
    logger.info("="*70)
    logger.info("ELO LEARN - PHASE 2-3 TRAINING PIPELINE")
    logger.info("="*70)
    
    # ==================== PHASE 2: Data Preparation ====================
    
    logger.info("\n[PHASE 2] Generating Synthetic Dataset...")
    
    # Generate interactions
    interactions_df = generate_dataset(
        n_students=500,
        output_path="datasets/student_interactions.csv",
        random_seed=42
    )
    
    # Generate concept graph
    concept_graph = generate_concept_graph("datasets/concept_graph.json")
    
    # ==================== Feature Engineering ====================
    
    logger.info("\n[PHASE 2] Feature Engineering...")
    
    student_embeddings, topic_embeddings, _ = engineer_features(interactions_df)
    
    # Save embeddings
    embeddings_file = Path("datasets/embeddings.json")
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Note: NumPy arrays need to be converted to lists for JSON
    embeddings_data = {
        "student_embeddings": {
            str(k): v.tolist() for k, v in student_embeddings.items()
        },
        "topic_embeddings": {
            k: v.tolist() for k, v in topic_embeddings.items()
        }
    }
    
    with open(embeddings_file, 'w') as f:
        json.dump(embeddings_data, f)
    
    logger.info(f"✓ Embeddings saved to {embeddings_file}")
    
    # ==================== PHASE 3: Recommendation Systems ====================
    
    logger.info("\n[PHASE 3] Training Recommendation Baselines...")
    
    # Train all baselines
    baselines = create_all_baselines(interactions_df, topic_embeddings)
    
    # ==================== Evaluation ====================
    
    logger.info("\n[PHASE 3] Evaluating Recommendation Systems...")
    
    # Split data: 80% train, 20% test
    n_samples = len(interactions_df)
    train_size = int(0.8 * n_samples)
    test_interactions = interactions_df.iloc[train_size:]
    
    evaluator = OfflineEvaluator()
    
    results = {}
    for model_name, model in baselines.items():
        logger.info(f"\nEvaluating {model_name}...")
        
        try:
            metrics = evaluator.evaluate_recommender(
                model, test_interactions,
                top_k_values=[5, 10, 20]
            )
            results[model_name] = metrics
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {e}")
    
    # Save results
    results_file = Path("results/recommendation_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\n✓ Results saved to {results_file}")
    
    # ==================== PHASE 4: RL Agent Training ====================
    
    logger.info("\n[PHASE 4] Training RL Adaptive Tutor...")
    
    # Extract student profiles for RL
    student_profiles = []
    for student_id in interactions_df['student_id'].unique()[:10]:  # Use subset
        student_data = interactions_df[interactions_df['student_id'] == student_id]
        profile = {
            'base_ability': student_data['score'].mean(),
            'engagement': len(student_data) / 100,  # Normalize
            'learning_speed': 1.0,
        }
        student_profiles.append(profile)
    
    # Prepare concept graph for RL
    rl_concept_graph = {
        node['id']: {'difficulty': node.get('difficulty', 0.5)}
        for node in concept_graph.get('nodes', [])
        if isinstance(node, dict) and 'id' in node
    }
    
    # Fallback if JSON structure is different
    if not rl_concept_graph:
        rl_concept_graph = {
            'Easy': {'difficulty': 0.3},
            'Medium': {'difficulty': 0.5},
            'Hard': {'difficulty': 0.7},
            'Very Hard': {'difficulty': 0.9},
        }
    
    # Train RL agent
    rl_agent, episode_rewards = train_rl_agent(
        num_episodes=100,
        student_profiles=student_profiles,
        concept_graph=rl_concept_graph
    )
    
    # Save RL results
    rl_results_file = Path("results/rl_training.json")
    rl_results = {
        'episodes': len(episode_rewards),
        'avg_final_reward': float(np.mean(episode_rewards[-10:])),
        'max_reward': float(np.max(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'episode_rewards': episode_rewards
    }
    
    with open(rl_results_file, 'w') as f:
        json.dump(rl_results, f, indent=2)
    
    logger.info(f"✓ RL results saved to {rl_results_file}")
    
    # ==================== Summary ====================
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("="*70)
    
    logger.info(f"\n📊 Summary:")
    logger.info(f"  - Students: {interactions_df['student_id'].nunique()}")
    logger.info(f"  - Interactions: {len(interactions_df)}")
    logger.info(f"  - Topics: {interactions_df['topic'].nunique()}")
    logger.info(f"  - Success Rate: {interactions_df['is_correct'].mean():.2%}")
    
    logger.info(f"\n🤖 Recommendation Results (Top-3 metrics):")
    for model_name, metrics in results.items():
        logger.info(f"\n  {model_name}:")
        for metric, value in sorted(metrics.items())[:3]:
            logger.info(f"    {metric}: {value:.4f}")
    
    logger.info(f"\n🎯 RL Agent Results:")
    logger.info(f"  - Final Avg Reward: {rl_results['avg_final_reward']:.4f}")
    logger.info(f"  - Max Reward: {rl_results['max_reward']:.4f}")
    
    logger.info("\n✅ All PHASE 2-3 components trained successfully!")
    logger.info("Next: Integrate knowledge graph (PHASE 5)")

if __name__ == "__main__":
    main()
