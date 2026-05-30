"""Synthetic Student Interaction Dataset Generator

Generates realistic simulated student learning data for training and testing
the recommendation systems and RL agents.

This is PHASE 2 critical component - creates the backbone dataset
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConceptHierarchy:
    """Define the concept dependency graph"""
    
    CONCEPTS = {
        # Linear Algebra
        "Linear Algebra Basics": {
            "difficulty": 0.4,
            "prerequisites": [],
            "category": "Mathematics"
        },
        "Vectors": {
            "difficulty": 0.5,
            "prerequisites": ["Linear Algebra Basics"],
            "category": "Mathematics"
        },
        "Matrices": {
            "difficulty": 0.6,
            "prerequisites": ["Linear Algebra Basics", "Vectors"],
            "category": "Mathematics"
        },
        "Matrix Operations": {
            "difficulty": 0.65,
            "prerequisites": ["Matrices"],
            "category": "Mathematics"
        },
        "Eigenvalues and Eigenvectors": {
            "difficulty": 0.7,
            "prerequisites": ["Matrices", "Matrix Operations"],
            "category": "Mathematics"
        },
        
        # Neural Networks
        "Neural Network Basics": {
            "difficulty": 0.5,
            "prerequisites": ["Linear Algebra Basics"],
            "category": "Deep Learning"
        },
        "Activation Functions": {
            "difficulty": 0.6,
            "prerequisites": ["Neural Network Basics"],
            "category": "Deep Learning"
        },
        "Backpropagation": {
            "difficulty": 0.7,
            "prerequisites": ["Activation Functions", "Matrices"],
            "category": "Deep Learning"
        },
        
        # Transformers
        "Attention Mechanisms": {
            "difficulty": 0.7,
            "prerequisites": ["Neural Network Basics", "Matrices"],
            "category": "Deep Learning"
        },
        "Transformer Architecture": {
            "difficulty": 0.8,
            "prerequisites": ["Attention Mechanisms", "Backpropagation"],
            "category": "Deep Learning"
        },
        
        # Python Basics
        "Python Fundamentals": {
            "difficulty": 0.3,
            "prerequisites": [],
            "category": "Programming"
        },
        "Python Data Structures": {
            "difficulty": 0.5,
            "prerequisites": ["Python Fundamentals"],
            "category": "Programming"
        },
        "NumPy Basics": {
            "difficulty": 0.5,
            "prerequisites": ["Python Fundamentals", "Linear Algebra Basics"],
            "category": "Programming"
        },
        "Pandas": {
            "difficulty": 0.6,
            "prerequisites": ["NumPy Basics", "Python Data Structures"],
            "category": "Programming"
        },
    }

class StudentProfileGenerator:
    """Generate realistic student learning profiles"""
    
    LEARNING_STYLES = [
        "visual",
        "auditory",
        "reading/writing",
        "kinesthetic"
    ]
    
    PROFICIENCY_LEVELS = ["beginner", "intermediate", "advanced"]
    
    @staticmethod
    def generate_student_profile(student_id: int) -> Dict:
        """Generate a realistic student profile"""
        
        proficiency = np.random.choice(StudentProfileGenerator.PROFICIENCY_LEVELS,
                                      p=[0.4, 0.4, 0.2])
        learning_style = np.random.choice(StudentProfileGenerator.LEARNING_STYLES)
        
        # Profile characteristics
        if proficiency == "beginner":
            base_ability = np.random.uniform(0.2, 0.5)
            learning_speed = np.random.uniform(0.5, 1.0)
            retention = np.random.uniform(0.4, 0.6)
        elif proficiency == "intermediate":
            base_ability = np.random.uniform(0.5, 0.75)
            learning_speed = np.random.uniform(0.8, 1.2)
            retention = np.random.uniform(0.6, 0.8)
        else:  # advanced
            base_ability = np.random.uniform(0.7, 1.0)
            learning_speed = np.random.uniform(1.0, 1.5)
            retention = np.random.uniform(0.8, 0.95)
        
        return {
            "student_id": student_id,
            "proficiency": proficiency,
            "learning_style": learning_style,
            "base_ability": base_ability,
            "learning_speed": learning_speed,
            "retention": retention,
            "engagement": np.random.uniform(0.5, 1.0),
            "preferred_difficulty": base_ability + np.random.normal(0, 0.1),
        }

class InteractionSimulator:
    """Simulate realistic student-concept interactions"""
    
    def __init__(self, concepts: Dict, max_interactions_per_student: int = 100):
        self.concepts = concepts
        self.max_interactions_per_student = max_interactions_per_student
    
    def simulate_interaction(self, student_profile: Dict, topic: str,
                           student_history: Dict[str, List[bool]]) -> Dict:
        """Simulate a single student-topic interaction"""
        
        # Calculate student ability on this topic
        ability = student_profile["base_ability"]
        
        # Adjust based on prerequisites mastery
        prerequisites = self.concepts[topic]["prerequisites"]
        if prerequisites:
            prerequisite_mastery = [
                np.mean(student_history.get(prereq, [0.5]))
                for prereq in prerequisites
            ]
            ability *= np.mean(prerequisite_mastery)
        
        # Topic difficulty
        topic_difficulty = self.concepts[topic]["difficulty"]
        
        # Calculate performance probability (using IRT-like model)
        discrimination = 1.7
        success_prob = 1 / (1 + np.exp(-discrimination * (ability - topic_difficulty)))
        
        # Add randomness and learning effect
        success_prob += np.random.normal(0, 0.05)
        success_prob = np.clip(success_prob, 0, 1)
        
        is_correct = np.random.random() < success_prob
        score = success_prob if is_correct else success_prob * 0.5 + np.random.uniform(0, 0.3)
        score = np.clip(score, 0, 1)
        
        # Time estimation (exponential decay with practice)
        attempts = len(student_history.get(topic, []))
        base_time = 300 + (topic_difficulty * 200)  # seconds
        time_spent = base_time / (1 + 0.5 * attempts * student_profile["learning_speed"])
        time_spent = np.clip(time_spent, 30, 1200)
        
        # Difficulty presented (adaptive)
        difficulty_presented = topic_difficulty
        recent_performance = student_history.get(topic, [0.5])[-3:]
        avg_recent = np.mean(recent_performance)
        if avg_recent > 0.8:
            difficulty_presented = min(1.0, difficulty_presented + 0.1)
        elif avg_recent < 0.4:
            difficulty_presented = max(0.0, difficulty_presented - 0.1)
        
        return {
            "topic": topic,
            "is_correct": is_correct,
            "score": score,
            "time_spent": time_spent,
            "attempts": 1,
            "difficulty_presented": difficulty_presented,
            "success_prob": success_prob,
        }

def generate_dataset(n_students: int = 500, 
                    output_path: str = "datasets/student_interactions.csv",
                    random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic student interaction dataset
    
    Args:
        n_students: Number of students to simulate
        output_path: Where to save the CSV
        random_seed: For reproducibility
    
    Returns:
        DataFrame with simulated interactions
    """
    
    np.random.seed(random_seed)
    logger.info(f"Generating synthetic data for {n_students} students...")
    
    concepts = ConceptHierarchy.CONCEPTS
    simulator = InteractionSimulator(concepts)
    
    data = []
    timestamp = datetime.now() - timedelta(days=365)  # Start 1 year ago
    
    for student_id in range(1, n_students + 1):
        if student_id % 50 == 0:
            logger.info(f"  Processing student {student_id}/{n_students}...")
        
        # Generate student profile
        profile = StudentProfileGenerator.generate_student_profile(student_id)
        
        # Track learning history
        student_history = {}
        
        # Simulate interactions
        n_interactions = np.random.randint(
            int(simulator.max_interactions_per_student * 0.5),
            simulator.max_interactions_per_student
        )
        
        for interaction_num in range(n_interactions):
            # Select topic (prefer accessible topics, gradually increase difficulty)
            topics_list = list(concepts.keys())
            # Weight topics by how well-suited they are
            weights = []
            for topic in topics_list:
                topic_diff = concepts[topic]["difficulty"]
                student_pref_diff = np.clip(
                    profile["base_ability"] + np.random.normal(0, 0.15),
                    0, 1
                )
                distance = abs(topic_diff - student_pref_diff)
                weight = 1 / (1 + distance * 2)  # Prefer closer difficulty
                
                # Prefer topics with satisfied prerequisites
                prerequisites = concepts[topic]["prerequisites"]
                if prerequisites:
                    mastered_prereqs = sum(
                        1 for p in prerequisites
                        if np.mean(student_history.get(p, [0])) > 0.7
                    )
                    weight *= (1 + mastered_prereqs * 0.3)
                
                weights.append(weight)
            
            weights = np.array(weights) / np.sum(weights)
            topic = np.random.choice(topics_list, p=weights)
            
            # Simulate interaction
            interaction = simulator.simulate_interaction(profile, topic, student_history)
            
            # Track history
            if topic not in student_history:
                student_history[topic] = []
            student_history[topic].append(interaction["score"])
            
            # Add timestamp
            timestamp += timedelta(hours=np.random.uniform(1, 72))  # 1-3 days between interactions
            session_id = f"session_{student_id}_{interaction_num // 10}"
            
            data.append({
                "student_id": student_id,
                "topic": topic,
                "is_correct": interaction["is_correct"],
                "score": interaction["score"],
                "time_spent": interaction["time_spent"],
                "difficulty_presented": interaction["difficulty_presented"],
                "timestamp": timestamp,
                "session_id": session_id,
                "is_review": False,  # Will be set later for spaced repetition
                "proficiency_level": profile["proficiency"],
                "base_ability": profile["base_ability"],
                "learning_speed": profile["learning_speed"],
                "retention_rate": profile["retention"],
            })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    
    logger.info(f"✓ Generated {len(df)} interactions")
    logger.info(f"✓ Saved to {output_file}")
    logger.info(f"✓ Data shape: {df.shape}")
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  - Students: {df['student_id'].nunique()}")
    logger.info(f"  - Topics: {df['topic'].nunique()}")
    logger.info(f"  - Avg interactions/student: {len(df) / df['student_id'].nunique():.1f}")
    logger.info(f"  - Overall success rate: {df['is_correct'].mean():.2%}")
    logger.info(f"  - Avg score: {df['score'].mean():.3f}")
    logger.info(f"  - Avg time spent: {df['time_spent'].mean():.0f}s")
    
    return df

def generate_concept_graph(output_path: str = "datasets/concept_graph.json"):
    """Generate and save concept dependency graph"""
    
    concepts = ConceptHierarchy.CONCEPTS
    
    graph = {
        "nodes": [
            {
                "id": concept,
                "difficulty": props["difficulty"],
                "category": props["category"]
            }
            for concept, props in concepts.items()
        ],
        "edges": []
    }
    
    # Add prerequisite edges
    for concept, props in concepts.items():
        for prereq in props["prerequisites"]:
            graph["edges"].append({
                "source": prereq,
                "target": concept,
                "type": "prerequisite"
            })
    
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, "w") as f:
        json.dump(graph, f, indent=2)
    
    logger.info(f"✓ Saved concept graph to {output_file}")
    return graph

if __name__ == "__main__":
    # Generate datasets
    df = generate_dataset(
        n_students=500,
        output_path="datasets/student_interactions.csv",
        random_seed=42
    )
    
    # Generate concept graph
    graph = generate_concept_graph("datasets/concept_graph.json")
    
    logger.info("\n" + "="*60)
    logger.info("PHASE 2 Dataset Generation Complete!")
    logger.info("="*60)
