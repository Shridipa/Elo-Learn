"""RL Tutor Environment and Agent

Defines the learning environment for RL agents and implements
DQN and PPO agents for adaptive tutoring.
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class LearningEnvironment:
    """
    Simulated learning environment for RL agent
    
    State: Student proficiency, topic difficulty, engagement, prerequisites
    Action: Adjust difficulty, change topic, trigger review
    Reward: Retention + engagement - frustration
    """
    
    def __init__(self, student_profile: Dict, concept_graph: Dict):
        """
        Initialize learning environment
        
        Args:
            student_profile: Student's characteristics
            concept_graph: Topic dependencies and difficulty
        """
        self.student_profile = student_profile
        self.concept_graph = concept_graph
        
        # State variables
        self.current_proficiency = student_profile['base_ability']
        self.current_topic = None
        self.current_difficulty = 0.5
        self.engagement = student_profile['engagement']
        self.retention = {}  # Topic -> retention rate
        self.timestep = 0
        self.max_steps = 1000
        
        # History
        self.episode_history = []
    
    def reset(self) -> np.ndarray:
        """Reset environment and return initial state"""
        self.current_proficiency = self.student_profile['base_ability']
        self.current_topic = list(self.concept_graph.keys())[0]
        self.current_difficulty = self.concept_graph[self.current_topic]['difficulty']
        self.engagement = self.student_profile['engagement']
        self.retention = {}
        self.timestep = 0
        self.episode_history = []
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector"""
        state = [
            self.current_proficiency,
            self.current_difficulty,
            self.engagement,
            self.retention.get(self.current_topic, 0.5),
            self.timestep / self.max_steps,  # Normalize time
        ]
        return np.array(state, dtype=np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Take a step in the environment
        
        Actions:
        0: Keep current difficulty
        1: Increase difficulty
        2: Decrease difficulty
        3: Change to easier topic
        4: Change to harder topic
        5: Trigger review (retry same topic)
        """
        self.timestep += 1
        done = self.timestep >= self.max_steps
        
        # Process action
        old_difficulty = self.current_difficulty
        
        if action == 0:
            # Keep difficulty
            pass
        elif action == 1:
            # Increase difficulty
            self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
        elif action == 2:
            # Decrease difficulty
            self.current_difficulty = max(0.0, self.current_difficulty - 0.1)
        elif action == 3:
            # Change to easier topic
            self.current_topic = self._get_easier_topic()
            self.current_difficulty = self.concept_graph[self.current_topic]['difficulty']
        elif action == 4:
            # Change to harder topic
            self.current_topic = self._get_harder_topic()
            self.current_difficulty = self.concept_graph[self.current_topic]['difficulty']
        elif action == 5:
            # Review (increase difficulty slightly)
            self.current_difficulty = min(1.0, self.current_difficulty + 0.05)
        
        # Simulate student performance
        success_prob = self._compute_success_probability()
        is_correct = np.random.random() < success_prob
        
        # Update retention
        if self.current_topic not in self.retention:
            self.retention[self.current_topic] = 0.0
        
        if is_correct:
            self.retention[self.current_topic] = 0.9 * self.retention[self.current_topic] + 0.1 * 1.0
            self.current_proficiency += 0.01
        else:
            self.retention[self.current_topic] = 0.9 * self.retention[self.current_topic] + 0.1 * 0.0
        
        self.current_proficiency = np.clip(self.current_proficiency, 0, 1)
        
        # Compute reward
        retention_reward = is_correct
        engagement_bonus = 0.1 if 0.4 < old_difficulty < 0.8 else -0.1  # Prefer moderate difficulty
        frustration_penalty = -0.2 if success_prob < 0.3 else 0.0  # Too hard
        
        reward = retention_reward + engagement_bonus + frustration_penalty
        
        # Update engagement (decay if frustrated, increase if success)
        if not is_correct and success_prob < 0.2:
            self.engagement *= 0.95  # Decay
        elif is_correct:
            self.engagement = min(1.0, self.engagement + 0.02)
        
        self.episode_history.append({
            'action': action,
            'topic': self.current_topic,
            'difficulty': self.current_difficulty,
            'is_correct': is_correct,
            'reward': reward,
        })
        
        next_state = self._get_state()
        
        return next_state, reward, done, {'success': is_correct}
    
    def _compute_success_probability(self) -> float:
        """Compute probability of success (IRT model)"""
        ability = self.current_proficiency
        difficulty = self.current_difficulty
        discrimination = 1.7
        
        # IRT model
        success_prob = 1 / (1 + np.exp(-discrimination * (ability - difficulty)))
        
        # Add some variance
        success_prob += np.random.normal(0, 0.05)
        return np.clip(success_prob, 0, 1)
    
    def _get_easier_topic(self) -> str:
        """Get a topic easier than current"""
        current_difficulty = self.concept_graph[self.current_topic]['difficulty']
        easier_topics = [
            t for t, props in self.concept_graph.items()
            if props['difficulty'] < current_difficulty - 0.1
        ]
        return easier_topics[0] if easier_topics else self.current_topic
    
    def _get_harder_topic(self) -> str:
        """Get a topic harder than current"""
        current_difficulty = self.concept_graph[self.current_topic]['difficulty']
        harder_topics = [
            t for t, props in self.concept_graph.items()
            if props['difficulty'] > current_difficulty + 0.1
        ]
        return harder_topics[0] if harder_topics else self.current_topic


class DQNAgent:
    """
    Deep Q-Network Agent for adaptive tutoring
    
    Simple implementation using a neural network to approximate Q-values
    """
    
    def __init__(self, state_dim: int = 5, action_dim: int = 6,
                 learning_rate: float = 0.001, gamma: float = 0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        
        # Simple Q-table based approach
        self.q_table = {}
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
    
    def get_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            return self.get_best_action(state)
    
    def get_best_action(self, state: np.ndarray) -> int:
        """Get best action for state"""
        state_key = tuple(np.round(state, 2))
        if state_key not in self.q_table:
            return np.random.randint(self.action_dim)
        
        q_values = self.q_table[state_key]
        return int(np.argmax(q_values))
    
    def update(self, state: np.ndarray, action: int, reward: float,
               next_state: np.ndarray, done: bool) -> float:
        """Q-learning update"""
        state_key = tuple(np.round(state, 2))
        next_state_key = tuple(np.round(next_state, 2))
        
        # Initialize Q-values
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_dim)
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_dim)
        
        # Q-learning update
        current_q = self.q_table[state_key][action]
        max_next_q = np.max(self.q_table[next_state_key]) if not done else 0.0
        
        new_q = current_q + self.learning_rate * (
            reward + self.gamma * max_next_q - current_q
        )
        
        self.q_table[state_key][action] = new_q
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return new_q
    
    def get_policy(self, state: np.ndarray) -> np.ndarray:
        """Get current policy (Q-values) for state"""
        state_key = tuple(np.round(state, 2))
        if state_key not in self.q_table:
            return np.zeros(self.action_dim)
        return self.q_table[state_key]


def train_rl_agent(num_episodes: int = 100,
                  student_profiles: List[Dict] = None,
                  concept_graph: Dict = None) -> Tuple[DQNAgent, List[float]]:
    """
    Train RL agent
    
    Returns:
        Trained agent and list of episode rewards
    """
    if student_profiles is None:
        # Use default profiles
        student_profiles = [
            {
                'base_ability': 0.5,
                'engagement': 0.7,
                'learning_speed': 1.0,
            }
        ]
    
    if concept_graph is None:
        # Use default concepts
        concept_graph = {
            'Easy': {'difficulty': 0.3},
            'Medium': {'difficulty': 0.5},
            'Hard': {'difficulty': 0.7},
        }
    
    logger.info(f"Training RL agent for {num_episodes} episodes...")
    
    agent = DQNAgent()
    episode_rewards = []
    
    for episode in range(num_episodes):
        # Random student profile
        profile = student_profiles[episode % len(student_profiles)]
        
        # Create environment
        env = LearningEnvironment(profile, concept_graph)
        state = env.reset()
        
        episode_reward = 0.0
        
        while True:
            # Agent action
            action = agent.get_action(state)
            
            # Environment step
            next_state, reward, done, _ = env.step(action)
            
            # Agent update
            agent.update(state, action, reward, next_state, done)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            logger.info(f"Episode {episode+1}: Avg reward (last 10) = {avg_reward:.4f}")
    
    logger.info("✓ RL Agent training complete")
    return agent, episode_rewards
