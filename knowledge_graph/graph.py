"""Knowledge Graph for Learning Concepts

Builds and manages the concept dependency graph for intelligent
prerequisite reasoning and optimal learning path generation.
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Set, Tuple, Optional, Any
import logging
import json

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    """
    Knowledge graph representing concept dependencies
    
    Nodes: Learning concepts/topics
    Edges: prerequisite, similar_to, depends_on relationships
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_embeddings = {}
    
    def add_concept(self, concept: str, difficulty: float = 0.5,
                   category: str = "General") -> None:
        """Add a concept node to the graph"""
        self.graph.add_node(
            concept,
            difficulty=difficulty,
            category=category,
            mastery=0.0  # Initially unmastered
        )
    
    def add_prerequisite(self, prerequisite: str, dependent: str) -> None:
        """Add prerequisite edge: prerequisite -> dependent"""
        self.graph.add_edge(
            prerequisite, dependent,
            relationship_type='prerequisite'
        )
    
    def load_from_file(self, filepath: str) -> None:
        """Load graph from JSON file"""
        logger.info(f"Loading knowledge graph from {filepath}...")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Add nodes
        for node in data.get('nodes', []):
            self.add_concept(
                node['id'],
                difficulty=node.get('difficulty', 0.5),
                category=node.get('category', 'General')
            )
        
        # Add edges
        for edge in data.get('edges', []):
            if edge.get('type') == 'prerequisite':
                self.add_prerequisite(edge['source'], edge['target'])
        
        logger.info(f"✓ Loaded {self.graph.number_of_nodes()} concepts and "
                   f"{self.graph.number_of_edges()} edges")
    
    def get_prerequisites(self, concept: str) -> Set[str]:
        """Get all prerequisites for a concept"""
        if concept not in self.graph:
            return set()

        prerequisites = set()
        
        def dfs(node):
            for predecessor in self.graph.predecessors(node):
                if predecessor not in prerequisites:
                    prerequisites.add(predecessor)
                    dfs(predecessor)
        
        dfs(concept)
        return prerequisites
    
    def get_dependents(self, concept: str) -> Set[str]:
        """Get all concepts that depend on this concept"""
        if concept not in self.graph:
            return set()

        dependents = set()
        
        def dfs(node):
            for successor in self.graph.successors(node):
                if successor not in dependents:
                    dependents.add(successor)
                    dfs(successor)
        
        dfs(concept)
        return dependents
    
    def is_prerequisite_satisfied(self, concept: str, mastery_levels: Dict[str, float],
                                 mastery_threshold: float = 0.7) -> bool:
        """Check if all prerequisites for a concept are mastered"""
        prerequisites = self.get_prerequisites(concept)
        
        for prereq in prerequisites:
            if mastery_levels.get(prereq, 0.0) < mastery_threshold:
                return False
        
        return True
    
    def get_missing_prerequisites(self, concept: str,
                                 mastery_levels: Dict[str, float],
                                 mastery_threshold: float = 0.7) -> List[str]:
        """Get list of unmastered prerequisites"""
        prerequisites = self.get_prerequisites(concept)
        
        missing = []
        for prereq in prerequisites:
            if mastery_levels.get(prereq, 0.0) < mastery_threshold:
                missing.append(prereq)
        
        return missing
    
    def find_learning_path(self, start_concept: str, end_concept: str) -> List[str]:
        """Find shortest learning path between two concepts"""
        try:
            path = nx.shortest_path(self.graph, start_concept, end_concept)
            return path
        except nx.NetworkXNoPath:
            logger.warning(f"No path from {start_concept} to {end_concept}")
            return []
    
    def get_recommended_next_concepts(self, current_mastery: Dict[str, float],
                                     top_k: int = 5,
                                     mastery_threshold: float = 0.7) -> List[str]:
        """
        Get recommended next concepts to learn
        
        Strategy: Concepts whose prerequisites are mostly satisfied
        """
        candidates = []
        
        for concept in self.graph.nodes():
            # Skip already mastered
            if current_mastery.get(concept, 0.0) >= mastery_threshold:
                continue
            
            # Check prerequisite satisfaction
            prerequisites = self.get_prerequisites(concept)
            
            if not prerequisites:
                # No prerequisites
                candidates.append((concept, 1.0))
            else:
                # Fraction of satisfied prerequisites
                satisfied = sum(
                    1 for prereq in prerequisites
                    if current_mastery.get(prereq, 0.0) >= mastery_threshold
                )
                satisfaction_ratio = satisfied / len(prerequisites)
                
                if satisfaction_ratio > 0.5:  # At least 50% prerequisites satisfied
                    candidates.append((concept, satisfaction_ratio))
        
        # Sort by satisfaction ratio and difficulty
        candidates.sort(
            key=lambda x: (
                -x[1],  # Higher satisfaction first
                self.graph.nodes[x[0]]['difficulty']  # Lower difficulty first
            )
        )
        
        return [concept for concept, _ in candidates[:top_k]]

    def get_prerequisite_chain(self, concept: str) -> List[str]:
        """Return an ordered prerequisite chain for a target concept."""
        if concept not in self.graph:
            return []

        prerequisites = self.get_prerequisites(concept)
        nodes = set(prerequisites) | {concept}
        subgraph = self.graph.subgraph(nodes).copy()

        try:
            ordered = list(nx.topological_sort(subgraph))
        except nx.NetworkXUnfeasible:
            ordered = list(nodes)

        return ordered

    def get_readiness(self, concept: str,
                      mastery_levels: Dict[str, float],
                      mastery_threshold: float = 0.7) -> float:
        """Compute a readiness score for a target concept based on mastery of prerequisites."""
        if concept not in self.graph:
            return 0.0

        prerequisites = self.get_prerequisites(concept)
        target_mastery = mastery_levels.get(concept, 0.0)
        if not prerequisites:
            return float(np.clip(target_mastery, 0.0, 1.0))

        prereq_masteries = [mastery_levels.get(prereq, 0.0) for prereq in prerequisites]
        avg_prereq = float(np.mean(prereq_masteries))
        missing = sum(1 for value in prereq_masteries if value < mastery_threshold)
        missing_ratio = missing / len(prereq_masteries)

        readiness = 0.6 * avg_prereq + 0.4 * target_mastery
        readiness *= max(0.0, 1.0 - 0.3 * missing_ratio)
        return float(np.clip(readiness, 0.0, 1.0))

    def get_readiness_scores(self, mastery_levels: Dict[str, float],
                             mastery_threshold: float = 0.7) -> Dict[str, float]:
        return {
            concept: self.get_readiness(concept, mastery_levels, mastery_threshold)
            for concept in self.graph.nodes()
        }

    def get_root_cause(self, concept: str,
                       mastery_levels: Dict[str, float],
                       mastery_threshold: float = 0.7) -> Dict[str, Any]:
        """Analyze why a student is struggling with a target concept."""
        if concept not in self.graph:
            return {
                'target': concept,
                'readiness': 0.0,
                'weak_prerequisites': [],
                'prerequisite_chain': []
            }

        prerequisites = self.get_prerequisites(concept)
        weak = [
            {
                'topic': prereq,
                'mastery': float(mastery_levels.get(prereq, 0.0))
            }
            for prereq in prerequisites
            if mastery_levels.get(prereq, 0.0) < mastery_threshold
        ]
        weak.sort(key=lambda x: x['mastery'])

        return {
            'target': concept,
            'readiness': self.get_readiness(concept, mastery_levels, mastery_threshold),
            'weak_prerequisites': weak,
            'prerequisite_chain': self.get_prerequisite_chain(concept)
        }

    def get_remediation_plan(self, concept: str,
                             mastery_levels: Dict[str, float],
                             mastery_threshold: float = 0.7,
                             max_steps: int = 10) -> Dict[str, Any]:
        """Generate an ordered remediation plan for a target concept."""
        chain = self.get_prerequisite_chain(concept)
        if not chain:
            return {
                'target': concept,
                'remediation_plan': [],
                'full_path': []
            }

        plan = []
        for topic in chain:
            mastery = float(mastery_levels.get(topic, 0.0))
            if mastery < mastery_threshold:
                plan.append({'topic': topic, 'mastery': mastery})

        if concept not in [item['topic'] for item in plan]:
            plan.append({'topic': concept, 'mastery': float(mastery_levels.get(concept, 0.0))})

        plan = plan[:max_steps]
        return {
            'target': concept,
            'remediation_plan': plan,
            'full_path': chain
        }

    def get_prerequisite_tree(self, concept: str) -> List[str]:
        """Return the full prerequisite chain for a target concept."""
        return self.get_prerequisite_chain(concept)

    def compute_concept_embeddings(self, embedding_dim: int = 64) -> Dict[str, np.ndarray]:
        """
        Compute embeddings for concepts based on graph structure
        
        Uses graph-based features: in-degree, out-degree, betweenness, etc.
        """
        embeddings = {}
        
        # Compute various centrality measures
        in_degree = dict(self.graph.in_degree())
        out_degree = dict(self.graph.out_degree())
        betweenness = nx.betweenness_centrality(self.graph)
        
        difficulties = nx.get_node_attributes(self.graph, 'difficulty')
        
        for concept in self.graph.nodes():
            features = [
                in_degree.get(concept, 0),
                out_degree.get(concept, 0),
                betweenness.get(concept, 0),
                difficulties.get(concept, 0.5),
                len(self.get_prerequisites(concept)),
                len(self.get_dependents(concept)),
            ]
            
            # Pad to embedding_dim
            if len(features) < embedding_dim:
                features.extend([0.0] * (embedding_dim - len(features)))
            else:
                features = features[:embedding_dim]
            
            embeddings[concept] = np.array(features, dtype=np.float32)
        
        return embeddings
    
    def visualize_subgraph(self, concepts: List[str]) -> Dict:
        """Create visualization data for a subgraph"""
        subgraph = self.graph.subgraph(concepts)
        
        nodes = [
            {
                'id': node,
                'difficulty': subgraph.nodes[node]['difficulty'],
                'category': subgraph.nodes[node]['category']
            }
            for node in subgraph.nodes()
        ]
        
        edges = [
            {'source': u, 'target': v}
            for u, v in subgraph.edges()
        ]
        
        return {'nodes': nodes, 'edges': edges}

    def infer_weaknesses(self, mastery_levels: Dict[str, float],
                         low_mastery_threshold: float = 0.5,
                         propagation_depth: int = 3) -> List[Tuple[str, float]]:
        """
        Infer likely upstream weaknesses given observed low mastery concepts.

        Strategy:
        - For each concept with mastery below `low_mastery_threshold`, traverse
          its prerequisite ancestors up to `propagation_depth` hops.
        - Score ancestor concepts by inverse distance (1 / (1 + distance))
          summed across observed failures.
        - Return a sorted list of (concept, score) descending.
        """
        scores: Dict[str, float] = {}

        # Identify target concepts (where student struggles)
        low_concepts = [c for c, m in mastery_levels.items() if m < low_mastery_threshold]

        for target in low_concepts:
            # Skip targets that are not in the graph
            if target not in self.graph:
                continue

            # BFS up the prerequisites
            visited = set()
            queue: List[Tuple[str, int]] = [(target, 0)]

            while queue:
                node, dist = queue.pop(0)
                if dist == 0:
                    # skip the target itself as a weakness candidate
                    pass
                else:
                    influence = 1.0 / (1 + dist)
                    scores[node] = scores.get(node, 0.0) + influence

                if dist >= propagation_depth:
                    continue

                try:
                    predecessors_iter = self.graph.predecessors(node)
                except Exception:
                    predecessors_iter = []

                for prereq in predecessors_iter:
                    if prereq in visited:
                        continue
                    visited.add(prereq)
                    queue.append((prereq, dist + 1))

        # Normalize scores to [0,1]
        if not scores:
            return []

        max_score = max(scores.values())
        ranked = sorted([(c, s / max_score) for c, s in scores.items()], key=lambda x: -x[1])
        return ranked


def create_sample_knowledge_graph() -> KnowledgeGraph:
    """Create a sample knowledge graph for testing"""
    kg = KnowledgeGraph()
    
    # Linear Algebra path
    kg.add_concept("Linear Algebra Basics", 0.4, "Mathematics")
    kg.add_concept("Vectors", 0.5, "Mathematics")
    kg.add_concept("Matrices", 0.6, "Mathematics")
    kg.add_concept("Matrix Operations", 0.65, "Mathematics")
    kg.add_concept("Eigenvalues and Eigenvectors", 0.7, "Mathematics")
    
    kg.add_prerequisite("Linear Algebra Basics", "Vectors")
    kg.add_prerequisite("Vectors", "Matrices")
    kg.add_prerequisite("Matrices", "Matrix Operations")
    kg.add_prerequisite("Matrix Operations", "Eigenvalues and Eigenvectors")
    
    # Neural Network path
    kg.add_concept("Neural Network Basics", 0.5, "Deep Learning")
    kg.add_concept("Activation Functions", 0.6, "Deep Learning")
    kg.add_concept("Backpropagation", 0.7, "Deep Learning")
    kg.add_concept("Transformers", 0.8, "Deep Learning")
    kg.add_concept("Attention Mechanisms", 0.75, "Deep Learning")
    kg.add_concept("Transformer Architecture", 0.85, "Deep Learning")
    
    kg.add_prerequisite("Linear Algebra Basics", "Neural Network Basics")
    kg.add_prerequisite("Neural Network Basics", "Activation Functions")
    kg.add_prerequisite("Activation Functions", "Backpropagation")
    kg.add_prerequisite("Matrices", "Backpropagation")
    kg.add_prerequisite("Backpropagation", "Transformers")
    kg.add_prerequisite("Attention Mechanisms", "Transformer Architecture")
    kg.add_prerequisite("Backpropagation", "Transformer Architecture")
    kg.add_prerequisite("Transformers", "Transformer Architecture")
    
    # Python path
    kg.add_concept("Python Fundamentals", 0.3, "Programming")
    kg.add_concept("Python Data Structures", 0.5, "Programming")
    kg.add_concept("NumPy Basics", 0.5, "Programming")
    kg.add_concept("Pandas", 0.6, "Programming")
    
    kg.add_prerequisite("Python Fundamentals", "Python Data Structures")
    kg.add_prerequisite("Python Fundamentals", "NumPy Basics")
    kg.add_prerequisite("NumPy Basics", "Pandas")
    
    return kg


if __name__ == "__main__":
    # Create and test sample graph
    kg = create_sample_knowledge_graph()
    
    logger.info("Knowledge Graph Statistics:")
    logger.info(f"  Concepts: {kg.graph.number_of_nodes()}")
    logger.info(f"  Prerequisites: {kg.graph.number_of_edges()}")
    
    # Test prerequisites
    logger.info(f"\nPrerequisites for 'Transformers': {kg.get_prerequisites('Transformers')}")
    
    # Test bottlenecks
    logger.info(f"\nBottleneck concepts: {kg.get_bottleneck_concepts()}")
    
    # Test embeddings
    embeddings = kg.compute_concept_embeddings(embedding_dim=32)
    logger.info(f"\nCreated {len(embeddings)} concept embeddings")
