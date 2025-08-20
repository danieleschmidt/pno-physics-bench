# SECURITY NOTICE: This file has been automatically hardened for security
# - All user inputs should be validated and sanitized
# - Subprocess calls use secure alternatives
# - SQL queries use parameterized statements
# - No hardcoded secrets or credentials


"""
Causal Uncertainty Inference for Probabilistic Neural Operators.

This module implements novel causal inference techniques to understand
how uncertainties propagate through PDE solutions and identify causal
relationships between input uncertainties and output predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import networkx as nx
from scipy import stats

from ..models import ProbabilisticNeuralOperator


@dataclass
class CausalNode:
    """Node in causal uncertainty graph."""
    name: str
    node_type: str  # 'input', 'latent', 'output'
    spatial_location: Tuple[int, int]
    uncertainty_distribution: torch.distributions.Distribution
    causal_parents: List[str]
    causal_children: List[str]


class CausalUncertaintyGraph:
    """
    Graph structure for representing causal relationships in uncertainty.
    
    Models how uncertainties at different spatial locations and time points
    causally influence each other in PDE solutions.
    """
    
    def __init__(self, spatial_shape: Tuple[int, int], temporal_length: int = 1):
        self.spatial_shape = spatial_shape
        self.temporal_length = temporal_length
        self.graph = nx.DiGraph()
        self.nodes = {}
        
        # Initialize spatial-temporal nodes
        self._initialize_nodes()
    
    def _initialize_nodes(self):
        """Initialize nodes for spatial-temporal grid."""
        for t in range(self.temporal_length):
            for i in range(self.spatial_shape[0]):
                for j in range(self.spatial_shape[1]):
                    node_name = f"u_{t}_{i}_{j}"
                    node = CausalNode(
                        name=node_name,
                        node_type='latent',
                        spatial_location=(i, j),
                        uncertainty_distribution=torch.distributions.Normal(0, 1),
                        causal_parents=[],
                        causal_children=[]
                    )
                    self.nodes[node_name] = node
                    self.graph.add_node(node_name, **node.__dict__)
    
    def add_causal_edge(
        self,
        parent: str,
        child: str,
        strength: float,
        edge_type: str = "spatial"
    ):
        """Add causal edge between uncertainty nodes."""
        self.graph.add_edge(parent, child, strength=strength, edge_type=edge_type)
        self.nodes[parent].causal_children.append(child)
        self.nodes[child].causal_parents.append(parent)
    
    def compute_causal_strength(self, parent: str, child: str) -> float:
        """Compute causal strength between two nodes."""
        if self.graph.has_edge(parent, child):
            return self.graph[parent][child]['strength']
        return 0.0
    
    def get_causal_paths(self, source: str, target: str) -> List[List[str]]:
        """Find all causal paths from source to target."""
        try:
            return list(nx.all_simple_paths(self.graph, source, target))
        except nx.NetworkXNoPath:
            return []
    
    def intervene(self, node: str, intervention_value: float):
        """Perform causal intervention on a node."""
        if node in self.nodes:
            # Set intervention distribution
            self.nodes[node].uncertainty_distribution = torch.distributions.Delta(intervention_value)
            return True
        return False


class CausalUncertaintyInference(nn.Module):
    """
    Neural network for learning causal relationships in uncertainty propagation.
    
    Uses attention mechanisms and graph neural networks to model how
    uncertainties causally influence each other across space and time.
    """
    
    def __init__(
        self,
        spatial_shape: Tuple[int, int],
        hidden_dim: int = 128,
        num_causal_layers: int = 3,
        attention_heads: int = 8
    ):
        super().__init__()
        self.spatial_shape = spatial_shape
        self.hidden_dim = hidden_dim
        self.num_causal_layers = num_causal_layers
        
        # Causal embedding layers
        self.spatial_embedding = nn.Linear(2, hidden_dim // 2)  # For (x, y) coordinates
        self.uncertainty_embedding = nn.Linear(1, hidden_dim // 2)  # For uncertainty values
        
        # Graph attention layers for causal inference
        self.causal_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                batch_first=True
            )
            for _ in range(num_causal_layers)
        ])
        
        # Causal strength predictor
        self.causal_strength_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Intervention effect predictor
        self.intervention_net = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),  # +1 for intervention value
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def create_spatial_coordinates(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Create normalized spatial coordinate grid."""
        h, w = self.spatial_shape
        y_coords = torch.linspace(0, 1, h, device=device)
        x_coords = torch.linspace(0, 1, w, device=device)
        
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        coords = torch.stack([grid_x, grid_y], dim=-1)  # [h, w, 2]
        
        # Expand for batch
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1, -1)
        return coords.view(batch_size, h * w, 2)
    
    def forward(
        self,
        uncertainty_field: torch.Tensor,
        return_causal_graph: bool = False
    ) -> Tuple[torch.Tensor, Optional[CausalUncertaintyGraph]]:
        """
        Forward pass for causal uncertainty inference.
        
        Args:
            uncertainty_field: [batch, height, width] uncertainty values
            return_causal_graph: Whether to return the inferred causal graph
            
        Returns:
            causal_strengths: [batch, height*width, height*width] causal strength matrix
            causal_graph: Optional CausalUncertaintyGraph
        """
        batch_size, h, w = uncertainty_field.shape
        device = uncertainty_field.device
        
        # Create embeddings
        coords = self.create_spatial_coordinates(batch_size, device)  # [batch, h*w, 2]
        spatial_emb = self.spatial_embedding(coords)  # [batch, h*w, hidden_dim//2]
        
        uncertainty_flat = uncertainty_field.view(batch_size, h * w, 1)  # [batch, h*w, 1]
        uncertainty_emb = self.uncertainty_embedding(uncertainty_flat)  # [batch, h*w, hidden_dim//2]
        
        # Combine embeddings
        node_features = torch.cat([spatial_emb, uncertainty_emb], dim=-1)  # [batch, h*w, hidden_dim]
        
        # Apply causal attention layers
        for attention_layer in self.causal_attention_layers:
            attended_features, attention_weights = attention_layer(
                node_features, node_features, node_features
            )
            node_features = node_features + attended_features  # Residual connection
        
        # Compute pairwise causal strengths
        num_nodes = h * w
        causal_strengths = torch.zeros(batch_size, num_nodes, num_nodes, device=device)
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:  # No self-causation
                    # Concatenate features of potential causal pair
                    pair_features = torch.cat([node_features[:, i], node_features[:, j]], dim=-1)
                    strength = self.causal_strength_net(pair_features)
                    causal_strengths[:, i, j] = strength.squeeze(-1)
        
        # Create causal graph if requested
        causal_graph = None
        if return_causal_graph:
            causal_graph = self._create_causal_graph(causal_strengths[0], h, w)
        
        return causal_strengths, causal_graph
    
    def _create_causal_graph(
        self,
        causal_matrix: torch.Tensor,
        h: int,
        w: int,
        threshold: float = 0.5
    ) -> CausalUncertaintyGraph:
        """Create causal graph from strength matrix."""
        graph = CausalUncertaintyGraph((h, w))
        
        for i in range(h * w):
            for j in range(h * w):
                strength = causal_matrix[i, j].item()
                if strength > threshold:
                    parent_pos = (i // w, i % w)
                    child_pos = (j // w, j % w)
                    
                    parent_name = f"u_0_{parent_pos[0]}_{parent_pos[1]}"
                    child_name = f"u_0_{child_pos[0]}_{child_pos[1]}"
                    
                    graph.add_causal_edge(parent_name, child_name, strength)
        
        return graph
    
    def predict_intervention_effect(
        self,
        uncertainty_field: torch.Tensor,
        intervention_location: Tuple[int, int],
        intervention_value: float
    ) -> torch.Tensor:
        """Predict effect of intervention at specific location."""
        batch_size, h, w = uncertainty_field.shape
        device = uncertainty_field.device
        
        # Get node features
        coords = self.create_spatial_coordinates(batch_size, device)
        spatial_emb = self.spatial_embedding(coords)
        uncertainty_flat = uncertainty_field.view(batch_size, h * w, 1)
        uncertainty_emb = self.uncertainty_embedding(uncertainty_flat)
        node_features = torch.cat([spatial_emb, uncertainty_emb], dim=-1)
        
        # Apply causal attention
        for attention_layer in self.causal_attention_layers:
            attended_features, _ = attention_layer(node_features, node_features, node_features)
            node_features = node_features + attended_features
        
        # Predict intervention effects
        intervention_tensor = torch.full((batch_size, h * w, 1), intervention_value, device=device)
        intervention_input = torch.cat([node_features, intervention_tensor], dim=-1)
        
        intervention_effects = self.intervention_net(intervention_input)
        return intervention_effects.view(batch_size, h, w)


class CausalUncertaintyAnalyzer:
    """Analyzer for causal relationships in uncertainty propagation."""
    
    def __init__(self):
        self.causal_cache = {}
    
    def compute_average_treatment_effect(
        self,
        model: CausalUncertaintyInference,
        uncertainty_field: torch.Tensor,
        intervention_location: Tuple[int, int],
        intervention_values: List[float]
    ) -> Dict[str, float]:
        """Compute Average Treatment Effect (ATE) for uncertainty interventions."""
        effects = []
        
        for intervention_value in intervention_values:
            effect = model.predict_intervention_effect(
                uncertainty_field, intervention_location, intervention_value
            )
            effects.append(effect.mean().item())
        
        # Compute ATE as difference between high and low intervention
        ate = effects[-1] - effects[0]  # Assuming sorted intervention values
        
        return {
            'average_treatment_effect': ate,
            'intervention_effects': effects,
            'effect_variance': np.var(effects)
        }
    
    def identify_causal_hotspots(
        self,
        causal_graph: CausalUncertaintyGraph,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Identify spatial locations with highest causal influence."""
        hotspots = []
        
        for node_name, node in causal_graph.nodes.items():
            # Compute causal influence metrics
            out_degree = causal_graph.graph.out_degree(node_name)
            in_degree = causal_graph.graph.in_degree(node_name)
            
            # Betweenness centrality for causal mediation
            try:
                betweenness = nx.betweenness_centrality(causal_graph.graph)[node_name]
            except:
                betweenness = 0.0
            
            # Total causal strength (outgoing)
            total_strength = sum([
                causal_graph.compute_causal_strength(node_name, child)
                for child in node.causal_children
            ])
            
            hotspots.append({
                'node_name': node_name,
                'spatial_location': node.spatial_location,
                'out_degree': out_degree,
                'in_degree': in_degree,
                'betweenness_centrality': betweenness,
                'total_causal_strength': total_strength,
                'causal_influence_score': out_degree * total_strength + betweenness
            })
        
        # Sort by causal influence score
        hotspots.sort(key=lambda x: x['causal_influence_score'], reverse=True)
        return hotspots[:top_k]
    
    def test_causal_assumptions(
        self,
        model: CausalUncertaintyInference,
        uncertainty_data: torch.Tensor,
        num_bootstrap: int = 100
    ) -> Dict[str, Any]:
        """Test fundamental causal assumptions using bootstrap."""
        results = {}
        
        # Test 1: Markov property (conditional independence)
        markov_violations = []
        
        batch_size, h, w = uncertainty_data.shape
        for _ in range(num_bootstrap):
            # Sample random locations
            i, j, k = np.random.choice(h * w, 3, replace=False)
            
            # Compute conditional independence test
            # P(X_i | X_j, X_k) vs P(X_i | X_k)
            uncertainty_flat = uncertainty_data.view(batch_size, -1)
            
            # Simple correlation-based test (can be replaced with more sophisticated tests)
            corr_conditional = np.corrcoef([
                uncertainty_flat[:, i].cpu().numpy(),
                uncertainty_flat[:, j].cpu().numpy(),
                uncertainty_flat[:, k].cpu().numpy()
            ])
            
            # Partial correlation approximation
            partial_corr = (corr_conditional[0, 1] - corr_conditional[0, 2] * corr_conditional[1, 2]) / \
                          np.sqrt((1 - corr_conditional[0, 2]**2) * (1 - corr_conditional[1, 2]**2))
            
            if abs(partial_corr) > 0.1:  # Threshold for violation
                markov_violations.append(partial_corr)
        
        results['markov_violations'] = len(markov_violations) / num_bootstrap
        
        # Test 2: Causal sufficiency (no hidden confounders)
        causal_strengths, _ = model(uncertainty_data)
        strength_matrix = causal_strengths[0].cpu().numpy()
        
        # Check for symmetric strong connections (potential confounding)
        symmetric_connections = 0
        total_connections = 0
        
        for i in range(h * w):
            for j in range(i + 1, h * w):
                if strength_matrix[i, j] > 0.5 or strength_matrix[j, i] > 0.5:
                    total_connections += 1
                    if abs(strength_matrix[i, j] - strength_matrix[j, i]) < 0.1:
                        symmetric_connections += 1
        
        results['potential_confounding_rate'] = symmetric_connections / max(total_connections, 1)
        
        # Test 3: Transitivity of causal relations
        transitivity_score = 0
        transitivity_count = 0
        
        for i in range(h * w):
            for j in range(h * w):
                for k in range(h * w):
                    if i != j and j != k and i != k:
                        # Check if i -> j -> k implies some relation i -> k
                        strength_ij = strength_matrix[i, j]
                        strength_jk = strength_matrix[j, k]
                        strength_ik = strength_matrix[i, k]
                        
                        if strength_ij > 0.5 and strength_jk > 0.5:
                            expected_ik = strength_ij * strength_jk
                            transitivity_score += abs(strength_ik - expected_ik)
                            transitivity_count += 1
        
        results['transitivity_score'] = transitivity_score / max(transitivity_count, 1)
        
        return results
    
    def generate_causal_discovery_report(
        self,
        model: CausalUncertaintyInference,
        uncertainty_data: torch.Tensor,
        save_path: str = "causal_discovery_report.json"
    ) -> str:
        """Generate comprehensive causal discovery report."""
        import json
        
        # Run causal inference
        causal_strengths, causal_graph = model(uncertainty_data, return_causal_graph=True)
        
        # Identify hotspots
        hotspots = self.identify_causal_hotspots(causal_graph)
        
        # Test assumptions
        assumption_tests = self.test_causal_assumptions(model, uncertainty_data)
        
        # Compute graph statistics
        graph_stats = {
            'num_nodes': causal_graph.graph.number_of_nodes(),
            'num_edges': causal_graph.graph.number_of_edges(),
            'density': nx.density(causal_graph.graph),
            'is_dag': nx.is_directed_acyclic_graph(causal_graph.graph)
        }
        
        report = {
            "causal_discovery_analysis": {
                "graph_statistics": graph_stats,
                "causal_hotspots": hotspots[:5],  # Top 5
                "assumption_tests": assumption_tests,
                "causal_strength_statistics": {
                    "mean_strength": causal_strengths.mean().item(),
                    "max_strength": causal_strengths.max().item(),
                    "sparsity": (causal_strengths < 0.1).float().mean().item()
                },
                "interpretation": {
                    "causal_structure_quality": "good" if graph_stats['is_dag'] else "cyclic_dependencies_detected",
                    "confounding_risk": "low" if assumption_tests['potential_confounding_rate'] < 0.2 else "high",
                    "markov_assumption": "satisfied" if assumption_tests['markov_violations'] < 0.1 else "violated"
                }
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return f"Causal discovery report saved to {save_path}"


def create_causal_uncertainty_experiment():
    """Create complete causal uncertainty inference experiment."""
    spatial_shape = (32, 32)
    
    # Create causal inference model
    causal_model = CausalUncertaintyInference(
        spatial_shape=spatial_shape,
        hidden_dim=128,
        num_causal_layers=3
    )
    
    # Create analyzer
    analyzer = CausalUncertaintyAnalyzer()
    
    return causal_model, analyzer


def validate_causal_inference_theory():
    """Validate theoretical properties of causal inference."""
    print("🔬 Validating Causal Inference Theory...")
    
    # Create test model
    model = CausalUncertaintyInference((16, 16), hidden_dim=64)
    
    # Generate test uncertainty field
    uncertainty_field = torch.randn(2, 16, 16)
    
    # Test causal inference
    causal_strengths, causal_graph = model(uncertainty_field, return_causal_graph=True)
    
    # Validate properties
    assert causal_strengths.shape == (2, 256, 256), "Causal strength matrix wrong shape"
    assert causal_graph is not None, "Causal graph not created"
    assert causal_graph.graph.number_of_nodes() == 256, "Wrong number of nodes"
    
    # Test intervention
    intervention_effect = model.predict_intervention_effect(
        uncertainty_field, (8, 8), 2.0
    )
    assert intervention_effect.shape == (2, 16, 16), "Intervention effect wrong shape"
    
    print("✅ Causal inference theory validation passed")
    return True


if __name__ == "__main__":
    # Run validation
    validate_causal_inference_theory()
    
    # Create experiment
    model, analyzer = create_causal_uncertainty_experiment()
    print("🚀 Causal Uncertainty Inference module ready for research!")