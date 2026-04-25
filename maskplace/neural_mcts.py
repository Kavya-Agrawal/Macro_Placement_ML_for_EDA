import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class SequenceGNN(nn.Module):
    def __init__(self, num_static_features):
        super(SequenceGNN, self).__init__()
        in_channels = num_static_features + 1 
        
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 64)
        
        self.policy_head = nn.Linear(64, 1)
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, static_X, edge_index, sequence_mask):
        dynamic_X = torch.cat([static_X, sequence_mask.unsqueeze(1)], dim=1)
        
        node_embeddings = F.relu(self.fc1(dynamic_X))
        node_embeddings = F.relu(self.fc2(node_embeddings))
        
        logits = self.policy_head(node_embeddings).squeeze(-1)
        logits[sequence_mask == 1.0] = -float('inf')
        action_probs = F.softmax(logits, dim=0)
        
        placed_embeddings = node_embeddings[sequence_mask == 1.0]
        if placed_embeddings.size(0) > 0:
            global_context = torch.mean(placed_embeddings, dim=0)
        else:
            global_context = torch.mean(node_embeddings, dim=0) 
            
        predicted_value = self.value_head(global_context)
        return action_probs, predicted_value

class MCTSNode:
    def __init__(self, prior_prob):
        self.prior = prior_prob     
        self.visit_count = 0        
        self.value_sum = 0.0        
        self.children = {}          
        
    @property
    def q_value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def uct_score(self, parent_visit_count, c_puct=1.5):
        q_term = -self.q_value 
        u_term = c_puct * self.prior * (math.sqrt(parent_visit_count) / (1 + self.visit_count))
        return q_term + u_term

    def expand(self, action_probs, legal_actions):
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(prior_prob=action_probs[action].item())

    def apply_dirichlet_noise(self, alpha=0.3, epsilon=0.25):
        actions = list(self.children.keys())
        noise = np.random.dirichlet([alpha] * len(actions))
        for i, action in enumerate(actions):
            self.children[action].prior = (1 - epsilon) * self.children[action].prior + epsilon * noise[i]

class NeuralMCTS:
    def __init__(self, network, static_X, edge_index, num_macros, num_simulations=50):
        self.network = network
        self.static_X = static_X
        self.edge_index = edge_index
        self.num_macros = num_macros
        self.num_simulations = num_simulations

    def get_action_sequence(self, is_training=True):
        self.network.eval()
        sequence = []
        sequence_mask = torch.zeros(self.num_macros, dtype=torch.float, device=self.static_X.device)
        
        root = MCTSNode(prior_prob=1.0)
        
        # Expand root immediately so we can apply noise
        with torch.no_grad():
            action_probs, _ = self.network(self.static_X, self.edge_index, sequence_mask)
        legal_actions = list(range(self.num_macros))
        root.expand(action_probs, legal_actions)
        
        if is_training:
            root.apply_dirichlet_noise()
        
        for step in range(self.num_macros):
            for _ in range(self.num_simulations):
                self._simulate(root, sequence.copy(), sequence_mask.clone())
                
            if is_training:
                # Temperature-based sampling
                actions = list(root.children.keys())
                visits = np.array([child.visit_count for child in root.children.values()])
                temperature = 1.0
                visits_scaled = visits ** (1 / temperature)
                probabilities = visits_scaled / np.sum(visits_scaled)
                best_action = np.random.choice(actions, p=probabilities)
            else:
                best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            
            sequence.append(best_action)
            sequence_mask[best_action] = 1.0
            root = root.children[best_action]
            
        return sequence

    def _simulate(self, node, current_sequence, current_mask):
        search_path = [node]
        
        while node.children:
            action, node = max(node.children.items(), key=lambda item: item[1].uct_score(node.visit_count))
            current_sequence.append(action)
            current_mask[action] = 1.0
            search_path.append(node)
            
        with torch.no_grad():
            action_probs, predicted_value = self.network(self.static_X, self.edge_index, current_mask)
            predicted_value = predicted_value.item()
            
        if len(current_sequence) < self.num_macros:
            legal_actions = [i for i in range(self.num_macros) if current_mask[i] == 0.0]
            node.expand(action_probs, legal_actions)
            
        for n in search_path:
            n.visit_count += 1
            n.value_sum += predicted_value