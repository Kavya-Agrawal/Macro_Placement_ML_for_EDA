import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SequenceGNN(nn.Module):
    def __init__(self, num_static_features):
        super(SequenceGNN, self).__init__()
        # We add +1 to features for the dynamic "Sequence Mask"
        in_channels = num_static_features + 1 
        
        # NOTE: For a production build, replace these linear layers with 
        # torch_geometric layers (e.g., GATConv or GCNConv) to actually use edge_index.
        # We use Linear here as a structural placeholder so the pipeline runs.
        self.fc1 = nn.Linear(in_channels, 64)
        self.fc2 = nn.Linear(64, 64)
        
        # Policy Head: Predicts logits for the next macro
        self.policy_head = nn.Linear(64, 1)
        
        # Value Head: Predicts the final HPWL score
        self.value_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, static_X, edge_index, sequence_mask):
        """
        static_X: [num_macros, num_features]
        sequence_mask: [num_macros] (1.0 if placed, 0.0 if unplaced)
        """
        # Combine static features with the dynamic state
        dynamic_X = torch.cat([static_X, sequence_mask.unsqueeze(1)], dim=1)
        
        # Feature extraction
        node_embeddings = F.relu(self.fc1(dynamic_X))
        node_embeddings = F.relu(self.fc2(node_embeddings))
        
        # --- Policy Calculation ---
        logits = self.policy_head(node_embeddings).squeeze(-1)
        
        # MASKING: We cannot pick a macro that is already in the sequence.
        # Set logits of already placed macros to negative infinity.
        logits[sequence_mask == 1.0] = -float('inf')
        
        # Convert to probabilities
        action_probs = F.softmax(logits, dim=0)
        
        # --- Value Calculation ---
        # Pool the embeddings of ONLY the currently placed nodes to predict the outcome
        placed_embeddings = node_embeddings[sequence_mask == 1.0]
        if placed_embeddings.size(0) > 0:
            global_context = torch.mean(placed_embeddings, dim=0)
        else:
            global_context = torch.mean(node_embeddings, dim=0) # Fallback for root node
            
        predicted_value = self.value_head(global_context)
        
        return action_probs, predicted_value
    

class MCTSNode:
    def __init__(self, prior_prob):
        self.prior = prior_prob     # P(s,a) from the neural network
        self.visit_count = 0        # N(s,a)
        self.value_sum = 0.0        # Cumulative predicted HPWL
        self.children = {}          # Action (macro_id) -> MCTSNode
        
    @property
    def q_value(self):
        # Q(s,a): Average value. Note: For HPWL, lower is better.
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def uct_score(self, parent_visit_count, c_puct=1.0):
        # UCT calculation. We invert Q because we want to MINIMIZE wirelength
        q_term = -self.q_value 
        u_term = c_puct * self.prior * (math.sqrt(parent_visit_count) / (1 + self.visit_count))
        return q_term + u_term

    def expand(self, action_probs, legal_actions):
        for action in legal_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(prior_prob=action_probs[action].item())


class NeuralMCTS:
    def __init__(self, network, static_X, edge_index, num_macros, num_simulations=50):
        self.network = network
        self.static_X = static_X
        self.edge_index = edge_index
        self.num_macros = num_macros
        self.num_simulations = num_simulations

    def get_action_sequence(self):
        """Generates a full ordered sequence of macros from scratch."""
        self.network.eval()
        sequence = []
        
        # CRITICAL FIX: Ensure the mask generates on the same device as the features
        sequence_mask = torch.zeros(self.num_macros, dtype=torch.float, device=self.static_X.device)
        
        root = MCTSNode(prior_prob=1.0)
        
        for step in range(self.num_macros):
            for _ in range(self.num_simulations):
                self._simulate(root, sequence.copy(), sequence_mask.clone())
                
            best_action = max(root.children.items(), key=lambda item: item[1].visit_count)[0]
            
            sequence.append(best_action)
            sequence_mask[best_action] = 1.0
            
            root = root.children[best_action]
            
        return sequence

    def _simulate(self, node, current_sequence, current_mask):
        """One rollout of MCTS from the current node."""
        search_path = [node]
        
        # 1. SELECTION
        while node.children:
            action, node = max(
                node.children.items(), 
                key=lambda item: item[1].uct_score(node.visit_count)
            )
            current_sequence.append(action)
            current_mask[action] = 1.0
            search_path.append(node)
            
        # 2. EXPANSION & EVALUATION
        with torch.no_grad():
            action_probs, predicted_value = self.network(self.static_X, self.edge_index, current_mask)
            predicted_value = predicted_value.item()
            
        if len(current_sequence) < self.num_macros:
            legal_actions = [i for i in range(self.num_macros) if current_mask[i] == 0.0]
            node.expand(action_probs, legal_actions)
            
        # 3. BACKPROPAGATION
        for n in search_path:
            n.visit_count += 1
            n.value_sum += predicted_value
