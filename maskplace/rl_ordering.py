import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from collections import deque
import pickle
import os


class ActorNetwork(nn.Module):
    """Actor network that scores each macro for placement ordering"""
    
    def __init__(self, state_dim, hidden_dim=256, num_layers=3):
        super(ActorNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # State embedding layers
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
        
        self.state_encoder = nn.Sequential(*layers)
        
        # Macro feature encoder
        self.macro_encoder = nn.Sequential(
            nn.Linear(8, hidden_dim // 2),  # 8 features per macro
            nn.ReLU(),
            nn.LayerNorm(hidden_dim // 2)
        )
        
        # Attention mechanism for macro scoring
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True
        )
        
        # Final scoring layer
        self.score_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, state_features, macro_features, placed_mask):
        """
        Args:
            state_features: (batch_size, state_dim) - global state features
            macro_features: (batch_size, num_macros, 8) - per-macro features
            placed_mask: (batch_size, num_macros) - 1 if already placed, 0 otherwise
        
        Returns:
            logits: (batch_size, num_macros) - scores for each macro
        """
        batch_size, num_macros, _ = macro_features.shape
        
        # Encode global state
        state_embed = self.state_encoder(state_features)  # (batch_size, hidden_dim)
        state_embed = state_embed.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Encode macro features
        macro_embed = self.macro_encoder(macro_features)  # (batch_size, num_macros, hidden_dim//2)
        
        # Concatenate with state context
        state_expanded = state_embed[..., :self.hidden_dim // 2].expand(-1, num_macros, self.hidden_dim // 2)
        macro_context = torch.cat([macro_embed, state_expanded], dim=-1)  # (batch_size, num_macros, hidden_dim)
        
        # Apply attention
        attended, _ = self.attention(macro_context, macro_context, macro_context)
        
        # Score each macro
        logits = self.score_layer(attended).squeeze(-1)  # (batch_size, num_macros)
        
        # Mask already placed macros
        logits = logits.masked_fill(placed_mask.bool(), -1e9)
        
        return logits


class CriticNetwork(nn.Module):
    """Critic network that estimates state value"""
    
    def __init__(self, state_dim, hidden_dim=256, num_layers=3):
        super(CriticNetwork, self).__init__()
        
        layers = []
        layers.append(nn.Linear(state_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.LayerNorm(hidden_dim))
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.LayerNorm(hidden_dim))
        
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, state_features):
        """
        Args:
            state_features: (batch_size, state_dim)
        
        Returns:
            value: (batch_size, 1) - estimated value of the state
        """
        return self.network(state_features)


class RLOrderingAgent:
    """RL agent for learning macro placement ordering"""
    
    def __init__(self, placedb, hidden_dim=256, lr=1e-4, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.placedb = placedb
        self.device = device
        self.num_macros = len(placedb.node_info)
        
        # Calculate state dimension
        self.state_dim = self._calculate_state_dim()
        
        # Initialize networks
        self.actor = ActorNetwork(self.state_dim, hidden_dim).to(device)
        self.critic = CriticNetwork(self.state_dim, hidden_dim).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=lr
        )
        
        # Training history
        self.episode_rewards = []
        self.episode_losses = []
        
    def _calculate_state_dim(self):
        """Calculate the dimension of state features"""
        # Global features: num_placed, avg_hpwl, canvas_utilization, etc.
        global_features = 10
        # Aggregate macro features: avg_area, avg_connectivity, etc.
        aggregate_features = 10
        return global_features + aggregate_features
    
    def _extract_state_features(self, ordering, placed_mask):
        """Extract global state features"""
        num_placed = placed_mask.sum().item()
        
        # Calculate current partial HPWL
        current_hpwl = self._calculate_partial_hpwl(ordering)
        
        # Canvas utilization
        placed_area = sum([
            self.placedb.node_info[self.placedb.node_id_to_name[idx]]['x'] * 
            self.placedb.node_info[self.placedb.node_id_to_name[idx]]['y']
            for idx in ordering
        ])
        canvas_area = self.placedb.max_height * self.placedb.max_width
        utilization = placed_area / canvas_area if canvas_area > 0 else 0
        
        # Connectivity stats
        avg_connectivity = np.mean([
            len(self.placedb.node_to_net_dict[self.placedb.node_id_to_name[i]])
            for i in range(self.num_macros)
        ])
        
        # Build feature vector
        features = [
            num_placed / self.num_macros,  # Normalized number placed
            current_hpwl / 1e6,  # Normalized HPWL
            utilization,
            avg_connectivity / 100,
            len(ordering) / self.num_macros,
            0, 0, 0, 0, 0  # Additional features (can be extended)
        ]
        
        # Aggregate macro features
        total_area = sum([
            self.placedb.node_info[name]['x'] * self.placedb.node_info[name]['y']
            for name in self.placedb.node_info
        ])
        avg_area = total_area / self.num_macros if self.num_macros > 0 else 0
        
        aggregate_features = [
            avg_area / 1e6,
            0, 0, 0, 0, 0, 0, 0, 0, 0  # Can add more aggregate stats
        ]
        
        return torch.FloatTensor(features + aggregate_features).to(self.device)
    
    def _extract_macro_features(self):
        """Extract per-macro features"""
        macro_features = []
        
        for node_name in self.placedb.node_id_to_name:
            node = self.placedb.node_info[node_name]
            
            # Macro dimensions
            width = node['x']
            height = node['y']
            area = width * height
            aspect_ratio = width / height if height > 0 else 1.0
            
            # Connectivity
            num_nets = len(self.placedb.node_to_net_dict[node_name])
            
            # Normalize features
            features = [
                width / self.placedb.max_width,
                height / self.placedb.max_height,
                area / (self.placedb.max_width * self.placedb.max_height),
                aspect_ratio / 10.0,  # Normalized aspect ratio
                num_nets / 100.0,  # Normalized connectivity
                node['id'] / self.num_macros,  # Original ID (normalized)
                0,  # Placeholder for additional features
                0   # Placeholder for additional features
            ]
            
            macro_features.append(features)
        
        return torch.FloatTensor(macro_features).unsqueeze(0).to(self.device)
    
    def _calculate_partial_hpwl(self, ordering):
        """Calculate HPWL for partially placed macros"""
        if len(ordering) == 0:
            return 0.0
        
        # Simple HPWL estimation based on placed macros
        total_hpwl = 0.0
        placed_names = {self.placedb.node_id_to_name[idx] for idx in ordering}
        
        for net_name in self.placedb.net_info:
            net_nodes = set(self.placedb.net_info[net_name]['nodes'].keys())
            placed_in_net = net_nodes & placed_names
            
            if len(placed_in_net) >= 2:
                # Calculate bounding box of placed nodes in this net
                xs = [self.placedb.node_info[name].get('raw_x', 0) for name in placed_in_net]
                ys = [self.placedb.node_info[name].get('raw_y', 0) for name in placed_in_net]
                
                if xs and ys:
                    hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    total_hpwl += hpwl
        
        return total_hpwl
    
    def select_action(self, ordering, placed_mask, training=True):
        """Select next macro to place"""
        # Extract features
        state_features = self._extract_state_features(ordering, placed_mask).unsqueeze(0)
        macro_features = self._extract_macro_features()
        placed_mask_tensor = placed_mask.unsqueeze(0).to(self.device)
        
        # Get logits from actor
        logits = self.actor(state_features, macro_features, placed_mask_tensor)
        
        # Create distribution
        dist = Categorical(logits=logits.squeeze(0))
        
        if training:
            action = dist.sample()
        else:
            action = logits.squeeze(0).argmax()
        
        log_prob = dist.log_prob(action)
        value = self.critic(state_features).squeeze()
        
        return action.item(), log_prob, value
    
    def train_episode(self, alpha=1.0, beta=0.1):
        """Run one training episode"""
        # Initialize episode
        ordering = []
        placed_mask = torch.zeros(self.num_macros).to(self.device)
        
        log_probs = []
        values = []
        
        # Generate ordering
        for _ in range(self.num_macros):
            action, log_prob, value = self.select_action(ordering, placed_mask, training=True)
            
            log_probs.append(log_prob)
            values.append(value)
            
            ordering.append(action)
            placed_mask[action] = 1
        
        # Evaluate the ordering
        hpwl, cost = self._evaluate_ordering(ordering)
        reward = -(alpha * hpwl + beta * cost)
        
        # Compute loss
        actor_loss = 0
        critic_loss = 0
        
        for log_prob, value in zip(log_probs, values):
            advantage = reward - value
            actor_loss += -log_prob * advantage.detach()
            critic_loss += advantage.pow(2)
        
        total_loss = actor_loss + 0.5 * critic_loss
        
        # Optimize
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            max_norm=1.0
        )
        self.optimizer.step()
        
        # Record stats
        self.episode_rewards.append(reward)
        self.episode_losses.append(total_loss.item())
        
        return ordering, reward, total_loss.item(), hpwl, cost
    
    def _evaluate_ordering(self, ordering):
        """Evaluate an ordering using placement black box"""
        # Convert ordering to node names
        ordered_names = [self.placedb.node_id_to_name[idx] for idx in ordering]
        
        # Calculate HPWL (simplified - you can use your actual placement engine)
        hpwl = self._calculate_full_hpwl(ordered_names)
        
        # Calculate additional cost (e.g., overlap, density)
        cost = self._calculate_placement_cost(ordered_names)
        
        return hpwl / 1e6, cost  # Normalize
    
    def _calculate_full_hpwl(self, ordered_names):
        """Calculate full HPWL for the ordering"""
        total_hpwl = 0.0
        
        for net_name in self.placedb.net_info:
            net_nodes = list(self.placedb.net_info[net_name]['nodes'].keys())
            
            if len(net_nodes) >= 2:
                xs = [self.placedb.node_info[name].get('raw_x', 0) for name in net_nodes]
                ys = [self.placedb.node_info[name].get('raw_y', 0) for name in net_nodes]
                
                if xs and ys:
                    hpwl = (max(xs) - min(xs)) + (max(ys) - min(ys))
                    total_hpwl += hpwl
        
        return total_hpwl
    
    def _calculate_placement_cost(self, ordered_names):
        """Calculate additional placement cost"""
        # Simple density-based cost
        placed_area = sum([
            self.placedb.node_info[name]['x'] * self.placedb.node_info[name]['y']
            for name in ordered_names
        ])
        canvas_area = self.placedb.max_height * self.placedb.max_width
        
        # Penalize high utilization
        utilization = placed_area / canvas_area if canvas_area > 0 else 0
        cost = max(0, utilization - 0.7)  # Penalize if > 70% utilization
        
        return cost
    
    def generate_ordering(self, training=False):
        """Generate a complete ordering"""
        ordering = []
        placed_mask = torch.zeros(self.num_macros).to(self.device)
        
        with torch.no_grad() if not training else torch.enable_grad():
            for _ in range(self.num_macros):
                action, _, _ = self.select_action(ordering, placed_mask, training=False)
                ordering.append(action)
                placed_mask[action] = 1
        
        return ordering
    
    def save_checkpoint(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_rewards': self.episode_rewards,
            'episode_losses': self.episode_losses,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.episode_losses = checkpoint.get('episode_losses', [])
        print(f"Checkpoint loaded from {path}")


def train_rl_ordering(placedb, num_episodes=1000, alpha=1.0, beta=0.1, 
                      checkpoint_dir='checkpoints', log_interval=10):
    """Train the RL ordering agent"""
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    agent = RLOrderingAgent(placedb)
    
    best_reward = float('-inf')
    recent_rewards = deque(maxlen=100)
    
    print(f"Starting training for {num_episodes} episodes...")
    print(f"Number of macros: {agent.num_macros}")
    print(f"State dimension: {agent.state_dim}")
    
    for episode in range(num_episodes):
        ordering, reward, loss, hpwl, cost = agent.train_episode(alpha, beta)
        recent_rewards.append(reward)
        
        if (episode + 1) % log_interval == 0:
            avg_reward = np.mean(recent_rewards)
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Reward: {reward:.4f} | "
                  f"Avg Reward (100): {avg_reward:.4f} | "
                  f"Loss: {loss:.4f} | "
                  f"HPWL: {hpwl:.4f} | "
                  f"Cost: {cost:.4f}")
        
        # Save best model
        if reward > best_reward:
            best_reward = reward
            agent.save_checkpoint(os.path.join(checkpoint_dir, 'best_model.pt'))
        
        # Save periodic checkpoints
        if (episode + 1) % 100 == 0:
            agent.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{episode+1}.pt'))
    
    print(f"\nTraining completed! Best reward: {best_reward:.4f}")
    return agent


if __name__ == "__main__":
    # Example usage
    from place_db import PlaceDB
    
    # Load benchmark
    placedb = PlaceDB("ariane")
    
    # Train agent
    agent = train_rl_ordering(
        placedb,
        num_episodes=1000,
        alpha=1.0,
        beta=0.1,
        checkpoint_dir='rl_checkpoints',
        log_interval=10
    )
    
    # Generate ordering with trained model
    ordering = agent.generate_ordering(training=False)
    ordered_names = [placedb.node_id_to_name[idx] for idx in ordering]
    
    print("\nLearned ordering (first 10 macros):")
    for i, name in enumerate(ordered_names[:10]):
        print(f"{i}: {name}")
