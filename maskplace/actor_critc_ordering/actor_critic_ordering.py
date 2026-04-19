import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# =========================
# ACTOR-CRITIC NETWORK
# =========================

class ActorCritic(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # Actor: per-node scoring
        self.actor_fc1 = nn.Linear(input_dim, 32)
        self.actor_fc2 = nn.Linear(32, 1)

        # Critic: global value
        self.critic_fc1 = nn.Linear(input_dim, 32)
        self.critic_fc2 = nn.Linear(32, 32)
        self.critic_out = nn.Linear(32, 1)

    def forward_actor(self, x, mask):
        """
        x: (N, F)
        mask: (N,) -> 1 if already placed
        """
        h = F.relu(self.actor_fc1(x))
        scores = self.actor_fc2(h).squeeze(-1)   # (N,)

        # mask already selected nodes
        scores[mask == 1] = -1e9

        return scores

    def forward_critic(self, x):
        """
        x: (N, F)
        """
        h = F.relu(self.critic_fc1(x))
        h = h.mean(dim=0)  # pooling
        h = F.relu(self.critic_fc2(h))
        value = self.critic_out(h)
        return value


# =========================
# STATE BUILDER
# =========================

def build_features(node_info, node_to_net_dict, placed_mask):
    """
    Build simple features per macro
    """
    features = []

    max_area = max(node_info[n]["x"] * node_info[n]["y"] for n in node_info)
    max_deg = max(len(node_to_net_dict[n]) for n in node_info)

    for node_name in node_info:
        area = node_info[node_name]["x"] * node_info[node_name]["y"] / max_area
        degree = len(node_to_net_dict[node_name]) / max_deg
        placed = placed_mask[node_name]

        features.append([area, degree, placed])

    return torch.tensor(features, dtype=torch.float32)


# =========================
# RL ORDERING FUNCTION
# =========================

def rl_ordering(node_info, node_to_net_dict, model):
    """
    Replaces get_node_id_to_name_topology
    """

    node_list = list(node_info.keys())
    placed_mask = {n: 0 for n in node_list}

    ordering = []
    log_probs = []
    values = []

    for step in range(len(node_list)):
        x = build_features(node_info, node_to_net_dict, placed_mask)

        mask_tensor = torch.tensor(
            [placed_mask[n] for n in node_list], dtype=torch.float32
        )

        logits = model.forward_actor(x, mask_tensor)
        dist = Categorical(logits=logits)

        action_idx = dist.sample()
        action_node = node_list[action_idx.item()]

        log_probs.append(dist.log_prob(action_idx))
        values.append(model.forward_critic(x))

        ordering.append(action_node)
        placed_mask[action_node] = 1

    return ordering, log_probs, values


# =========================
# TRAIN STEP
# =========================

def train_step(model, optimizer, log_probs, values, reward):
    actor_loss = 0
    critic_loss = 0

    for lp, v in zip(log_probs, values):
        advantage = reward - v
        actor_loss += -lp * advantage.detach()
        critic_loss += advantage.pow(2)

    loss = actor_loss + 0.5 * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()