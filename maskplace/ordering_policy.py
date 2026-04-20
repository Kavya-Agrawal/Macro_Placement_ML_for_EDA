import torch
import torch.nn.functional as F
from graph_utils import build_graph


def sample_ordering(model, node_info, node_to_net_dict, device):

    x, adj, node_names = build_graph(node_info, node_to_net_dict)

    x = x.to(device)
    adj = adj.to(device)

    selected, log_prob = model(x, adj)

    ordering = [node_names[i] for i in selected]

    return ordering, log_prob


# def sample_ordering(model, node_info, node_to_net_dict, device):

#     x, adj, node_names = build_graph(node_info, node_to_net_dict)

#     x = x.to(device)
#     adj = adj.to(device)

#     scores = model(x, adj)  # [N]

#     probs = F.softmax(scores, dim=0)

#     selected = []
#     log_probs = []

#     # ---- IMPORTANT: do NOT modify probs in-place ----
#     mask = torch.ones_like(probs)

#     for _ in range(len(node_names)):

#         probs_masked = probs * mask
#         probs_masked = probs_masked / (probs_masked.sum() + 1e-8)

#         dist = torch.distributions.Categorical(probs_masked)
#         idx = dist.sample()

#         log_probs.append(dist.log_prob(idx))
#         selected.append(idx.item())

#         # ---- SAFE mask update (no inplace on graph tensor) ----
#         mask = mask.clone()
#         mask[idx] = 0.0

#     ordering = [node_names[i] for i in selected]

#     return ordering, torch.stack(log_probs).sum()