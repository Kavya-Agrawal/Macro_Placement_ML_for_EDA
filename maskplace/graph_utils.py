# graph_utils.py

import torch

def build_graph(node_info, node_to_net_dict):
    node_names = list(node_info.keys())
    N = len(node_names)
    name_to_idx = {n: i for i, n in enumerate(node_names)}

    # -------- Features --------
    max_area = max(node_info[n]['x'] * node_info[n]['y'] for n in node_names)
    max_deg = max(len(node_to_net_dict[n]) for n in node_names)

    features = []
    for n in node_names:
        area = (node_info[n]['x'] * node_info[n]['y']) / max_area
        deg = len(node_to_net_dict[n]) / max_deg
        features.append([area, deg])

    x = torch.tensor(features, dtype=torch.float32)

    # -------- Adjacency --------
    adj = torch.zeros((N, N), dtype=torch.float32)

    for n1 in node_names:
        for n2 in node_names:
            if n1 == n2:
                continue
            if len(node_to_net_dict[n1] & node_to_net_dict[n2]) > 0:
                i, j = name_to_idx[n1], name_to_idx[n2]
                adj[i, j] = 1.0

    # normalize
    deg = adj.sum(dim=1, keepdim=True) + 1e-6
    adj = adj / deg

    return x, adj, node_names