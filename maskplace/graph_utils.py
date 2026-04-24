# # graph_utils.py

# import torch

# def build_graph(node_info, node_to_net_dict):
#     node_names = list(node_info.keys())
#     N = len(node_names)
#     name_to_idx = {n: i for i, n in enumerate(node_names)}

#     # -------- Features --------
#     max_area = max(node_info[n]['x'] * node_info[n]['y'] for n in node_names)
#     max_deg = max(len(node_to_net_dict[n]) for n in node_names)

#     features = []
#     for n in node_names:
#         area = (node_info[n]['x'] * node_info[n]['y']) / max_area
#         deg = len(node_to_net_dict[n]) / max_deg
#         features.append([area, deg])

#     x = torch.tensor(features, dtype=torch.float32)

#     # -------- Adjacency --------
#     adj = torch.zeros((N, N), dtype=torch.float32)

#     for n1 in node_names:
#         for n2 in node_names:
#             if n1 == n2:
#                 continue
#             if len(node_to_net_dict[n1] & node_to_net_dict[n2]) > 0:
#                 i, j = name_to_idx[n1], name_to_idx[n2]
#                 adj[i, j] = 1.0

#     # normalize
#     deg = adj.sum(dim=1, keepdim=True) + 1e-6
#     adj = adj / deg

#     return x, adj, node_names


import torch

def build_graph(node_info, node_to_net_dict, device):

    node_names = list(node_info.keys())
    N = len(node_names)

    # max_area1=0
    # areas=[]
    # ans=""

    name_to_idx = {n: i for i, n in enumerate(node_names)}

    # ---------------- FEATURE NORMALIZATION ----------------
    areas = [node_info[n]['x'] * node_info[n]['y'] for n in node_names]
    # for n in node_names:
    #     area1 = node_info[n]['x'] * node_info[n]['y']
    #     areas.append(area1)
    #     if(area1>max_area1):
    #         max_area1=area1
    #         ans=n
    
    # print(ans)

    widths = [node_info[n]['x'] for n in node_names]
    heights = [node_info[n]['y'] for n in node_names]
    degrees = [len(node_to_net_dict[n]) for n in node_names]

    max_area = max(areas)
    max_w = max(widths)
    max_h = max(heights)
    max_deg = max(degrees)

    # print(max_area)

    x = []
    for n in node_names:
        area = (node_info[n]['x'] * node_info[n]['y']) / max_area
        deg = len(node_to_net_dict[n]) / max_deg
        w = node_info[n]['x'] / max_w
        h = node_info[n]['y'] / max_h

        x.append([area, deg, w, h])

    x = torch.tensor(x, dtype=torch.float32, device=device)

    # ---------------- FAST ADJACENCY BUILD ----------------
    adj = torch.zeros((N, N), device=device)

    # Build net → nodes mapping ONCE
    net_to_nodes = {}
    for node, nets in node_to_net_dict.items():
        for net in nets:
            if net not in net_to_nodes:
                net_to_nodes[net] = []
            net_to_nodes[net].append(node)

    # Connect nodes sharing a net
    for net, nodes in net_to_nodes.items():
        idxs = [name_to_idx[n] for n in nodes]

        for i in idxs:
            for j in idxs:
                adj[i, j] = 1.0

    # ---------------- NORMALIZATION ----------------
    deg = adj.sum(dim=1, keepdim=True) + 1e-6
    adj = adj / deg

    return x, adj, node_names
    # return x, adj, node_names