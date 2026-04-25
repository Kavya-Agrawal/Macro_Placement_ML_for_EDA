import torch
from itertools import combinations

def build_static_graph_tensors(node_info, node_to_net_dict, net_info):
    num_nodes = len(node_info)
    num_features = 2 
    X = torch.zeros((num_nodes, num_features), dtype=torch.float)
    
    max_area = max([node['x'] * node['y'] for node in node_info.values()])
    max_degree = max([len(node_to_net_dict[name]) for name in node_info.keys()])
    
    for name, node in node_info.items():
        node_id = node['id']
        area = node['x'] * node['y']
        degree = len(node_to_net_dict[name])
        
        X[node_id, 0] = area / max_area if max_area > 0 else 0
        X[node_id, 1] = degree / max_degree if max_degree > 0 else 0

    edges = []
    for net_name, net_data in net_info.items():
        nodes_in_net = list(net_data['nodes'].keys())
        for node_a, node_b in combinations(nodes_in_net, 2):
            id_a = node_info[node_a]['id']
            id_b = node_info[node_b]['id']
            edges.append([id_a, id_b])
            edges.append([id_b, id_a])
            
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return X, edge_index

if __name__ == "__main__":
    from place_db import PlaceDB
    print("Loading database...")
    placedb = PlaceDB("ariane") 
    X, edge_index = build_static_graph_tensors(placedb.node_info, placedb.node_to_net_dict, placedb.net_info)
    print(f"Node Feature Matrix (X) shape: {X.shape}")
    print(f"Edge Index shape: {edge_index.shape}")