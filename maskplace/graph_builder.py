git import torch
from itertools import combinations

def build_static_graph_tensors(node_info, node_to_net_dict, net_info):
    """
    Converts MaskPlace dictionary data into PyTorch tensors for GNN processing.
    """
    num_nodes = len(node_info)
    
    # ---------------------------------------------------------
    # 1. Node Feature Matrix (X)
    # Shape: [num_nodes, num_features]
    # ---------------------------------------------------------
    num_features = 2 # Starting with [Normalized Area, Normalized Degree]
    X = torch.zeros((num_nodes, num_features), dtype=torch.float)
    
    # Calculate global maximums to prevent exploding gradients
    max_area = max([node['x'] * node['y'] for node in node_info.values()])
    max_degree = max([len(node_to_net_dict[name]) for name in node_info.keys()])
    
    for name, node in node_info.items():
        node_id = node['id']
        area = node['x'] * node['y']
        degree = len(node_to_net_dict[name])
        
        # Populate tensor with normalized values (0.0 to 1.0)
        X[node_id, 0] = area / max_area if max_area > 0 else 0
        X[node_id, 1] = degree / max_degree if max_degree > 0 else 0

    # ---------------------------------------------------------
    # 2. Edge Index Tensor (Adjacency Matrix)
    # Shape: [2, num_edges]
    # ---------------------------------------------------------
    edges = []
    
    for net_name, net_data in net_info.items():
        nodes_in_net = list(net_data['nodes'].keys())
        
        # Clique Expansion: Create an edge between every pair of nodes in a net
        for node_a, node_b in combinations(nodes_in_net, 2):
            id_a = node_info[node_a]['id']
            id_b = node_info[node_b]['id']
            
            # Undirected graph requires edges in both directions
            edges.append([id_a, id_b])
            edges.append([id_b, id_a])
            
    # Convert list of pairs to a [2, E] PyTorch tensor
    # .contiguous() ensures memory is aligned for PyTorch Geometric (PyG)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    
    return X, edge_index

# --- Testing the integration ---
if __name__ == "__main__":
    from place_db import PlaceDB
    
    print("Loading database...")
    placedb = PlaceDB("ariane") # Use the smaller benchmark for testing
    
    X, edge_index = build_static_graph_tensors(
        placedb.node_info, 
        placedb.node_to_net_dict, 
        placedb.net_info
    )
    
    print(f"Node Feature Matrix (X) shape: {X.shape}")
    print(f"Edge Index shape: {edge_index.shape}")