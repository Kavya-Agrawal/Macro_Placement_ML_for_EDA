import argparse
import time
import sys
import torch

from place_db import PlaceDB
from graph_builder import build_static_graph_tensors
from neural_mcts import SequenceGNN, NeuralMCTS

def main():
    parser = argparse.ArgumentParser(description="MCTS Generator for MaskPlace")
    parser.add_argument('--benchmark', type=str, default='ariane')
    parser.add_argument('--mcts_weights', type=str, required=True, help='Path to trained MCTS .pth file')
    parser.add_argument('--simulations', type=int, default=50)
    args, _ = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running MCTS Generator on {device}...")

    # 1. Load Database & Build Tensors
    placedb = PlaceDB(args.benchmark)
    num_macros = len(placedb.node_info)
    
    X, edge_index = build_static_graph_tensors(placedb.node_info, placedb.node_to_net_dict, placedb.net_info)
    X, edge_index = X.to(device), edge_index.to(device)
    
    # 2. Initialize Trained MCTS Agent
    mcts_agent = SequenceGNN(num_static_features=2).to(device)
    mcts_agent.load_state_dict(torch.load(args.mcts_weights, map_location=device))
    mcts_agent.eval() 
    
    print("\nStarting Neural MCTS Sequence Generation...")
    start_time = time.time()
    
    mcts = NeuralMCTS(mcts_agent, X, edge_index, num_macros, num_simulations=args.simulations)
    
    with torch.inference_mode(): 
        best_sequence_ids = mcts.get_action_sequence(is_training=False)
        
    print(f"Sequence generated in {time.time() - start_time:.2f} seconds.")
    
    # 3. Map IDs back to Names
    id_to_name_map = {v['id']: k for k, v in placedb.node_info.items()}
    print(best_sequence_ids)
    print(id_to_name_map)
    mapped_sequence = [id_to_name_map[idx] for idx in best_sequence_ids]
    print(mapped_sequence)
    # ---------------------------------------------------------
    # 4. IN-MEMORY HANDOFF
    # ---------------------------------------------------------
    # Inject the generated sequence directly into the MaskPlace class
    PlaceDB.custom_mcts_sequence = mapped_sequence
    
    print("\n✅ Sequence injected into memory. Handing control over to original MaskPlace codebase...")

    # 5. Trigger Native MaskPlace Execution
    # We fake the command line arguments so PPO2 knows what to do
    sys.argv = [sys.argv[0], '--benchmark', args.benchmark, '--is_test', '--save_fig']
    
    # Importing PPO2 triggers its global execution logic natively
    import PPO2
    
    # NOTE: If your specific version of MaskPlace wraps PPO2 execution in a main() function, 
    # uncomment the line below. Otherwise, the import alone will run it!
    PPO2.main()

if __name__ == "__main__":
    main()