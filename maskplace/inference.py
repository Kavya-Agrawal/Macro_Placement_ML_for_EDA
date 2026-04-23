import sys
import argparse
import time
import os

# ---------------------------------------------------------
# 1. DYNAMIC ARGUMENT PARSING (Inference Args)
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Inference for Neural MCTS MaskPlace")
parser.add_argument('--benchmark', type=str, default='ariane', help='Benchmark name')
parser.add_argument('--mcts_weights', type=str, required=True, help='Path to your trained MCTS .pth file')
parser.add_argument('--maskplace_weights', type=str, required=True, help='Path to MaskPlace .pkl file')
parser.add_argument('--simulations', type=int, default=50, help='Number of MCTS rollouts per step')
# We add the flag here so you can toggle it from Kaggle
parser.add_argument('--save_fig', action='store_true', help='Trigger MaskPlace native image generation')

inference_args, _ = parser.parse_known_args()

# ---------------------------------------------------------
# 2. THE SYS.ARGV HIJACK (The Native Trigger)
# ---------------------------------------------------------
# We build the fake command line for PPO2.py. 
hijacked_args = [sys.argv[0], '--benchmark', inference_args.benchmark, '--is_test']

# If you pass --save_fig to our script, we pass it down to MaskPlace!
if inference_args.save_fig:
    hijacked_args.append('--save_fig')

sys.argv = hijacked_args

# ---------------------------------------------------------
# 3. SAFE IMPORTS
# ---------------------------------------------------------
import torch
from place_db import PlaceDB
from graph_builder import build_static_graph_tensors
from neural_mcts import SequenceGNN, NeuralMCTS
from maskplace_engine import MaskPlaceEngine

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on {device}...")

    placedb = PlaceDB(inference_args.benchmark)
    num_macros = len(placedb.node_info)
    
    X, edge_index = build_static_graph_tensors(placedb.node_info, placedb.node_to_net_dict, placedb.net_info)
    X, edge_index = X.to(device), edge_index.to(device)
    
    mcts_agent = SequenceGNN(num_static_features=2).to(device)
    mcts_agent.load_state_dict(torch.load(inference_args.mcts_weights, map_location=device))
    mcts_agent.eval() 
    
    maskplace_engine = MaskPlaceEngine(
        placedb, 
        num_macros, 
        model_weights_path=inference_args.maskplace_weights, 
        device=device
    )
    
    print("\nStarting Neural MCTS Sequence Generation...")
    start_time = time.time()
    
    mcts = NeuralMCTS(mcts_agent, X, edge_index, num_macros, num_simulations=inference_args.simulations)
    
    with torch.inference_mode(): 
        best_sequence = mcts.get_action_sequence()
        
    mcts_time = time.time() - start_time
    print(f"Sequence generated in {mcts_time:.2f} seconds.")
    
    print("\nRunning MaskPlace Engine to generate final layout...")
    # Because we injected --save_fig, MaskPlace will automatically handle the drawing internally!
    final_hpwl = maskplace_engine.evaluate_sequence(best_sequence)
    
    total_time = time.time() - start_time
    print(f"\n✅ Inference Complete!")
    print(f"Final HPWL: {final_hpwl}")
    print(f"Total Inference Time: {total_time:.2f} seconds.")
    
    if inference_args.save_fig:
        print("\n🎨 Check the MaskPlace default output directory (usually 'fig/' or root) for your .png file!")

if __name__ == "__main__":
    main()