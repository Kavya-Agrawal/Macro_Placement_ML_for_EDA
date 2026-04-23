import sys
import argparse
import os

# ---------------------------------------------------------
# 1. DYNAMIC ARGUMENT PARSING (MCTS Args)
# ---------------------------------------------------------
parser = argparse.ArgumentParser(description="Run Neural MCTS for MaskPlace Sequence Generation")
parser.add_argument('--benchmark', type=str, default='ariane', help='Name of the benchmark')
parser.add_argument('--weights', type=str, required=True, help='Path to the pre-trained MaskPlace .pkl file')

# Parse our arguments FIRST
mcts_args, _ = parser.parse_known_args()

# ---------------------------------------------------------
# 2. THE SYS.ARGV HIJACK
# ---------------------------------------------------------
sys.argv = [sys.argv[0], '--benchmark', mcts_args.benchmark, '--is_test']

# ---------------------------------------------------------
# 3. SAFE IMPORTS
# ---------------------------------------------------------
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt  # [NEW] For plotting

from place_db import PlaceDB
from place_db import get_node_id_to_name_topology
from graph_builder import build_static_graph_tensors
from neural_mcts import SequenceGNN, NeuralMCTS
from maskplace_engine import MaskPlaceEngine

def main():
    benchmark = mcts_args.benchmark
    weights_path = mcts_args.weights
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading database for {benchmark}...")
    placedb = PlaceDB(benchmark)
    num_macros = len(placedb.node_info)
    
    X, edge_index = build_static_graph_tensors(placedb.node_info, placedb.node_to_net_dict, placedb.net_info)
    X = X.to(device)
    edge_index = edge_index.to(device)
    
    mcts_agent = SequenceGNN(num_static_features=2).to(device)
    optimizer = optim.Adam(mcts_agent.parameters(), lr=1e-3)
    
    maskplace_engine = MaskPlaceEngine(placedb, num_macros, model_weights_path=weights_path, device=device)
    
    print("Calculating dynamic baseline HPWL using MaskPlace default heuristic...")
    default_name_sequence = get_node_id_to_name_topology(
        placedb.node_info, 
        placedb.node_to_net_dict, 
        placedb.net_info, 
        benchmark
    )
    
    name_to_id_map = {k: v['id'] for k, v in placedb.node_info.items()}
    heuristic_sequence_ids = [name_to_id_map[name] for name in default_name_sequence]
    
    BASELINE_HPWL = maskplace_engine.evaluate_sequence(heuristic_sequence_ids)
    print(f"Dynamic Baseline HPWL established: {BASELINE_HPWL}")
    
    # ---------------------------------------------------------
    # [NEW] TRACKING VARIABLES FOR KAGGLE
    # ---------------------------------------------------------
    best_hpwl = float('inf')
    history_epochs = []
    history_hpwl = []
    os.makedirs("outputs", exist_ok=True) # Directory to save weights and plots
    
    epochs = 1000
    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch} ---")
        
        # --- A. Generation Phase ---
        mcts = NeuralMCTS(mcts_agent, X, edge_index, num_macros, num_simulations=20)
        sequence = mcts.get_action_sequence()
        
        # --- B. Evaluation Phase ---
        true_hpwl = maskplace_engine.evaluate_sequence(sequence)
        
        # --- C. Tracking and Checkpointing ---
        history_epochs.append(epoch)
        history_hpwl.append(true_hpwl)
        
        if true_hpwl < best_hpwl:
            best_hpwl = true_hpwl
            save_path = f"outputs/best_mcts_agent_{benchmark}.pth"
            print(f"⭐ New Best HPWL: {best_hpwl}! Saving model weights to {save_path}...")
            torch.save(mcts_agent.state_dict(), save_path)
            
            # Optionally save the winning sequence to a text file
            with open(f"outputs/best_sequence_{benchmark}.txt", "w") as f:
                f.write(",".join(map(str, sequence)))
        
        advantage = (BASELINE_HPWL - true_hpwl) / BASELINE_HPWL 
        print(f"True HPWL: {true_hpwl} | Advantage: {advantage:.4f}")
        
        # --- D. Plotting Progress (Every 10 epochs) ---
        if epoch % 10 == 0 or epoch == epochs - 1:
            plt.figure(figsize=(10, 5))
            plt.plot(history_epochs, history_hpwl, label='MCTS Sequence HPWL', color='blue')
            plt.axhline(y=BASELINE_HPWL, color='red', linestyle='--', label='MaskPlace Baseline HPWL')
            
            plt.title(f'Neural MCTS Optimization Progress ({benchmark})')
            plt.xlabel('Epoch')
            plt.ylabel('Final HPWL (Lower is Better)')
            plt.legend()
            plt.grid(True)
            
            plot_path = f"outputs/training_curve_{benchmark}.png"
            plt.savefig(plot_path)
            plt.close() # Close to free up memory
            print(f"Updated training plot saved to {plot_path}")
        
        # --- E. Backpropagation ---
        mcts_agent.train()
        optimizer.zero_grad()
        
        sequence_mask = torch.zeros(num_macros, dtype=torch.float).to(device)
        policy_loss = 0
        value_loss = 0
        
        for step, macro_id in enumerate(sequence):
            action_probs, predicted_hpwl = mcts_agent(X, edge_index, sequence_mask)
            
            log_prob = torch.log(action_probs[macro_id] + 1e-8)
            policy_loss += -log_prob * advantage 
            
            true_value_tensor = torch.tensor([true_hpwl], dtype=torch.float).to(device)
            value_loss += F.mse_loss(predicted_hpwl, true_value_tensor)
            
            sequence_mask[macro_id] = 1.0 
            
        total_loss = (policy_loss / num_macros) + (value_loss / num_macros)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(mcts_agent.parameters(), 1.0)
        optimizer.step()
        
        print(f"Loss -> Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f}")

    final_save_path = f"outputs/final_mcts_agent_{benchmark}.pth"
    print(f"\nTraining completed. Saving final weights to {final_save_path}...")
    torch.save(mcts_agent.state_dict(), final_save_path)

if __name__ == "__main__":
    main()