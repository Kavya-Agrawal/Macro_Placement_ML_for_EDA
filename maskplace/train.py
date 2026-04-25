import argparse
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import gym

from place_db import PlaceDB
from place_db import get_node_id_to_name_topology
from graph_builder import build_static_graph_tensors
from neural_mcts import SequenceGNN, NeuralMCTS
from comp_res import comp_res

# Import Native MaskPlace PPO
from PPO2 import PPO

# ---------------------------------------------------------
# NATIVE MASKPLACE EVALUATOR (In-Memory)
# ---------------------------------------------------------
class NativeEvaluator:
    def __init__(self, placedb, weights_path, device):
        self.placedb = placedb
        self.device = device
        self.num_macros = len(placedb.node_info)
        
        # Native Gym Environment Initialization
        self.env = gym.make('place_env-v0', 
                            placedb=self.placedb, 
                            placed_num_macro=self.num_macros, 
                            grid=224).unwrapped
                            
        # Native MaskPlace PPO Initialization (Kept in memory to prevent leaks!)
        self.agent = PPO()
        self.agent.load_param(weights_path)
        self.agent.actor_net.to(self.device)
        self.agent.actor_net.eval()

    def get_overlap_area(self, node_pos):
        """Calculates total overlapping area to penalize stacked macros."""
        penalty_area = 0.0
        nodes = list(node_pos.keys())
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                n1, n2 = nodes[i], nodes[j]
                x1, y1 = node_pos[n1][0], node_pos[n1][1]
                w1, h1 = self.placedb.node_info[n1]['x'], self.placedb.node_info[n1]['y']
                x2, y2 = node_pos[n2][0], node_pos[n2][1]
                w2, h2 = self.placedb.node_info[n2]['x'], self.placedb.node_info[n2]['y']
                
                dx = min(x1 + w1, x2 + w2) - max(x1, x2)
                dy = min(y1 + h1, y2 + h2) - max(y1, y2)
                if dx > 0 and dy > 0:
                    penalty_area += (dx * dy)
        return penalty_area

    def get_area_delay_penalty(self, mcts_sequence):
        """Penalizes placing large blocks late in the sequence to prevent tetris-crashing."""
        penalty = 0.0
        id_to_name = {v['id']: k for k, v in self.placedb.node_info.items()}
        for step, macro_id in enumerate(mcts_sequence):
            macro_name = id_to_name[macro_id]
            area = self.placedb.node_info[macro_name]['x'] * self.placedb.node_info[macro_name]['y']
            penalty += (step * area) # Later step = higher multiplier
        return penalty

    def evaluate(self, mcts_sequence):
        id_to_name_map = {v['id']: k for k, v in self.placedb.node_info.items()}
        mapped_sequence = [id_to_name_map[idx] for idx in mcts_sequence]
        
        self.placedb.node_id_to_name = mapped_sequence
        if hasattr(self.env, 'node_name_list'): self.env.node_name_list = mapped_sequence
        if hasattr(self.env, 'node_id_to_name'): self.env.node_id_to_name = mapped_sequence
        
        state = self.env.reset()
        done = False
        
        while not done:
            state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            with torch.inference_mode():
                action_probs, _, _ = self.agent.actor_net(state_tensor)
            best_action = torch.argmax(action_probs).item()
            state, _, done, _ = self.env.step(best_action)
            
        placed_count = len(self.env.node_pos)
        if placed_count < self.num_macros:
            return float('inf'), None
            
        try:
            hpwl, _ = comp_res(self.placedb, self.env.node_pos, self.env.ratio)
            overlap = self.get_overlap_area(self.env.node_pos)
            delay = self.get_area_delay_penalty(mcts_sequence)
            
            # THE SHAPED REWARD (Tune alpha and beta if needed)
            alpha = 5000.0  # Severity of overlapping blocks
            beta = 0.001    # Severity of placing large blocks late
            final_score = hpwl + (alpha * overlap) + (beta * delay)
            
            return final_score, self.env.node_pos
        except Exception as e:
            return float('inf'), None

# ---------------------------------------------------------
# MASTER TRAINING LOOP
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Neural MCTS Training")
    parser.add_argument('--benchmark', type=str, default='ariane')
    parser.add_argument('--weights', type=str, required=True, help="Path to MaskPlace .pkl")
    args, _ = parser.parse_known_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing Native Engine on {device}...")

    placedb = PlaceDB(args.benchmark)
    num_macros = len(placedb.node_info)
    
    X, edge_index = build_static_graph_tensors(placedb.node_info, placedb.node_to_net_dict, placedb.net_info)
    X, edge_index = X.to(device), edge_index.to(device)
    
    mcts_agent = SequenceGNN(num_static_features=2).to(device)
    optimizer = optim.Adam(mcts_agent.parameters(), lr=1e-3, weight_decay=1e-5)
    
    # Initialize the Native Evaluator
    evaluator = NativeEvaluator(placedb, args.weights, device)
    
    print("Establishing Baseline...")
    default_name_seq = get_node_id_to_name_topology(placedb.node_info, placedb.node_to_net_dict, placedb.net_info, args.benchmark)
    name_to_id = {k: v['id'] for k, v in placedb.node_info.items()}
    heuristic_ids = [name_to_id[name] for name in default_name_seq]
    
    BASELINE_SCORE, _ = evaluator.evaluate(heuristic_ids)
    print(f"Dynamic Baseline Score (HPWL + Penalties): {BASELINE_SCORE}")
    
    best_score = float('inf')
    
    # Tracking Lists
    history_epochs, history_score = [], []
    history_policy_loss, history_value_loss = [], []
    
    os.makedirs("outputs", exist_ok=True)
    
    for epoch in range(1000):
        print(f"\n--- Epoch {epoch} ---")
        
        mcts = NeuralMCTS(mcts_agent, X, edge_index, num_macros, num_simulations=30)
        sequence = mcts.get_action_sequence(is_training=True) # Applies Dirichlet Noise
        
        true_score, _ = evaluator.evaluate(sequence)
        
        # Guard against infinity crashing the gradients (Dropped macros)
        if true_score == float('inf'):
            print("⚠️ Episode dropped macros. Heavy penalty applied.")
            true_score = BASELINE_SCORE * 5.0
            
        history_epochs.append(epoch)
        history_score.append(true_score)
        
        if true_score < best_score and true_score != float('inf'):
            best_score = true_score
            torch.save(mcts_agent.state_dict(), f"outputs/best_mcts_agent_{args.benchmark}.pth")
            print(f"⭐ New Best Composite Score: {best_score}!")
            
        advantage = (BASELINE_SCORE - true_score) / BASELINE_SCORE 
        print(f"Score: {true_score:.2f} | Advantage: {advantage:.4f}")
        
        mcts_agent.train()
        optimizer.zero_grad()
        
        sequence_mask = torch.zeros(num_macros, dtype=torch.float).to(device)
        policy_loss, value_loss = 0, 0
        
        for step, macro_id in enumerate(sequence):
            action_probs, predicted_score = mcts_agent(X, edge_index, sequence_mask)
            
            log_prob = torch.log(action_probs[macro_id] + 1e-8)
            policy_loss += -log_prob * advantage 
            
            true_val_tensor = torch.tensor([true_score], dtype=torch.float).to(device)
            value_loss += F.mse_loss(predicted_score, true_val_tensor)
            
            sequence_mask[macro_id] = 1.0 
            
        total_loss = (policy_loss / num_macros) + (value_loss / num_macros)
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(mcts_agent.parameters(), 1.0)
        optimizer.step()
        
        print(f"Loss -> Policy: {policy_loss.item():.4f} | Value: {value_loss.item():.4f}")
        
        history_policy_loss.append(policy_loss.item())
        history_value_loss.append(value_loss.item())

        # ---------------------------------------------------------
        # GENERATE TRAINING METRICS GRAPH (Every 10 epochs)
        # ---------------------------------------------------------
        if epoch % 10 == 0 or epoch == 999:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
            
            # Top Subplot: Composite Score vs Baseline
            ax1.plot(history_epochs, history_score, label='MCTS Composite Score', color='blue')
            ax1.axhline(y=BASELINE_SCORE, color='red', linestyle='--', label='Baseline Score')
            ax1.set_title(f'Neural MCTS Optimization ({args.benchmark})')
            ax1.set_ylabel('Score (Lower is Better)')
            ax1.legend()
            ax1.grid(True, linestyle='--', alpha=0.7)
            
            # Bottom Subplot: Network Losses
            ax2.plot(history_epochs, history_policy_loss, label='Policy Loss', color='purple')
            ax2.plot(history_epochs, history_value_loss, label='Value Loss', color='orange')
            ax2.set_title('Network Loss Tracking')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Loss Value')
            ax2.legend()
            ax2.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(f"outputs/training_metrics_{args.benchmark}.png", dpi=300)
            plt.close()

    # Save final agent when loop finishes
    torch.save(mcts_agent.state_dict(), f"outputs/final_mcts_agent_{args.benchmark}.pth")

if __name__ == "__main__":
    main()