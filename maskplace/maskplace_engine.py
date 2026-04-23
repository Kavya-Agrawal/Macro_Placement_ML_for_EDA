import torch
import gym
import numpy as np
import place_env  
from PPO2 import PPO  
from comp_res import comp_res 

class MaskPlaceEngine:
    def __init__(self, placedb, num_macros, model_weights_path, device=None):
        """
        Initializes the frozen MaskPlace RL environment to be used as a reward engine.
        """
        self.placedb = placedb
        self.num_macros = num_macros
        self.grid = 224
        
        # Inherit device from train.py, or auto-detect
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.env = gym.make('place_env-v0', 
                            placedb=self.placedb, 
                            placed_num_macro=self.num_macros, 
                            grid=self.grid).unwrapped
                            
        self.agent = PPO()
        self.agent.load_param(model_weights_path)
        
        # Force the original networks onto the correct device
        self.agent.actor_net.to(self.device)
        self.agent.critic_net.to(self.device)
        
        self.agent.actor_net.eval()
        self.agent.critic_net.eval()

    def evaluate_sequence(self, mcts_sequence):
        """
        Forces the environment to place macros in the exact order dictated by MCTS.
        Returns the final Half-Perimeter Wirelength (HPWL).
        """
        # 1. Map your integer IDs back to the string names MaskPlace uses
        id_to_name_map = {v['id']: k for k, v in self.placedb.node_info.items()}
        mapped_sequence = [id_to_name_map[idx] for idx in mcts_sequence]
        
        # 2. INJECTION (THE CRITICAL FIX): 
        # Because we set node_id_to_name to None in place_db.py, we MUST populate
        # the database and the environment with our custom sequence before calling reset().
        self.placedb.node_id_to_name = mapped_sequence
        
        # Depending on the exact MaskPlace version, the environment uses one of these two variables
        if hasattr(self.env, 'node_name_list'):
            self.env.node_name_list = mapped_sequence
        if hasattr(self.env, 'node_id_to_name'):
            self.env.node_id_to_name = mapped_sequence
        
        # 3. Start the placement episode
        state = self.env.reset()
        done = False
        
        while not done:
            state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            
            # Forward pass through the frozen MaskPlace network
            with torch.inference_mode():
                action_probs, _, _ = self.agent.actor_net(state_tensor)
            
            # In inference, we take the absolute best move (argmax), no random sampling
            best_action = torch.argmax(action_probs).item()
            
            next_state, reward, done, info = self.env.step(best_action)
            state = next_state
            
        # 4. Episode is done. Calculate and return the final HPWL.
        try:
            hpwl, _ = comp_res(self.placedb, self.env.node_pos, self.env.ratio)
            return hpwl
        except Exception as e:
            print(f"Routing failed: {e}")
            # Return an artificially high penalty if the sequence causes an invalid layout
            return float('inf')