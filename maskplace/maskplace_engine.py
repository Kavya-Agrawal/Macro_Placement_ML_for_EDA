import torch
import gym
import numpy as np
import place_env  
from PPO2 import PPO  
from comp_res import comp_res 

class MaskPlaceEngine:
    def __init__(self, placedb, num_macros, model_weights_path, device=None):
        self.placedb = placedb
        self.num_macros = num_macros
        self.grid = 224
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.env = gym.make('place_env-v0', 
                            placedb=self.placedb, 
                            placed_num_macro=self.num_macros, 
                            grid=self.grid).unwrapped
                            
        self.agent = PPO()
        self.agent.load_param(model_weights_path)
        
        self.agent.actor_net.to(self.device)
        self.agent.critic_net.to(self.device)
        self.agent.actor_net.eval()
        self.agent.critic_net.eval()

    def evaluate_sequence(self, mcts_sequence):
        id_to_name_map = {v['id']: k for k, v in self.placedb.node_info.items()}
        mapped_sequence = [id_to_name_map[idx] for idx in mcts_sequence]
        
        self.placedb.node_id_to_name = mapped_sequence
        if hasattr(self.env, 'node_name_list'):
            self.env.node_name_list = mapped_sequence
        if hasattr(self.env, 'node_id_to_name'):
            self.env.node_id_to_name = mapped_sequence
        
        state = self.env.reset()
        done = False
        
        while not done:
            state_tensor = torch.from_numpy(state).float().to(self.device).unsqueeze(0)
            
            with torch.inference_mode():
                action_probs, _, _ = self.agent.actor_net(state_tensor)
            
            best_action = torch.argmax(action_probs).item()
            next_state, reward, done, info = self.env.step(best_action)
            state = next_state
            
        try:
            hpwl, _ = comp_res(self.placedb, self.env.node_pos, self.env.ratio)
            return hpwl, self.env.node_pos
        except Exception as e:
            print(f"Routing failed: {e}")
            return float('inf'), None