import torch
import torch.optim as optim
import gym
import numpy as np

from neural_model import GNNOrderingModel
from ordering_policy import sample_ordering
from comp_res import comp_res
from place_db import PlaceDB
from PPO2 import PPO


# -------------------- SETUP --------------------
benchmark = "adaptec1"
placedb = PlaceDB(benchmark)

placed_num_macro = 543   # MUST match PPO training
grid = 224

env = gym.make(
    'place_env-v0',
    placedb=placedb,
    placed_num_macro=placed_num_macro,
    grid=grid
).unwrapped


# -------------------- LOAD PPO --------------------
agent = PPO()
agent.load_param("/kaggle/working/Macro_Placement_ML_for_EDA/maskplace/train.py")

agent.actor_net.eval()
agent.critic_net.eval()


# -------------------- RUN PLACEMENT --------------------
def run_placement(ordering):

    env.node_id_to_name = ordering

    state = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action, _ = agent.select_action(state)
        state, reward, done, info = env.step(action)

    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)

    # normalize (VERY IMPORTANT)
    reward = (-hpwl - cost) / 1e6

    return reward, hpwl, cost


# -------------------- TRAINING LOOP --------------------
def train(node_info, node_to_net_dict, epochs=1000):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GNNOrderingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    baseline = None
    best_reward = -1e9

    for epoch in range(epochs):

        # ---- sample ordering ----
        ordering, log_prob = sample_ordering(
            model,
            node_info,
            node_to_net_dict,
            device
        )

        # ---- evaluate ----
        reward, hpwl, cost = run_placement(ordering)

        # ---- baseline ----
        if baseline is None:
            baseline = reward
        else:
            baseline = 0.9 * baseline + 0.1 * reward

        advantage = reward - baseline

        # ---- loss ----
        loss = -log_prob * advantage

        optimizer.zero_grad()
        loss.backward()

        # ---- gradient clipping (VERY IMPORTANT) ----
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        # ---- track best ----
        if reward > best_reward:
            best_reward = reward
            torch.save(model.state_dict(), "gnn_ordering_model_best.pth")

        # ---- DEBUG LOGGING ----
        if epoch % 1 == 0:
            print("="*60)
            print(f"Epoch: {epoch}")
            print(f"Reward: {reward:.4f}")
            print(f"HPWL: {hpwl:.2f}")
            print(f"Cost: {cost:.2f}")
            print(f"Baseline: {baseline:.4f}")
            print(f"Advantage: {advantage:.4f}")
            print(f"Loss: {loss.item():.4f}")
            print(f"LogProb: {log_prob.item():.4f}")
            print(f"Ordering sample: {ordering[:5]}")
            print("="*60)

    torch.save(model.state_dict(), "gnn_ordering_model_final.pth")

    return model


# -------------------- MAIN --------------------
if __name__ == "__main__":
    train(
        placedb.node_info,
        placedb.node_to_net_dict,
        epochs=1000
    )
