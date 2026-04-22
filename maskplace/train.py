import torch
import torch.optim as optim
import gym
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

# from neural_model import GNNOrderingModel
from pointer_model import PointerOrderingModel
from ordering_policy import sample_ordering
from comp_res import comp_res
from place_db import PlaceDB
from PPO2 import PPO
from graph_utils import build_graph
from torch.distributions import Categorical


def plot_training(hpwl_history, reward_history):

    plt.figure(figsize=(12, 5))

    # ---- HPWL plot ----
    plt.subplot(1, 2, 1)
    plt.plot(hpwl_history)
    plt.title("HPWL over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("HPWL")

    # ---- Reward plot ----
    plt.subplot(1, 2, 2)
    plt.plot(reward_history)
    plt.title("Reward over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Reward")

    plt.tight_layout()
    plt.savefig("training_curve.png")
    plt.show()


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
agent.load_param("/kaggle/working/Macro_Placement_ML_for_EDA/maskplace/model/pretrained_model.pkl")

print("Actor first layer weights:",next(agent.actor_net.parameters())[0][:5])

agent.actor_net.eval()
agent.critic_net.eval()


# -------------------- RUN PLACEMENT --------------------
def run_placement(ordering):

    env.node_name_list = ordering   # ✅ correct

    state = env.reset()
    done = False

    while not done:
        with torch.no_grad():
            action, _ = agent.select_action(state)
        state, reward, done, info = env.step(action)

    hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)

    reward = (-hpwl) / 1e6

    return reward, hpwl, cost



def pretrain(model, placedb, node_info, node_to_net_dict, device, epochs=200):

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    heuristic = placedb.node_id_to_name
    node_names = list(node_info.keys())

    name_to_idx = {n: i for i, n in enumerate(node_names)}
    target = torch.tensor([name_to_idx[n] for n in heuristic], device=device)

    x, adj, _ = build_graph(node_info, node_to_net_dict, device)
    best_loss = float('inf')
    for epoch in range(epochs):

        logits = model.forward_supervised(x, adj, target)

        loss = F.cross_entropy(logits, target)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), "pretrain_best.pth")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"[Pretrain] Epoch {epoch} | Loss: {loss.item():.4f}")

    print("✅ Pretraining complete\n")



# -------------------- TRAINING LOOP --------------------
def train(node_info, node_to_net_dict, epochs=1000):

    hpwl_history = []
    reward_history = []

    # heuristic_ordering = placedb.node_id_to_name.copy()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model = GNNOrderingModel().to(device)
    model = PointerOrderingModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)

    model.load_state_dict(torch.load("pretrain_best.pth"))

    baseline = None
    best_reward = -1e9
    best_hpwl = float('inf')


    for epoch in range(epochs):

        # ---- sample ordering ----
        ordering, log_prob = sample_ordering(
            model,
            node_info,
            node_to_net_dict,
            device
        )
        print("DEBUG ordering id:", id(ordering))
        print("First 5:", ordering[:5])

        # ---- evaluate ----
        reward, hpwl, cost = run_placement(ordering)
        hpwl_history.append(hpwl)
        reward_history.append(reward)



        
        if hpwl < best_hpwl:
            best_hpwl = hpwl
            print("🔥 NEW BEST HPWL:", best_hpwl)

        # ---- baseline ----
        if baseline is None:
            baseline = reward
        else:
            baseline = 0.9 * baseline + 0.1 * reward

        advantage = reward - baseline

        # dist = torch.distributions.Categorical(probs)
        # entropy = dist.entropy().mean()

        # loss = -log_prob * advantage - 0.01 * entropy

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
    plot_training(hpwl_history, reward_history)

    return model


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = PointerOrderingModel().to(device)

    🔥 STEP 1: Imitation
    pretrain(
        model,
        placedb,
        placedb.node_info,
        placedb.node_to_net_dict,
        device,
        epochs=560
    )

    # 🔥 STEP 2: RL fine-tuning
    train(
        placedb.node_info,
        placedb.node_to_net_dict,
        epochs=1000
    )


if __name__ == "__main__":
    main()
    

