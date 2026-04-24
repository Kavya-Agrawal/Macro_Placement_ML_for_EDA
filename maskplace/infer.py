import torch
import gym
import numpy as np

from pointer_model import PointerOrderingModel
from ordering_policy import sample_ordering
from comp_res import comp_res
from place_db import PlaceDB
from PPO2 import PPO


# -------------------- CONFIG --------------------
MODEL_PATH = "/home/pratyush-kumar-swain/Desktop/Macro_Placement_ML_for_EDA/maskplace/gnn_ordering_model_best.pth"
PPO_PATH = "/home/pratyush-kumar-swain/Desktop/Macro_Placement_ML_for_EDA/maskplace/model/pretrained_model.pkl"   # change if needed

BENCHMARK = "adaptec2"
GRID = 224
NUM_SAMPLES = 50   # number of inference runs


# -------------------- SETUP --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

placedb = PlaceDB(BENCHMARK)
placed_num_macro = placedb.node_cnt

env = gym.make(
    'place_env-v0',
    placedb=placedb,
    placed_num_macro=placed_num_macro,
    grid=GRID
).unwrapped


# -------------------- LOAD MODELS --------------------
# ---- Load ordering model ----
model = PointerOrderingModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ---- Load PPO agent ----
agent = PPO()
agent.load_param(PPO_PATH)

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

    return hpwl, cost


# -------------------- INFERENCE --------------------
def inference():
    print(BENCHMARK)
    best_hpwl = float('inf')
    best_ordering = None

    print("\n🚀 Running inference...\n")

    for i in range(NUM_SAMPLES):

        # ---- generate ordering ----
        ordering, _ = sample_ordering(
            model,
            placedb.node_info,
            placedb.node_to_net_dict,
            device
        )

        # print(ordering[:10])

        # ---- evaluate ----
        hpwl, cost = run_placement(ordering)

        print(f"[Run {i}] HPWL: {hpwl:.2f} | Cost: {cost:.2f}")

        if hpwl < best_hpwl:
            best_hpwl = hpwl
            best_ordering = ordering
            print("🔥 NEW BEST!")

    print("\n==============================")
    print("✅ BEST RESULT")
    print("==============================")
    print(f"Best HPWL: {best_hpwl:.2f}")
    print("Ordering sample:", best_ordering[:10])

    return best_ordering, best_hpwl


# -------------------- OPTIONAL: DETERMINISTIC MODE --------------------
def greedy_ordering():

    # monkey patch: replace sampling with argmax
    def greedy_sample(model, node_info, node_to_net_dict, device):
        x, adj, node_names = build_graph(node_info, node_to_net_dict, device)

        embeddings = model.encoder(x, adj)

        N, D = embeddings.shape
        mask = torch.zeros(N, device=device)

        h = torch.zeros(1, D, device=device)
        c = torch.zeros(1, D, device=device)
        inp = embeddings.mean(dim=0, keepdim=True)

        selected = []

        for _ in range(N):
            h, c = model.decoder.lstm(inp, (h, c))

            query = model.decoder.W_q(h)
            keys = model.decoder.W_k(embeddings)

            scores = model.decoder.v(torch.tanh(query + keys)).squeeze(-1)
            scores = scores + mask

            idx = torch.argmax(scores)

            selected.append(idx.item())
            mask[idx] = -1e9

            inp = embeddings[idx].unsqueeze(0)

        ordering = [node_names[i] for i in selected]
        return ordering

    ordering = greedy_sample(
        model,
        placedb.node_info,
        placedb.node_to_net_dict,
        device
    )

    hpwl, cost = run_placement(ordering)

    print("\n🎯 Greedy Result")
    print(f"HPWL: {hpwl:.2f} | Cost: {cost:.2f}")


# -------------------- MAIN --------------------
if __name__ == "__main__":

    best_ordering, best_hpwl = inference()

    # Optional deterministic run
    # greedy_ordering()