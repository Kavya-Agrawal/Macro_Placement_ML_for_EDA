# import torch
# import gym
# import numpy as np

# from pointer_model import PointerOrderingModel
# from ordering_policy import sample_ordering
# from comp_res import comp_res
# from place_db import PlaceDB
# from PPO2 import PPO


# # -------------------- CONFIG --------------------
# MODEL_PATH = "/kaggle/working/Macro_Placement_ML_for_EDA/maskplace/model_best_adaptec4.pth"
# PPO_PATH = "/kaggle/working/Macro_Placement_ML_for_EDA/maskplace/model/pretrained_model.pkl"   # change if needed

# BENCHMARK = "adaptec4"
# GRID = 224
# NUM_SAMPLES = 50   # number of inference runs


# # -------------------- SETUP --------------------
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# placedb = PlaceDB(BENCHMARK)
# placed_num_macro = placedb.node_cnt

# env = gym.make(
#     'place_env-v0',
#     placedb=placedb,
#     placed_num_macro=placed_num_macro,
#     grid=GRID
# ).unwrapped


# # -------------------- LOAD MODELS --------------------
# # ---- Load ordering model ----
# model = PointerOrderingModel().to(device)
# model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
# model.eval()

# # ---- Load PPO agent ----
# agent = PPO()
# agent.load_param(PPO_PATH)

# agent.actor_net.eval()
# agent.critic_net.eval()


# # -------------------- RUN PLACEMENT --------------------
# def run_placement(ordering):

#     env.node_id_to_name = ordering

#     state = env.reset()
#     done = False

#     while not done:
#         with torch.no_grad():
#             action, _ = agent.select_action(state)
#         state, reward, done, info = env.step(action)

#     hpwl, cost = comp_res(placedb, env.node_pos, env.ratio)

#     return hpwl, cost


# # -------------------- INFERENCE --------------------
# def inference():
#     print(BENCHMARK)
#     best_hpwl = float('inf')
#     best_ordering = None

#     print("\n🚀 Running inference...\n")

#     for i in range(NUM_SAMPLES):

#         # ---- generate ordering ----
#         ordering, _ = sample_ordering(
#             model,
#             placedb.node_info,
#             placedb.node_to_net_dict,
#             device
#         )

#         # print(ordering[:10])

#         # ---- evaluate ----
#         hpwl, cost = run_placement(ordering)

#         print(f"[Run {i}] HPWL: {hpwl:.2f} | Cost: {cost:.2f}")

#         if hpwl < best_hpwl:
#             best_hpwl = hpwl
#             best_ordering = ordering
#             print("🔥 NEW BEST!")

#     print("\n==============================")
#     print("✅ BEST RESULT")
#     print("==============================")
#     print(f"Best HPWL: {best_hpwl:.2f}")
#     print("Ordering sample:", best_ordering[:10])

#     return best_ordering, best_hpwl


# # -------------------- OPTIONAL: DETERMINISTIC MODE --------------------
# def greedy_ordering():

#     # monkey patch: replace sampling with argmax
#     def greedy_sample(model, node_info, node_to_net_dict, device):
#         x, adj, node_names = build_graph(node_info, node_to_net_dict, device)

#         embeddings = model.encoder(x, adj)

#         N, D = embeddings.shape
#         mask = torch.zeros(N, device=device)

#         h = torch.zeros(1, D, device=device)
#         c = torch.zeros(1, D, device=device)
#         inp = embeddings.mean(dim=0, keepdim=True)

#         selected = []

#         for _ in range(N):
#             h, c = model.decoder.lstm(inp, (h, c))

#             query = model.decoder.W_q(h)
#             keys = model.decoder.W_k(embeddings)

#             scores = model.decoder.v(torch.tanh(query + keys)).squeeze(-1)
#             scores = scores + mask

#             idx = torch.argmax(scores)

#             selected.append(idx.item())
#             mask[idx] = -1e9

#             inp = embeddings[idx].unsqueeze(0)

#         ordering = [node_names[i] for i in selected]
#         return ordering

#     ordering = greedy_sample(
#         model,
#         placedb.node_info,
#         placedb.node_to_net_dict,
#         device
#     )

#     hpwl, cost = run_placement(ordering)

#     print("\n🎯 Greedy Result")
#     print(f"HPWL: {hpwl:.2f} | Cost: {cost:.2f}")


# # -------------------- MAIN --------------------
# if __name__ == "__main__":

#     best_ordering, best_hpwl = inference()

#     # Optional deterministic run
#     # greedy_ordering()

import torch
import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pointer_model import PointerOrderingModel
from ordering_policy import sample_ordering
from comp_res import comp_res
from place_db import PlaceDB
from PPO2 import PPO


# -------------------- CONFIG --------------------
MODEL_PATH = "/kaggle/working/Macro_Placement_ML_for_EDA/maskplace/model_best_adaptec4.pth"
PPO_PATH = "/kaggle/working/Macro_Placement_ML_for_EDA/maskplace/model/pretrained_model.pkl"

BENCHMARK = "adaptec4"
GRID = 224
NUM_SAMPLES = 50


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
model = PointerOrderingModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

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

    return hpwl, cost, env.node_pos, env.ratio


# -------------------- SAVE .PL FILE --------------------
def save_placement(file_path, node_pos, ratio, placedb):

    with open(file_path, 'w') as fwrite:

        node_place = {}

        for node_name in node_pos:
            x, y, _, _ = node_pos[node_name]

            x = round(x * ratio + ratio)
            y = round(y * ratio + ratio)

            node_place[node_name] = (x, y)

        print("len node_place:", len(node_place), "/", placedb.node_cnt)

        for node_name in placedb.node_info:
            if node_name not in node_place:
                continue

            x, y = node_place[node_name]
            fwrite.write(f"{node_name}\t{x}\t{y}\t:\tN /FIXED\n")

    print(f"✅ .pl saved to {file_path}")


# -------------------- SAVE FIGURE --------------------
def save_fig(file_path, node_pos, grid):

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    for node_name in node_pos:
        x, y, size_x, size_y = node_pos[node_name]

        rect = patches.Rectangle(
            (x / grid, y / grid),
            size_x / grid,
            size_y / grid,
            linewidth=1,
            edgecolor='black',
            facecolor='skyblue'
        )
        ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(file_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"📸 Placement image saved to {file_path}")


# -------------------- INFERENCE --------------------
def inference():

    print(f"\n📦 Benchmark: {BENCHMARK}")
    print("\n🚀 Running inference...\n")

    best_hpwl = float('inf')
    best_ordering = None
    best_node_pos = None
    best_ratio = None

    for i in range(NUM_SAMPLES):

        # ---- generate ordering ----
        ordering, _ = sample_ordering(
            model,
            placedb.node_info,
            placedb.node_to_net_dict,
            device
        )

        # ---- evaluate ----
        hpwl, cost, node_pos, ratio = run_placement(ordering)

        print(f"[Run {i}] HPWL: {hpwl:.2f} | Cost: {cost:.2f}")

        if hpwl < best_hpwl:
            best_hpwl = hpwl
            best_ordering = ordering
            best_node_pos = node_pos.copy()
            best_ratio = ratio

            print("🔥 NEW BEST!")

    print("\n==============================")
    print("✅ BEST RESULT")
    print("==============================")
    print(f"Best HPWL: {best_hpwl:.2f}")
    print("Ordering sample:", best_ordering[:10])

    return best_ordering, best_hpwl, best_node_pos, best_ratio


# -------------------- MAIN --------------------
if __name__ == "__main__":

    best_ordering, best_hpwl, best_node_pos, best_ratio = inference()

    # ---- Save .pl ----
    save_placement(
        f"{BENCHMARK}_best.pl",
        best_node_pos,
        best_ratio,
        placedb
    )

    # ---- Save image ----
    save_fig(
        f"{BENCHMARK}_placement.png",
        best_node_pos,
        GRID
    )
