import torch
import torch.optim as optim
import gym
import numpy as np

# from neural_model import GNNOrderingModel
from pointer_model import PointerOrderingModel
from ordering_policy import sample_ordering
from comp_res import comp_res
from place_db import PlaceDB
from PPO2 import PPO

benchmark = "adaptec1"
placedb = PlaceDB(benchmark)

heuristic_ordering = placedb.node_id_to_name.copy()
print(heuristic_ordering)