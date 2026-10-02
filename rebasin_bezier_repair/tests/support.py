import torch
from torch import nn


def mlp_case(seed=0):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(4, 8), nn.ReLU(),
            nn.Linear(8, 6), nn.ReLU(), nn.Linear(6, 3),
        )
        x = torch.randn(24, 4)
        y = (x @ torch.randn(4, 3)).argmax(1)
    return model, [(x, y)]


def copy_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}
