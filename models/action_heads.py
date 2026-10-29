"""Action heads for pico-WAM.

Every head maps the fused latent world state z_t to an action a_t and
exposes the same two methods:

    loss(actions, cond) -> scalar training loss
    sample(cond)        -> (B, action_dim) actions

so the WAM never needs to know which head it is talking to.

    mlp       : regress a_t directly (MSE)
    diffusion : DDPM denoising over actions
    flow      : flow matching from noise to actions
"""

import torch.nn as nn
import torch.nn.functional as F

from .diffusion_head import DiffusionConfig, DiffusionPolicyHead
from .flow_matching_head import FlowMatchingActionHead


class MLPActionHead(nn.Module):
    def __init__(self, action_dim, cond_dim, hidden_dim=128, num_layers=1):
        super().__init__()
        layers, width = [], cond_dim
        for _ in range(num_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        self.net = nn.Sequential(*layers, nn.Linear(width, action_dim))

    def loss(self, actions, cond):
        return F.mse_loss(self.net(cond), actions)

    def sample(self, cond):
        return self.net(cond)


def build_action_head(name, action_dim, cond_dim, diffusion_T=16, flow_steps=8, hidden_dim=None, num_layers=None):
    """hidden_dim defaults to cond_dim; num_layers (hidden layers) to 1 for mlp, 2 for flow.
    The diffusion head keeps its own fixed network."""
    hidden_dim = hidden_dim or cond_dim

    if name == "mlp":
        return MLPActionHead(action_dim, cond_dim, hidden_dim=hidden_dim, num_layers=num_layers or 1)

    if name == "diffusion":
        cfg = DiffusionConfig(T=diffusion_T, action_dim=action_dim, cond_dim=cond_dim)
        return DiffusionPolicyHead(cfg)

    if name == "flow":
        return FlowMatchingActionHead(
            action_dim=action_dim,
            cond_dim=cond_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers or 2,
            num_inference_steps=flow_steps,
        )

    raise ValueError(f"Unknown action_head: {name}")
