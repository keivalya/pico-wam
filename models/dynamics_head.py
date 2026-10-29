"""World-model heads for pico-WAM: latent dynamics + state readout.

The WAM keeps a latent world state z_t (the fused observation token).

    LatentDynamicsMLP : z_t, a_t  -> z_t+1       "what will my action do?"
    StateReadout      : z         -> s           "what does z say about the world?"

Because the dynamics model lives entirely in latent space, the WAM can
"imagine" many steps ahead without rendering a single image:

    z_t  --a_t-->  z_t+1  --a_t+1-->  z_t+2  ...

and every imagined latent can be read out as a state (StateReadout) or as
an image (ImageDecoder in image_decoder.py). That is what makes the inner
world model visible.
"""

import torch
import torch.nn as nn


class LatentDynamicsMLP(nn.Module):
    """
    Action-conditioned residual dynamics in latent space:

        z_t+1 = LayerNorm(z_t + f([z_t, a_t]))

    The residual form means "the world mostly stays the same; predict the change".
    The LayerNorm keeps imagined latents on the same scale as encoded ones
    (the fusion module also ends in a LayerNorm), so long rollouts do not drift
    off to infinity.
    """

    def __init__(self, d_model: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(d_model + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, z, action):
        delta = self.net(torch.cat([z, action], dim=-1))
        return self.ln(z + delta)

    def rollout(self, z, actions):
        """
        Open-loop imagination.

        z:       (B, d_model)        starting latent
        actions: (B, H, action_dim)  action sequence
        returns: (B, H, d_model)     imagined latents z_t+1 ... z_t+H
        """
        zs = []
        for k in range(actions.shape[1]):
            z = self(z, actions[:, k])
            zs.append(z)
        return torch.stack(zs, dim=1)


class StateReadout(nn.Module):
    """
    Reads the physical state back out of a latent.

    linear=True (default) makes this a *linear probe*: if a linear map can
    recover agent position, velocity, object pose, ... from z, then those
    quantities are explicitly (linearly) encoded in the latent space. That is
    the simplest useful definition of an "interpretable" world model, and the
    weights of this layer tell you which latent directions mean what.
    """

    def __init__(self, d_model: int, state_dim: int, linear: bool = True):
        super().__init__()

        if linear:
            self.net = nn.Linear(d_model, state_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, state_dim),
            )

    def forward(self, z):
        return self.net(z)
