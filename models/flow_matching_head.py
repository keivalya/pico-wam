"""Flow-matching action head for pico-WAM.

Learns to transport Gaussian noise into expert actions.

Training:
    x0 ~ N(0, I)
    x1 = expert action
    t  ~ Uniform(0, 1)

    xt = (1 - t) * x0 + t * x1
    target_velocity = x1 - x0

    model predicts velocity v_theta(xt, t, cond)

Inference:
    start from Gaussian noise
    integrate dx/dt = v_theta(x, t, cond)
    return final action
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: (B,) continuous time in [0, 1]
        returns: (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -torch.log(torch.tensor(10000.0, device=device))
            * torch.arange(half_dim, device=device).float()
            / max(half_dim - 1, 1)
        )

        args = t[:, None] * freqs[None, :]

        emb = torch.cat(
            [torch.sin(args), torch.cos(args)],
            dim=-1,
        )

        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))

        return emb


class FlowMatchingActionHead(nn.Module):
    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 2,
        num_inference_steps: int = 8,
    ):
        super().__init__()

        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.num_inference_steps = num_inference_steps

        self.time_embed = SinusoidalTimeEmbedding(time_dim)

        layers, width = [], action_dim + cond_dim + time_dim
        for _ in range(num_layers):
            layers += [nn.Linear(width, hidden_dim), nn.ReLU()]
            width = hidden_dim
        self.net = nn.Sequential(*layers, nn.Linear(width, action_dim))

    def velocity(self, x_t, t, cond):
        """
        x_t:  (B, action_dim)
        t:    (B,)
        cond: (B, cond_dim)
        """
        t_emb = self.time_embed(t)
        inp = torch.cat([x_t, cond, t_emb], dim=-1)
        return self.net(inp)

    def loss(self, actions, cond):
        """
        Flow-matching training loss.

        actions: (B, action_dim)
        cond:    (B, cond_dim)
        """
        batch_size = actions.shape[0]

        x0 = torch.randn_like(actions)
        x1 = actions

        t = torch.rand(
            batch_size,
            device=actions.device,
            dtype=actions.dtype,
        )

        x_t = (1.0 - t[:, None]) * x0 + t[:, None] * x1

        target_velocity = x1 - x0
        pred_velocity = self.velocity(x_t, t, cond)

        return F.mse_loss(pred_velocity, target_velocity)

    @torch.no_grad()
    def sample(self, cond):
        """
        Euler integration from noise to action.

        cond: (B, cond_dim)
        returns: (B, action_dim)
        """
        batch_size = cond.shape[0]
        device = cond.device

        x = torch.randn(
            batch_size,
            self.action_dim,
            device=device,
        )

        dt = 1.0 / self.num_inference_steps

        for i in range(self.num_inference_steps):
            t_value = i / self.num_inference_steps
            t = torch.full(
                (batch_size,),
                t_value,
                device=device,
            )

            v = self.velocity(x, t, cond)
            x = x + dt * v

        return x