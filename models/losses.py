"""Training losses for pico-WAM (and its mini-VLA baseline).

VLA:
    L = w_a * L_action

WAM:
    L = w_a * L_action                                    act
      + w_s * (L_state_recon + L_state_pred)              know the state now / later
      + w_i * (L_image_recon + L_image_pred)              see the world now / later
      + w_z * L_latent                                    imagine consistently

where, for an H-step window starting at time t,

    z_t          = encode(o_t)
    z^_t+k       = dynamics(z^_t+k-1, a_t+k-1)        (imagined, k = 1..H, z^_t = z_t)
    L_state_pred = mean_k || readout(z^_t+k) - s_t+k ||^2
    L_image_pred = mean_k || decode(z^_t+k) - img_t+k ||^2
    L_latent     = mean_k || z^_t+k - stopgrad(encode(o_t+k)) ||^2

The recon terms anchor z_t to the real world (and stop the latent from
collapsing to a constant); the pred terms teach the dynamics what actions
do; the latent term keeps imagined latents "in distribution", so the
action head and decoders still make sense on them.

All state losses are computed in normalized state space (see
PicoWAMPolicy.normalize_state), so every state dimension counts equally.
"""

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    action: float = 1.0
    state: float = 0.5
    image: float = 1.0
    latent: float = 1.0
    image_focus: float = 0.0   # extra weight on pixels that differ from the mean image: the T and the agent


def image_loss(pred, target, mean_image, focus):
    """
    Pixel MSE. With focus > 0, pixels where the target differs from the mean image,
    the objects that move, count 1 + focus times as much as the static background,
    so a small T on a white table is not drawn as a blur to save background error.
    """
    if focus <= 0:
        return F.mse_loss(pred, target)
    moving = ((target - mean_image).abs().amax(dim=-3, keepdim=True) > 0.1).float()
    w = 1.0 + focus * moving
    return ((pred - target) ** 2 * w).sum() / (w.sum() * pred.shape[-3])


def compute_wam_loss(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], model, weights: LossWeights):
    logs = {}

    action_loss = outputs["action_loss"]
    logs["action_loss"] = action_loss.detach()
    total = weights.action * action_loss

    if "state_recon" in outputs:
        target_now = model.normalize_state(batch["state"])
        target_future = model.normalize_state(batch["future_states"])

        recon = F.mse_loss(outputs["state_recon"], target_now)
        pred = F.mse_loss(outputs["state_pred"], target_future)

        logs["state_recon_loss"] = recon.detach()
        logs["state_pred_loss"] = pred.detach()
        total = total + weights.state * (recon + pred)

    if "image_recon" in outputs:
        mean = model.image_decoder.mean_image
        recon = image_loss(outputs["image_recon"], batch["image"], mean, weights.image_focus)
        pred = image_loss(outputs["image_pred"], batch["future_images"], mean, weights.image_focus)

        logs["image_recon_loss"] = recon.detach()
        logs["image_pred_loss"] = pred.detach()
        total = total + weights.image * (recon + pred)

    if "z_pred" in outputs:
        latent = F.mse_loss(outputs["z_pred"], outputs["z_target"].detach())
        logs["latent_loss"] = latent.detach()
        total = total + weights.latent * latent

    logs["total_loss"] = total.detach()
    return total, logs
