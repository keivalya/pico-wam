"""World Action Model Policy.

VLA:
    image_t + text + state_t -> action_t

pico-WAM:
    image_t + text + state_t -> action_t + state_t+1 + image_t+1

How the pieces fit together:

    image_t ──► ImageEncoder ──┐
    text    ──► TextEncoder  ──┼─► FusionMLP ──► z_t   (latent world state)
    state_t ──► StateEncoder ──┘                 │
                                                 ├─► ActionHead      ──► a_t
                                                 ├─► StateReadout    ──► s_t        (recon)
                                                 ├─► ImageDecoder    ──► image_t    (recon)
                                                 │
                         a_t ──► LatentDynamics(z_t, a_t) ──► z^_t+1   (imagined)
                                                 ├─► StateReadout    ──► s^_t+1
                                                 ├─► ImageDecoder    ──► image^_t+1
                                                 └─► ActionHead      ──► a^_t+1 ...

The policy, the readouts and the dynamics all live on the same latent z,
so the WAM can "dream": roll the policy forward entirely in latent space
and decode what it expects to see (see `imagine`).

model_type="vla" switches the world model off, leaving mini-VLA
(https://github.com/keivalya/mini-vla) with the same encoders and action head:
the baseline every WAM experiment should be compared against.

Supported action heads:
    mlp
    diffusion
    flow
"""

import torch
import torch.nn as nn

from .encoders import build_image_encoder, TextEncoderTinyGRU, StateEncoderMLP
from .fusion import FusionMLP
from .action_heads import build_action_head
from .dynamics_head import LatentDynamicsMLP, StateReadout
from .image_decoder import ImageDecoder


class PicoWAMPolicy(nn.Module):
    def __init__(
        self,
        vocab_size,
        state_dim,
        action_dim,
        d_model=128,
        action_head="mlp",
        image_encoder="keypoints",
        image_size=64,
        model_type="wam",
        predict_state=True,
        predict_image=True,
        linear_readout=True,
        use_state_input=True,
        diffusion_T=16,
        flow_steps=8,
        action_chunk=1,
        policy_hidden=None,
        policy_layers=None,
    ):
        super().__init__()

        if model_type not in ["vla", "wam"]:
            raise ValueError(f"Unknown model_type: {model_type}")

        # everything needed to rebuild this model from a checkpoint
        self.config = dict(
            vocab_size=vocab_size,
            state_dim=state_dim,
            action_dim=action_dim,
            d_model=d_model,
            action_head=action_head,
            image_encoder=image_encoder,
            image_size=image_size,
            model_type=model_type,
            predict_state=predict_state,
            predict_image=predict_image,
            linear_readout=linear_readout,
            use_state_input=use_state_input,
            diffusion_T=diffusion_T,
            flow_steps=flow_steps,
            action_chunk=action_chunk,
            policy_hidden=policy_hidden,
            policy_layers=policy_layers,
        )

        self.is_wam = model_type == "wam"
        self.predict_state = self.is_wam and predict_state
        self.predict_image = self.is_wam and predict_image
        self.use_state_input = use_state_input

        # observation encoder: (image, text, state) -> z
        self.img_encoder = build_image_encoder(image_encoder, d_model=d_model)
        self.txt_encoder = TextEncoderTinyGRU(vocab_size=vocab_size, d_word=64, d_model=d_model)
        self.state_encoder = StateEncoderMLP(state_dim=state_dim, d_model=d_model)
        self.fusion = FusionMLP(d_model=d_model)

        # policy: z -> a, or with action_chunk = K > 1 the next K actions at once
        # (as in Diffusion Policy); act() returns the first, act_chunk() all K.
        # policy_hidden / policy_layers size the head (default: d_model wide,
        # 1 hidden layer for mlp, 2 for flow)
        self.action_dim, self.action_chunk = action_dim, action_chunk
        self.policy_head = build_action_head(
            action_head,
            action_dim=action_dim * action_chunk,
            cond_dim=d_model,
            diffusion_T=diffusion_T,
            flow_steps=flow_steps,
            hidden_dim=policy_hidden,
            num_layers=policy_layers,
        )

        # world model: (z, a) -> z', and z -> (state, image)
        if self.is_wam:
            self.dynamics = LatentDynamicsMLP(d_model=d_model, action_dim=action_dim)
        if self.predict_state:
            self.state_readout = StateReadout(d_model=d_model, state_dim=state_dim, linear=linear_readout)
        if self.predict_image:
            self.image_decoder = ImageDecoder(d_model=d_model, image_size=image_size)

        # state normalization statistics (set from the dataset before training)
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))

    # ------------------------------------------------------------------ utils

    def set_state_stats(self, mean, std):
        self.state_mean.copy_(torch.as_tensor(mean, dtype=torch.float32))
        self.state_std.copy_(torch.as_tensor(std, dtype=torch.float32).clamp_min(1e-3))

    def set_image_mean(self, mean_image):
        """mean_image: (3, H, W) in [0, 1]; the decoder predicts the difference from it."""
        if self.predict_image:
            self.image_decoder.mean_image.copy_(torch.as_tensor(mean_image, dtype=torch.float32))

    def normalize_state(self, s):
        return (s - self.state_mean) / self.state_std

    def denormalize_state(self, s):
        return s * self.state_std + self.state_mean

    @classmethod
    def from_checkpoint(cls, path, device="cpu"):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = cls(**ckpt["model_config"]).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        return model, ckpt

    # ----------------------------------------------------------------- encode

    def encode_obs(self, image, text_tokens, state):
        """(B, 3, H, W), (B, T), (B, state_dim) -> z: (B, d_model)"""
        img_token = self.img_encoder(image)
        txt_token = self.txt_encoder(text_tokens)
        state_token = self.state_encoder(self.normalize_state(state))
        if not self.use_state_input:
            # image + language only: the model must infer the state from pixels
            state_token = torch.zeros_like(state_token)
        return self.fusion(img_token, txt_token, state_token)

    def encode_sequence(self, images, text_tokens, states):
        """(B, H, 3, H, W), (B, T), (B, H, state_dim) -> (B, H, d_model)"""
        B, H = images.shape[:2]
        text_rep = text_tokens.unsqueeze(1).expand(B, H, -1).reshape(B * H, -1)
        z = self.encode_obs(images.flatten(0, 1), text_rep, states.flatten(0, 1))
        return z.view(B, H, -1)

    # --------------------------------------------------------------- training

    def forward(self, batch):
        """
        batch (from scripts/train.py):
            image         : (B, 3, H, W)
            text_ids      : (B, T)
            state         : (B, state_dim)
            action        : (B, action_dim)             a_t
            is_expert     : (B,)  optional              imitate a_t only where 1
            action_chunk  : (B, K, action_dim)          a_t ... a_t+K-1, if action_chunk = K > 1
            chunk_expert  : (B,)                        1 where all K actions are expert actions
          WAM only:
            actions       : (B, Hz, action_dim)         a_t ... a_t+Hz-1
            future_images : (B, Hz, 3, H, W)            image_t+1 ... image_t+Hz
            future_states : (B, Hz, state_dim)          state_t+1 ... state_t+Hz

        returns a dict of predictions; models/losses.py turns it into a loss.
        """
        z = self.encode_obs(batch["image"], batch["text_ids"], batch["state"])

        # the policy imitates expert actions only; exploration actions are
        # there to teach the world model what actions do
        target, mask = batch["action"], batch.get("is_expert")
        if self.action_chunk > 1:
            target, mask = batch["action_chunk"].flatten(1), batch["chunk_expert"]
        expert = mask > 0.5 if mask is not None else slice(None)
        if isinstance(expert, torch.Tensor) and not expert.any():
            action_loss = z.sum() * 0.0
        else:
            action_loss = self.policy_head.loss(target[expert], z[expert])

        outputs = {"z": z, "action_loss": action_loss}

        if not self.is_wam:
            return outputs

        # imagine the future using the actions that were actually taken
        z_pred = self.dynamics.rollout(z, batch["actions"])           # (B, Hz, D)
        outputs["z_pred"] = z_pred
        outputs["z_target"] = self.encode_sequence(
            batch["future_images"], batch["text_ids"], batch["future_states"]
        )

        if self.predict_state:
            outputs["state_recon"] = self.state_readout(z)
            outputs["state_pred"] = self.state_readout(z_pred)

        if self.predict_image:
            outputs["image_recon"] = self.image_decoder(z)
            outputs["image_pred"] = self.image_decoder(z_pred)

        return outputs

    # -------------------------------------------------------------- inference

    @torch.no_grad()
    def act(self, image, text_tokens, state):
        return self.act_chunk(image, text_tokens, state)[:, 0]

    @torch.no_grad()
    def act_chunk(self, image, text_tokens, state):
        """(B, K, action_dim): the next K actions (K = 1 without action chunking)."""
        z = self.encode_obs(image, text_tokens, state)
        return self.policy_head.sample(z).view(z.shape[0], self.action_chunk, self.action_dim)

    @torch.no_grad()
    def decode(self, z):
        """Read a latent (real or imagined) back out as state and/or image."""
        out = {}
        if self.predict_state:
            out["state"] = self.denormalize_state(self.state_readout(z))
        if self.predict_image:
            out["image"] = self.image_decoder(z).clamp(0.0, 1.0)
        return out

    @torch.no_grad()
    def imagine(self, image, text_tokens, state, horizon=10, actions=None):
        """
        Dream `horizon` steps ahead from one real observation.

        actions=None : the policy acts inside its own imagination
                       (z -> a = policy(z) -> z' = dynamics(z, a) -> ...)
        actions given: (B, horizon, action_dim) "what if I did this?"

        returns:
            z       : (B, horizon + 1, d_model)  latents, index 0 is the real z_t
            actions : (B, horizon, action_dim)
            state   : (B, horizon + 1, state_dim)  if predict_state
            image   : (B, horizon + 1, 3, H, W)    if predict_image
        """
        if not self.is_wam:
            raise RuntimeError("model_type='vla' has no world model to imagine with.")

        z = self.encode_obs(image, text_tokens, state)
        zs, acts = [z], []

        for k in range(horizon):
            a = self.policy_head.sample(z)[:, : self.action_dim] if actions is None else actions[:, k]
            z = self.dynamics(z, a)
            zs.append(z)
            acts.append(a)

        out = {"z": torch.stack(zs, dim=1), "actions": torch.stack(acts, dim=1)}
        out.update(self.decode(out["z"]))
        return out
