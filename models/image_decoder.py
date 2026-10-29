"""Image decoder for pico-WAM: latent -> picture of the world.

The decoder turns any latent z (encoded from a real observation, or imagined
by the latent dynamics) back into an RGB image. Decoding imagined latents is
how we literally *see* what the WAM thinks will happen next.

    z (d_model) -> 128 x 4 x 4 -> [Upsample x2 -> Conv3x3] x log2(image_size / 4) -> 3 x H x W

Two small choices that matter a lot for a tiny decoder:
    * Upsample + Conv instead of ConvTranspose (no checkerboard artifacts).
    * Predict the difference from the dataset's mean image, with no sigmoid.
      Most pixels are static background, so the decoder starts out already
      drawing the background and spends its capacity on what moves. A sigmoid
      output saturates and can stall learning; we only clamp to [0, 1] for display.
"""

import math

import torch
import torch.nn as nn


class ImageDecoder(nn.Module):
    def __init__(self, d_model: int = 128, image_size: int = 64, base_channels: int = 128):
        super().__init__()

        num_up = int(round(math.log2(image_size / 4)))
        if 4 * 2 ** num_up != image_size:
            raise ValueError(f"image_size must be 4 * 2^k, got {image_size}")

        self.base_channels = base_channels
        self.fc = nn.Linear(d_model, base_channels * 4 * 4)

        layers = []
        ch = base_channels
        for i in range(num_up):
            out_ch = max(ch // 2, 32)
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(),
            ]
            ch = out_ch
        layers.append(nn.Conv2d(ch, 3, kernel_size=3, padding=1))
        self.deconv = nn.Sequential(*layers)

        # set from the dataset before training (PicoWAMPolicy.set_image_mean)
        self.register_buffer("mean_image", torch.full((3, image_size, image_size), 0.5))

    def forward(self, z):
        """
        z: (..., d_model)
        returns: (..., 3, H, W), roughly in [0, 1] (clamp for display)
        """
        lead = z.shape[:-1]
        x = self.fc(z.reshape(-1, z.shape[-1]))
        x = torch.relu(x).view(-1, self.base_channels, 4, 4)
        x = self.mean_image + self.deconv(x)
        return x.view(*lead, *x.shape[1:])
