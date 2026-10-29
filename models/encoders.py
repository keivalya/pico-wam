"""Encoders for images, text, and states."""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ImageEncoderTinyCNN(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.proj = nn.Linear(128, d_model)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.mean(dim=[2, 3])  # GAP
        x = self.proj(x)
        x = self.ln(x)
        return x  # (B, d_model)


class TextEncoderTinyGRU(nn.Module):
    def __init__(self, vocab_size, d_word=64, d_model=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_word)
        self.gru = nn.GRU(d_word, d_model, batch_first=True)
        self.ln = nn.LayerNorm(d_model)

    def forward(self, token_ids):
        x = self.embed(token_ids)  # (B, T, d_word)
        _, h_last = self.gru(x)
        x = h_last[0]  # (B, d_model)
        x = self.ln(x)
        return x


class StateEncoderMLP(nn.Module):
    def __init__(self, state_dim, d_model=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(self, s):
        x = self.net(s)
        x = self.ln(x)
        return x


class ImageEncoderKeypoints(nn.Module):
    """
    CNN + spatial softmax ("where is stuff?").

    Each of the K keypoint channels produces a heatmap over the image; the
    spatial softmax turns it into one (x, y) coordinate = the expected
    location under that heatmap. Global average pooling (TinyCNN above)
    throws away *where* things are, but a world model needs exactly that.

    Bonus for interpretability: the K keypoints can be drawn on the image to
    see what the encoder is tracking (see scripts/visualize_rollout.py).
    """

    def __init__(self, d_model=128, num_keypoints=16):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.to_heatmaps = nn.Conv2d(64, num_keypoints, kernel_size=1)
        self.proj = nn.Linear(2 * num_keypoints + 64, d_model)
        self.ln = nn.LayerNorm(d_model)

    def heatmaps(self, x):
        """
        x: (B, 3, H, W)
        returns: features (B, 64, h, w), attention (B, K, h, w) summing to 1 over (h, w)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        feat = F.relu(self.conv3(x))
        logits = self.to_heatmaps(feat)
        B, K, h, w = logits.shape
        attn = F.softmax(logits.view(B, K, h * w), dim=-1).view(B, K, h, w)
        return feat, attn

    def keypoints(self, x):
        """
        returns: (B, K, 2) expected (x, y) in [-1, 1] image coordinates
                 (x to the right, y downward, like image rows)
        """
        _, attn = self.heatmaps(x)
        return self._expected_xy(attn)

    def _expected_xy(self, attn):
        B, K, h, w = attn.shape
        ys = torch.linspace(-1, 1, h, device=attn.device)
        xs = torch.linspace(-1, 1, w, device=attn.device)
        kp_x = (attn.sum(dim=2) * xs).sum(dim=-1)
        kp_y = (attn.sum(dim=3) * ys).sum(dim=-1)
        return torch.stack([kp_x, kp_y], dim=-1)

    def forward(self, x):
        feat, attn = self.heatmaps(x)
        kp = self._expected_xy(attn)                 # (B, K, 2)  where
        gap = feat.mean(dim=[2, 3])                  # (B, 64)    what
        x = self.proj(torch.cat([kp.flatten(1), gap], dim=-1))
        x = self.ln(x)
        return x  # (B, d_model)


def build_image_encoder(name, d_model=128):
    if name == "cnn":
        return ImageEncoderTinyCNN(d_model=d_model)
    if name == "keypoints":
        return ImageEncoderKeypoints(d_model=d_model)
    raise ValueError(f"Unknown image_encoder: {name}")
