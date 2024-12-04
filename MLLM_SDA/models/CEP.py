import re
import math
import torch
from torch import nn
from functools import partial
from timm.layers.norm_act import LayerNormAct2d


class FeatureIRLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
        nn.Linear(in_dim, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class TokenDownLayer(nn.Module):
    def __init__(self, shape) -> None:
        super().__init__()
        self.dwn = nn.Sequential(
            nn.AdaptiveAvgPool2d(shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        assert h * h == num_tokens
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwn(x)

        x = x.flatten(2).transpose(1, 2)

        return x


class PosInjectLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1, mode="sum", layer_scale_init_value=1e-6) -> None:
        super().__init__()
        self.mode = mode
        self.norm = nn.LayerNorm(in_dim)

        self.peg = nn.Conv2d(out_dim, out_dim, 3, stride, 1, bias=True, groups=out_dim)

        self.pointwise_conv = nn.Conv2d(out_dim, out_dim, kernel_size=1, stride=1, padding=0)

        self.f = nn.Linear(out_dim, 2 * out_dim)
        self.act = nn.GELU()
        self.g = nn.Linear(out_dim, out_dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_dim)), requires_grad=True)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))

        assert h * h == num_tokens

        x = self.norm(x)

        cnn_feat = x.transpose(1, 2).reshape(b, c, h, h)

        cnn_feat = self.peg(cnn_feat)
        cnn_feat = self.pointwise_conv(cnn_feat)

        b, c, h, w = cnn_feat.shape
        cnn_feat = cnn_feat.flatten(2).transpose(1, 2)

        x = self.f(cnn_feat)

        x1, x2 = x.reshape(b, h * w, 2, -1).unbind(2)
        print(x1.shape)
        print(x2.shape)
        x = self.act(x1) + x2 if self.mode == "sum" else self.act(x1) * x2
        print(x.shape)

        x = self.g(x)

        x = self.gamma * x
        print("The gamma shape is %d", self.gamma.shape)

        x = self.norm(x)
        x = x + residual

        x = x.reshape(b, num_tokens, -1)

        return x


class cep_projector(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        inc, ouc = 5632, 4096
        self.mlp = FeatureIRLayer(inc, ouc)
        self.dwn = TokenDownLayer((12, 12))
        self.peg = PosInjectLayer(ouc, ouc, stride=1)

    def forward(self, x):
        x = self.mlp(x)

        x = self.dwn(x)
        residual = x
        x = self.peg(x, residual)
        return x

