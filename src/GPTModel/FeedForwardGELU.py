import torch.nn as nn

from .GELU import GELU


class FeedForwardGELU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            # GPT 前馈层通常先扩展到 4 倍维度，提供更大的中间表示空间。
            nn.Linear(cfg['emb_dim'], 4*cfg['emb_dim']),
            GELU(),
            # 再投影回 emb_dim，保证残差连接前后的张量形状一致。
            nn.Linear(4*cfg['emb_dim'], cfg['emb_dim']),
        )

    def forward(self, x):
        return self.layers(x)
