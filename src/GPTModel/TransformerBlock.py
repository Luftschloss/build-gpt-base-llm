from .MultiHeadAttention import MultiHeadAttention
from .FeedForwardGELU import FeedForwardGELU
from .LayerNorm import LayerNorm
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        attn_drop_rate = cfg.get("attn_drop_rate", cfg["drop_rate"])
        shortcut_drop_rate = cfg.get("shortcut_drop_rate", cfg["drop_rate"])

        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=attn_drop_rate,
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(shortcut_drop_rate)

    def forward(self, x):

        # GPT-2 使用 pre-LayerNorm：先归一化，再进入注意力模块。
        resid_conn = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.dropout(x)
        # 残差连接保留原始信息流，缓解深层网络中的梯度消失。
        x = x + resid_conn

        # 前馈网络同样采用 pre-LayerNorm + 残差连接。
        resid_conn = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + resid_conn
        return x
