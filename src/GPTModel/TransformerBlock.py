from .MultiHeadAttention import MultiHeadAttention
from .FeedForwardGELU import FeedForwardGELU
from .LayerNorm import LayerNorm
from torch import nn

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForwardGELU(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.dropout = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        resid_conn = x
        x = self.norm1(x)  # pre-LayerNorm
        x = self.attn(x)
        x = self.dropout(x)
        x = x + resid_conn

        resid_conn = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + resid_conn
        return x
