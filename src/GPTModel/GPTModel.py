import torch
import torch.nn as nn

from .LayerNorm import LayerNorm
from .TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    """第 4 章完整 GPT 架构：Embedding -> N 个 TransformerBlock -> LayerNorm -> 输出层。"""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # token embedding 负责把 token ID 映射成连续向量。
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # position embedding 为同一个 token 在不同位置提供位置信息。
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg.get("emb_drop_rate", cfg["drop_rate"]))

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出层将每个位置的隐藏向量映射回词表大小，得到下一个 token 的 logits。
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        _, seq_len = in_idx.shape
        if seq_len > self.cfg["context_length"]:
            raise ValueError(
                f"输入序列长度 {seq_len} 超过模型上下文窗口 "
                f"{self.cfg['context_length']}"
            )

        tok_embeds = self.tok_emb(in_idx)
        # arange 放在输入同一设备上，保证模型迁移到 GPU 时位置索引不会留在 CPU。
        pos_ids = torch.arange(seq_len, device=in_idx.device)
        pos_embeds = self.pos_emb(pos_ids)

        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
