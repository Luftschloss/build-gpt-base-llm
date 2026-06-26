import torch
import torch.nn as nn


# scale 和 shift 是两个可训练参数（与输入具有相同的维度）。
# 在训练中会自动调整这些参数，以改善模型在训练任务上的性能。这使得模型能够学习适合数据处理的最佳缩放和偏移方式。
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        # eps是一个小的常数，用于防止除以零的情况，通常设置为1e-5
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
