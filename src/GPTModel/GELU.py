import torch
import math

class GELU(torch.nn.Module):
    """Gaussian Error Linear Unit，GPT 中常用的平滑激活函数。"""
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        # 近似 GELU 公式：比 ReLU 更平滑，负半轴也保留少量梯度。
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
