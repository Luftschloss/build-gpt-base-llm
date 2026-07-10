from torch import nn
import torch

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module.

    Args:
        d_in (int): Input dimension.
        d_out (int): Output dimension.
        context_length (int): The length of the input sequence.
        dropout (float): Dropout probability.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): Whether to include bias in query, key, and value projections. Default is False.

    Attributes:
        d_out (int): Output dimension.
        num_heads (int): Number of attention heads.
        head_dim (int): Dimension of each attention head.
        w_queries (nn.Linear): Linear projection for queries.
        w_keys (nn.Linear): Linear projection for keys.
        w_values (nn.Linear): Linear projection for values.
        out_proj (nn.Linear): Linear projection for output.
        dropout (nn.Dropout): Dropout layer.
        mask (torch.Tensor): Lower triangular mask to ensure causality.
    """

    def __init__(self, d_in: int, d_out: int, context_length: int,
                 dropout: float, num_heads: int, qkv_bias: bool = False):
        super(MultiHeadAttention, self).__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        # 三个线性层分别把输入投影为 Query / Key / Value。
        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        # 下三角因果 mask：当前位置只能关注自己和之前的 token，不能看到未来 token。
        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        batches, num_tokens, _ = x.shape

        # 线性投影后形状为 (batch, num_tokens, d_out)。
        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)

        # 拆成多个 head，并把 head 维度提前，便于并行计算注意力。
        queries = queries.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # 缩放点积注意力：除以 sqrt(head_dim) 可以稳定 softmax 的数值范围。
        attn_scores = (queries @ keys.transpose(2, 3)) / (self.head_dim ** 0.5)

        # mask 会按 batch 和 head 广播，屏蔽所有未来位置。
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :num_tokens, :num_tokens] == 0, float('-inf'))

        # 注意力权重表示每个 token 应该从历史 token 中读取多少信息。
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 合并所有 head 的上下文向量，恢复为 (batch, num_tokens, d_out)。
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out)

        # 输出投影让不同 head 的信息进一步混合。
        context_vec = self.out_proj(context_vec)

        return context_vec
