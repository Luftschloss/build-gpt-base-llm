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

        self.w_queries = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_keys = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.w_values = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            'mask',
            torch.tril(torch.ones(context_length, context_length)).unsqueeze(0).unsqueeze(0)
        )

    def forward(self, x):
        batches, num_tokens, dim_in = x.shape

        # Linear projections
        queries = self.w_queries(x)
        keys = self.w_keys(x)
        values = self.w_values(x)

        # Reshape and transpose for multi-head attention
        queries = queries.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batches, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention score calculation
        attn_scores = (queries @ keys.transpose(2, 3)) / (self.head_dim ** 0.5)

        # Apply mask: Broadcasting across batches and heads
        attn_scores = attn_scores.masked_fill(self.mask[:, :, :num_tokens, :num_tokens] == 0, float('-inf'))

        # Softmax to get attention weights
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Context vector computation
        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(batches, num_tokens, self.d_out)

        # Final linear projection
        context_vec = self.out_proj(context_vec)

        return context_vec