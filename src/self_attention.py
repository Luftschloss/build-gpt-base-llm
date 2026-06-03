import torch
import torch.nn as nn

"""
带可训练权重的自注意力机制框架

输入:
    X: [num_tokens, d_in]
       每一行是一个token的输入向量，例如token embedding + position embedding。

可训练权重:
    W_query: [d_in, d_out]
    W_key:   [d_in, d_out]
    W_value: [d_in, d_out]

框架图:

                +------------------+
                | X [tokens,d_in]  |
                +------------------+
                   |       |       |
                   |       |       |
                   v       v       v
              X @ W_q  X @ W_k  X @ W_v
                   |       |       |
                   v       v       v
                  Q       K       V
                  |       |       |
                  |       v       |
                  +--> Q @ K.T    |
                         |        |
                         v        |
                 / sqrt(d_out)    |
                         |        |
                         v        |
                      softmax     |
                         |        |
                         v        |
                attention weights |
                         |        |
                         +------> @ V
                                  |
                                  v
                         context vectors

核心公式:
    Q = X @ W_query
    K = X @ W_key
    V = X @ W_value

    attention_scores = Q @ K.T
    attention_weights = softmax(attention_scores / sqrt(d_out))
    context_vectors = attention_weights @ V

原理:
    1. Q表示当前token主动提出的查询: 我应该关注哪些token。
    2. K表示每个token可被匹配的特征: 我具有什么可被关注的信息。
    3. Q @ K.T计算每个token对所有token的相关性得分。
    4. softmax将相关性得分转换为注意力权重，每一行权重和为1。
    5. attention_weights @ V按照注意力权重汇总信息，得到每个token的新上下文向量。

训练:
    W_query/W_key/W_value初始是随机权重。
    它们被注册为nn.Parameter或nn.Linear内部参数后，会在loss.backward()中获得梯度，
    再通过optimizer.step()被更新，最终学习到如何生成更有效的Q/K/V表示。

为什么使用Q、K和V向量？
    在注意力机制的上下文中，“键”（key）、“查询”（query）和“值”（value）这些术语来源于信息检索和数据库领域，在这些领域中也使用类似的概念来存储、搜索和检索信息
    查询（query）类似于数据库中的搜索查询。它代表模型当前关注或试图理解的项（如句子中的某个词或 token）。通过查询，模型可以探查输入序列中的其他部分，以确定对它们应关注的程度。
    键（key）类似于数据库中用于索引和查找的键。在注意力机制中，输入序列的每个元素（例如句子中的每个单词）都对应一个关联的‘键’。这些‘键’用于与‘查询’进行匹配。
    值（value）类似于数据库中的键值对中的“值”。它表示输入项的实际内容或表示。当模型确定哪些键（即输入中的哪些部分）与查询（当前的关注项）最相关时，就会检索出对应的值。
"""


# SelfAttention_v1 是一个从 nn.Module 派生的类
class SelfAttention_v1(nn.Module):
    # 初始化了用于计算查询（query）、键（key）和值（value）的可训练权重矩阵（W_query、W_key 和 W_value），每个矩阵都将输入维度 d_in 转换为输出维度 d_out
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # 前向传播过程在 forward 方法中实现，我们通过将查询（query）和键（key）相乘来计算注意力得分（attn_scores）
    # 并使用 softmax 对这些得分进行归一化。最后，我们使用这些归一化的注意力得分对值（value）加权，生成上下文向量
    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value
        attn_scores = queries @ keys.T # omega
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# 通过使用 PyTorch 的 nn.Linear 层来进一步改进 SelfAttention_v1 的实现。
# 当禁用偏置单元时，nn.Linear 层可以有效地执行矩阵乘法。
# 此外，使用 nn.Linear 替代手动实现的 nn.Parameter(torch.rand(...)) 的一个显著优势在于，nn.Linear 具有优化的权重初始化方案，从而有助于实现更稳定和更高效的模型训练
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec


# 因果自注意力机制：用于GPT这类自回归模型，当前位置只能关注自己和之前的token，不能看到后续token。
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)

        # 上三角矩阵表示“未来token”的位置。register_buffer会随模型保存/加载，但不会作为可训练参数更新。
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x, return_attn_weights=False):
        batch_size, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1, 2)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values

        if return_attn_weights:
            return context_vec, attn_weights
        return context_vec


# 多头注意力包装器：将多个单头因果注意力并行应用到同一输入，并把各头输出拼接。
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, num_heads=2, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
