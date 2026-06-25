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


# SelfAttention_v1 是一个从 nn.Module 派生的类，SelfAttentionWithTrainableWeights的实现
class SelfAttention_v1(nn.Module):
    # 初始化了用于计算查询（query）、键（key）和值（value）的可训练权重矩阵（W_query、W_key 和 W_value），每个矩阵都将输入维度 d_in 转换为输出维度 d_out
    def __init__(self, d_in, d_out):
        super().__init__()
        self.d_out = d_out
        # v1 直接把 Q/K/V 投影矩阵注册为 nn.Parameter。
        # 如果输入 x 的形状是 [num_tokens, d_in]，则 x @ W_* 的结果形状是 [num_tokens, d_out]。
        # 这些矩阵是模型参数，真实训练时会通过 loss.backward() 得到梯度，再由 optimizer.step() 更新。
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    # 前向传播过程在 forward 方法中实现，我们通过将查询（query）和键（key）相乘来计算注意力得分（attn_scores）
    # 并使用 softmax 对这些得分进行归一化。最后，我们使用这些归一化的注意力得分对值（value）加权，生成上下文向量
    def forward(self, x):
        # x: [num_tokens, d_in]
        # keys/queries/values: [num_tokens, d_out]
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # attn_scores: [num_tokens, num_tokens]
        # 第 i 行表示第 i 个 token 的 query 与所有 token 的 key 的匹配分数。
        attn_scores = queries @ keys.T # omega

        # 除以 sqrt(d_k) 是 scaled dot-product attention 的缩放项。
        # dim=-1 表示对每一行做 softmax，因此每一行注意力权重之和为 1。
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1)

        # context_vec: [num_tokens, d_out]
        # 第 i 行是第 i 个 token 按自己的注意力权重对所有 value 加权求和后的上下文向量。
        context_vec = attn_weights @ values
        return context_vec


# 通过使用 PyTorch 的 nn.Linear 层来进一步改进 SelfAttention_v1 的实现。
# 当禁用偏置单元时，nn.Linear 层可以有效地执行矩阵乘法。
# 此外，使用 nn.Linear 替代手动实现的 nn.Parameter(torch.rand(...)) 的一个显著优势在于，nn.Linear 具有优化的权重初始化方案，从而有助于实现更稳定和更高效的模型训练
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        # v2 使用 nn.Linear 封装线性投影。bias=False 时，数学上等价于一次矩阵乘法。
        # 注意：nn.Linear 的 weight 形状是 [d_out, d_in]，其内部计算等价于 x @ weight.T。
        # 因此它和 v1 的 x @ W_* 方向不同，但表达的是同一种从 d_in 到 d_out 的线性变换。
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        # keys/queries/values: [num_tokens, d_out]
        # 这里调用 nn.Linear，比手写 nn.Parameter 更接近真实 Transformer 代码。
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 后续注意力计算与 v1 完全一致：
        # scores = Q @ K.T -> softmax(scores / sqrt(d_k)) -> weights @ V
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values
        return context_vec

# Dropout 是一种正则化技术，用于防止神经网络在训练过程中过拟合。它的核心思想是在训练过程中随机“丢弃”一部分神经元，使得模型在每次迭代中都只能依赖于部分特征，从而增强模型的泛化能力。
# 注意力机制存在冗余性：在 Transformer 的注意力机制中，模型通常会对多个 token 进行注意力计算，实际上会有一些冗余信息。也就是说，不同 token 之间的信息通常会有部分重叠，并且模型能够从多个来源获取类似的信息。在这种情况下，dropout 随机丢弃一部分注意力权重并不会完全破坏模型的性能，因为模型可以依赖于其他未被丢弃的注意力路径来获取所需信息
# 缩放操作的作用：在应用 dropout 后，为了保持训练和推理阶段的输出期望值一致，通常会对剩余的神经元进行缩放。具体来说，如果 dropout 的概率为 p，那么在训练阶段，保留下来的神经元的输出会除以 (1-p) 进行缩放。这确保了在训练和推理阶段，神经网络的输出期望值保持一致，从而避免了模型在推理阶段出现性能下降的问题。
# 训练过程中多次迭代弥补信息丢失：在训练过程中，每个 batch 中的 dropout 掩码都是随机生成的。也就是说，在每次训练时被丢弃的注意力权重是随机的，并不会始终忽略相同的 token。这种随机性确保了在训练过程中，模型会在多个迭代中多次关注到每个 token。因此，即便某个 token 在当前的训练步中被忽略，在未来的训练步骤中它仍然会被关注到，从而在整体上避免了信息丢失的问题。
    
# 因果自注意力机制：用于GPT这类自回归模型，当前位置只能关注自己和之前的token，不能看到后续token。
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout=0.0, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 与之前的 SelfAttention_v1 类相比，我们添加了一个 dropout 层
        self.dropout = nn.Dropout(dropout)

        # 上三角矩阵表示“未来token”的位置。register_buffer会随模型保存/加载，但不会作为可训练参数更新。
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        
        # 这里因为Keys[batch, num_tokens, d_out]，所以需要对Keys进行转置，才能与Queries相乘
        # 交换第 1 和第 2 个维度，同时保持批次(batch)维度在第1个位置（索引0），得到的attn_scores形状为[batch, num_tokens, num_tokens]，表示每个token对所有token的注意力得分。
        attn_scores = queries @ keys.transpose(1, 2)   
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # 和masked_fill不同，在 PyTorch 中，带有下划线后缀的操作会在原有内存空间执行，直接修改变量本身，从而避免不必要的内存拷贝
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        # 对注意力权重矩阵应用dropout，由此生成的注意力权重矩阵中，部分元素被置零，剩余的元素重新进行了缩放
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
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
