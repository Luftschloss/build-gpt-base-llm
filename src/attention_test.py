"""
Attention 机制模块
大语言模型之前，有一个方向是针对翻译任务的循环神经网络(RNN)架构，由于翻译在不同语言之间存在差异（如语法结构、词序等）为解决逐词翻译的局限性，通常使用包含两个子模块的深度神经网络，即所谓的编码器（encoder）和解码器（decoder）。
编码器的任务是先读取并处理整个文本，每一步更新一个隐状态（一个新的嵌入向量，用于保存局部上下文信息），然后解码器生成翻译后的文本。
然而，编码器-解码器架构在处理长文本时会遇到困难，因为编码器需要将整个输入文本压缩成一个固定长度的向量，这可能导致信息丢失。为了解决这个问题，Attention 机制被引入到编码器-解码器架构中。

RNN 的局限性和 Attention 机制的解决方法
1、局限性：假设我们有一个长句子：“The cat, who was sitting on the windowsill, jumped down because it saw a bird flying outside the window.”
    假设任务是预测句子最后的内容，即要理解“it”指的是“the cat”而不是“the windowsill”或其他内容。对于 RNN 来说，这个任务是有难度的，原因如下：
    * 长距离依赖问题：在 RNN 中，每个新输入的词会被依次传递到下一个时间步。随着句子长度增加，模型的隐状态会不断被更新，但早期信息（如“the cat”）会在层层传播中逐渐消失。因此，模型可能无法在“it”出现时有效地记住“the cat”是“it”的指代对象。
    * 梯度消失问题：RNN 在反向传播中的梯度会随着时间步的增加逐渐减小，这种“梯度消失”使得模型很难在长句中保持信息的准确传播，从而难以捕捉到长距离的语义关联。

2、注意力机制的解决方法
    为了弥补 RNN 的这些不足，注意力机制被引入。它的关键思想是在处理每个词时，不仅依赖于最后的隐藏状态，而是允许模型直接关注序列中的所有词。这样，即使是较远的词也能在模型计算当前词的语义时直接参与。
    在上例中，注意力机制如何帮助模型理解“it”指代“the cat”呢？  
    * 注意力机制的工作原理：当模型处理“it”时，注意力机制会将“it”与整个句子中的其他词进行相似度计算，判断“it”应该关注哪些词。
      由于“the cat”与“it”在语义上更相关，注意力机制会为“the cat”分配较高的权重，而其他词（如“windowsill”或“down”）则获得较低的权重。
    * 信息的直接引用：通过注意力机制，模型可以跳过中间步骤，直接将“it”与“the cat”关联，而不需要依赖所有的中间隐藏状态。

3、示例中的注意力矩阵
    假设使用一个简单的注意力矩阵，模型在处理“it”时，给每个词的权重可能如下（至于如何计算这些权重值后文会详细介绍）：
    词	  The  cat	who	  was	sitting	...	it   saw	bird	flying	...	window
    权重  0.1  0.3	0.05  0.05  0.05	...	0.4  0.05	0.02	0.01	...	0.02
    在这个注意力矩阵中，可以看到“it”对“the cat”有较高的关注权重（0.3），而对其他词的关注权重较低。这种直接的关注能力让模型能够高效捕捉长距离依赖关系，理解“it”与“the cat”的语义关联。

2014年RNN开发了第一代注意力机制，2017年研究发现自然语言处理的深度神经网络并不需要RNN架构，后来出现了Transformer架构引入了自注意力机制（self-attention），由此开启了大语言模型（LLM）的新时代。


这里演示四种不同的注意力机制变体
1）Simplified self-attention：最基本的注意力机制，直接计算查询向量与输入向量之间的点积来得到注意力得分，然后进行归一化。
2）Self-attention：在Transformer中使用的注意力机制，输入向量会被线性变换成查询（Q）、键（K）和值（V）三个向量，然后通过计算查询与键的点积来得到注意力得分，最后将得分应用于值向量来得到输出。
3）Causal attention（因果自注意力）：在生成式模型中使用的注意力机制，确保每个位置只能关注之前的位置，防止模型在生成文本时看到未来的信息。
4）Multi-Head attention：将输入向量分成多个头（head），每个头独立计算注意力得分和上下文向量。

"""

from self_attention import (
    CausalAttention,
    MultiHeadAttentionWrapper,
    SelfAttention_v1,
    SelfAttention_v2,
)
import torch
import tiktoken
from pathlib import Path

def build_attention_inputs_from_sentence(
        sentence,
        output_dim=3,
        add_position_embeddings=True,
        seed=123):
    """
    将一句文本转换为可传入attention计算的inputs。
    流程: sentence -> token ids -> token embeddings -> + position embeddings。

    注意: 这里每次新建embedding层仅用于教学演示；真实模型中embedding层应作为模型参数被长期持有并参与训练。
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    print("gpt2 tokenizer vocab_size: ", tokenizer.n_vocab)
    token_ids = tokenizer.encode(sentence)
    print("Token IDs:\n", token_ids)
    token_ids_tensor = torch.tensor(token_ids)
    with torch.random.fork_rng(devices=[]): # 临时保存当前 PyTorch 随机数生成器的状态，确保在函数执行后恢复原状态，避免影响外部随机数生成器的状态。
        if seed is not None:
            torch.manual_seed(seed)

        token_embedding_layer = torch.nn.Embedding(tokenizer.n_vocab, output_dim)
        token_embeddings = token_embedding_layer(token_ids_tensor)

        if not add_position_embeddings:
            return token_embeddings

        context_length = len(token_ids)
        pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
        pos_embeddings = pos_embedding_layer(torch.arange(context_length))

    return token_embeddings + pos_embeddings


    """
    用一个简单的3维输入向量来演示注意力机制的计算过程。每行代表一个token的嵌入向量。
    个人思考：这里对于注意力的得分计算比较笼统，仅说明了将当前的输入Token向量与其他输入的Token向量进行点积计算的注意力得分。
    实际上，每个输入Token会先通过权重矩阵W分别计算出它的Q、K、V三个向量，这三个向量的定义如下：
    * Q (Query)向量:  查询向量, 代表了这个词在寻找相关信息时提出的问题.
    * K (Key)向量:    键向量, 代表了一个单词的特征，或者说是这个单词如何"展示"自己，以便其它单词可以与它进行匹配.
    * V (Value)向量:  值向量, 携带的是这个单词的具体信息, 也就是当一个单词被"注意到"时, 它提供给关注者的内容.

    想象我们在图书馆寻找一本书（Q向量），我们知道要找的主题（Q向量），于是查询目录（K向量），目录告诉我哪本书涉及这个主题，最终我找到这本书并阅读内容（V向量），获取了我需要的信息。
    每个Token具体生成K、Q、V向量的方式主要通过线性变换：（其中W_Q，W_K和W_V是Transformer训练出的权重（每一层不同））
    Q = W_Q * (E1 + Pos1)
    K = W_K * (E2 + Pos2)
    V = W_V * (E3 + Pos3)

    Softmax是一种常见的激活函数（Activation Function）是神经网络中的核心组件，它的作用类似于神经元的“开关”或“过滤器”，负责决定神经元是否被激活（即输出信号），以及激活的程度。
    在神经网络中，激活函数通常用于将输入信号转换为输出信号，从而实现非线性变换。 常见的激活函数包括：
    ·Sigmoid：将输入信号转换为0到1之间的概率值，常用于二分类问题。
    ·ReLU：将输入信号转换为0到正无穷之间的值，常用于多分类问题。
    ·Softmax：将输入信号转换为0到1之间的概率值，常用于多分类问题。

    """

# 1 简化的自注意力机制的简单示例
def SimpleSelfAttentionExample(inputs):
    token_len = inputs.shape[0]  # 输入token的数量
    # 每个token与每个输入token之间的中间注意力得分计算如下
    x_2 = inputs[1]  # 以第二个token "journey" 的嵌入向量作为查询向量 (q)
    attn_scores_2 = torch.empty(token_len)
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, x_2)  # 计算查询向量与每个输入token的点积, 得到注意力得分, 点积越高表示越相关
    print("Attention Scores for 'journey':\n", attn_scores_2)
    
    # 简单归一化（score / sum(scores)）：简单直观，只保持比例，要求所有分数必须为正数，否则可能会得到负权重或不稳定的结果
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("====Simple Normalized Attention====")
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    # softmax归一化（exp(score) / sum(exp(scores))）， softmax 无论输入是正数还是负数，输出永远都是正数，适合分类任务
    # 这样较大的分数会被进一步放大，较小的分数会被进一步压低，这种方法更擅长处理极端值, 并且在训练过程中提供了更有利的梯度特性
    def softmax_naive(x):
        exp_x = torch.exp(x)
        return torch.exp(x) / torch.exp(x).sum(dim=0)
    attn_weights_2_naive = softmax_naive(attn_scores_2)
    print("====Simple Softmax Normalized Attention====")
    print("Attention weights:", attn_weights_2_naive)
    print("Sum:", attn_weights_2_naive.sum())

    # 更建议使用pyTorch内置的softmax函数, 该函数在数值稳定性方面进行了优化，能够更好地处理极端值，避免梯度消失或爆炸的问题。
    attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
    print("====Softmax Normalized Attention====")
    print("Attention weights:", attn_weights_2)
    print("Sum:", attn_weights_2.sum())

    # 计算上下文向量（context vector）: 将每个输入token的嵌入向量乘以对应的注意力权重，然后将这些加权的嵌入向量求和，得到一个新的向量，这个向量就是上下文向量，它包含了与查询token相关的信息。
    context_vec_2 = torch.zeros(x_2.shape)
    for i,x_i in enumerate(inputs):
        context_vec_2 += attn_weights_2[i]*x_i
    print("Context Vector for 'journey':\n", context_vec_2)

    # 为所有输入token计算注意力权重
    attn_scores_tmp = torch.empty(token_len, token_len)
    for i, x_i in enumerate(inputs):
        for j, x_j in enumerate(inputs):
            attn_scores_tmp[i, j] = torch.dot(x_i, x_j)
    print("Attention Scores:\n", attn_scores_tmp)

    # 利用imputs @ inputs.T可以直接计算出完整的注意力得分矩阵，效率更高
    attn_scores = inputs @ inputs.T
    print("Attention Scores (matrix form):\n", attn_scores)

    # 在使用 PyTorch 时，像 torch.softmax 这样的函数中的 dim 参数指定了将在输入张量中的哪个维度上进行归一化计算。
    # 通过设置 dim=-1，我们指示 softmax 函数沿着 attn_scores 张量的最后一个维度进行归一化操作。
    # 如果 attn_scores 是一个二维张量（例如，形状为 [行数, 列数]），则 dim=-1 将沿列方向进行归一化，使得每一行的值（沿列方向求和）之和等于 1。
    attn_weights = torch.softmax(attn_scores, dim=-1)
    print("Attention Weights:\n", attn_weights)

    # row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
    # print("Row 2 sum:", row_2_sum)
    # print("All row sums:", attn_weights.sum(dim=-1))
    all_context_vecs = attn_weights @ inputs
    print("Context Vectors:\n", all_context_vecs)
    print("Previous 2nd context vector:", context_vec_2)

# 2 添加可训练权重的自注意力机制示例
def SelfAttentionWithTrainableWeightsExample(inputs):
    x_2 = inputs[1]         # 以第二个token "journey"
    d_in = inputs.shape[1]  # 输入token嵌入向量维度
    d_out = d_in            # 输出维度，Q/K/V向量维度，教学示例中取较小值便于观察

    # step1: 构建线性变换生成查询（Q）、键（K）和值（V）向量
    # 一般GPT类模型中，输入维度和输出通常是相同的，下面初始化三个权重矩阵W_query、W_key和W_value，这些矩阵是模型的可训练参数，在训练过程中会被优化以学习如何有效地计算查询、键和值向量。
    # requires_grad=True 表示这些权重矩阵会进入autograd计算图，可以通过loss.backward()计算梯度。
    # 如果只是想让打印结果更清晰，不要关闭requires_grad；可以在打印时使用tensor.detach()。
    # 注意：这里的权重矩阵W中“权重”是“权重参数”，指神经网络训练过程中被优化的数值参数，而不是注意力权重，注意力权重用于确定上下文向量对输入文本的不同部分的依赖程度，即神经网络对输入不同部分的关注程度。
    # 总之，权重参数是神经网络的基本学习系数，用于定义网络层之间的连接关系，而注意力权重则是根据上下文动态生成的特定值，用于衡量不同词语或位置在当前上下文中的重要性
    torch.manual_seed(123)
    W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
    W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

    query_2 = x_2 @ W_query
    key_2 = x_2 @ W_key
    value_2 = x_2 @ W_value
    print("Q vector for 'journey':", query_2, "; K vector for 'journey':", key_2, "; V vector for 'journey':", value_2)

    queries = inputs @ W_query
    keys = inputs @ W_key
    values = inputs @ W_value
    print("====Self-Attention With Trainable Weights====")
    print("Queries shape:", queries.shape)
    print("Keys shape:", keys.shape)
    print("Values shape:", values.shape)

    # step2: 计算注意力得分和权重
    keys_2 = keys[1]
    attn_score_22 = query_2.dot(keys_2)
    print("Attention score between 'journey' and itself:", attn_score_22)
    attn_scores_2 = query_2 @ keys.T # All attention scores for given query
    print("Attention scores for 'journey':", attn_scores_2)

    # 每个query与所有key做点积，得到完整的自注意力得分矩阵。
    attn_scores = queries @ keys.T
    print("Attention scores:\n", attn_scores)

    # step3: 将注意力得分转换为注意力权重
    d_k = keys.shape[-1]
    # 通过将注意力得分除以keys嵌入维度的平方根来进行缩放（注意，取平方根在数学上等同于指数为 0.5 的运算）
    attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
    print("Attention weights for 'journey':", attn_weights_2)

    # 缩放点积注意力：除以sqrt(d_k)可以避免点积值过大导致softmax梯度过小。
    attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
    print("Attention weights:\n", attn_weights)
    print("Attention weights row sums:\n", attn_weights.sum(dim=-1))

    # step4: 计算上下文向量
    context_vec_2 = attn_weights_2 @ values
    print("Context vector for 'journey':", context_vec_2)

    context_vecs = attn_weights @ values
    print("Context vectors:\n", context_vecs)

    # 这里的loss仅用于演示W_query/W_key/W_value是可训练参数；真实训练应使用语言模型预测损失，需要开启权重的requires_grad=True。
    # dummy_loss = context_vecs[1].sum()
    # dummy_loss.backward()
    # print("W_query has gradient:", W_query.grad is not None)
    # print("W_key has gradient:", W_key.grad is not None)
    # print("W_value has gradient:", W_value.grad is not None)

    torch.manual_seed(123)
    sa_v1 = SelfAttention_v1(d_in, d_out)
    print("SelfAttention_v1 output:\n", sa_v1(inputs))

# 3 因果自注意力机制示例
def CausalAttentionExample(inputs):
    print("====Causal Attention 1====")
    torch.manual_seed(789)
    d_in = inputs.shape[1]
    d_out = d_in   
    sa_v2 = SelfAttention_v2(d_in, d_out)
    print("SelfAttention_v2 output:\n", sa_v2(inputs))
    queries = sa_v2.W_query(inputs)
    keys = sa_v2.W_key(inputs)
    attn_scores = queries @ keys.T
    print("Attention scores:\n", attn_scores)
    attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=1)
    print("Causal attention weights:\n", attn_weights)
    context_length = inputs.shape[0]
    # 使用 PyTorch 的 tril 函数来生成一个掩码矩阵，使对角线以上的值为零：
    mask_simple = torch.tril(torch.ones(context_length, context_length))
    print("Causal mask:\n", mask_simple)
    masked_simple = attn_weights*mask_simple
    print("Masked simple attention weights:\n", masked_simple)
    row_sums = masked_simple.sum(dim=1, keepdim=True)   # 计算每一行的和，保持维度不变，便于后续归一化
    masked_simple_norm = masked_simple / row_sums
    print("Masked simple attention weights (normalized):\n", masked_simple_norm)

    print("\n====Causal Attention 2====")
    # 通过创建一个对角线以上全为 1 的掩码，然后将这些 1 替换为负无穷大（-inf）值，从而实现这种更高效的掩码技巧：
    mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
    masked_attn_scores = attn_scores.masked_fill(mask.bool(), -torch.inf)
    print("Masked attention scores:\n", masked_attn_scores)
    attn_weights = torch.softmax(masked_attn_scores / keys.shape[-1]**0.5, dim=1)
    print("Causal attention weights:\n", attn_weights)

    print("\n====Causal Attention 3====")
    # 生成一个三维张量，包含 2 个输入文本，每个文本包含 6 个 token，每个 token 表示为一个 3 维嵌入向量
    batch = torch.stack((inputs, inputs), dim=0)
    print("Batch shape:", batch.shape)
    torch.manual_seed(123)
    ca = CausalAttention(d_in, d_out, context_length, 0.0)
    context_vecs = ca(batch)
    print("Causal attention output:", context_vecs)
    


# 4 多头注意力机制示例
def MultiHeadAttentionExample(inputs):
    batch = torch.stack((inputs, inputs), dim=0)
    context_length = inputs.shape[0]
    d_in = inputs.shape[1]
    d_out = d_in

    print("====Multi-Head Attention Wrapper====")
    torch.manual_seed(123)
    multihead_attn = MultiHeadAttentionWrapper(
        d_in=d_in,
        d_out=d_out,
        context_length=context_length,
        dropout=0.0,
        num_heads=batch.shape[0]
    )
    multihead_context_vecs = multihead_attn(batch)
    print("Multi-head attention output:\n", multihead_context_vecs)
    print("multihead_context_vecs.shape:", multihead_context_vecs.shape)


def main():
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your     (x^1)
    [0.55, 0.87, 0.66],     # journey  (x^2)
    [0.57, 0.85, 0.64],     # starts   (x^3)
    [0.22, 0.58, 0.33],     # with     (x^4)
    [0.77, 0.25, 0.10],     # one      (x^5)
    [0.05, 0.80, 0.55]])    # step     (x^6)

    # inputs = build_attention_inputs_from_sentence("Your journey starts with one step")  
    # print("Attention Inputs:\n", inputs)
    # SimpleSelfAttentionExample(inputs)
    # SelfAttentionWithTrainableWeightsExample(inputs)
    # CausalAttentionExample(inputs)
    MultiHeadAttentionExample(inputs)


if __name__ == "__main__":
    main()
