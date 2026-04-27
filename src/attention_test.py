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



"""

import torch
import tiktoken
from pathlib import Path


def SimpleAttentionExaple():
    """
    用一个简单的3维输入向量来演示注意力机制的计算过程。每行代表一个token的嵌入向量。
    个人思考：这里对于注意力的得分计算比较笼统，仅说明了将当前的输入Token向量与其他输入的Token向量进行点积计算的注意力得分。
    实际上，每个输入Token会先通过权重矩阵W分别计算出它的Q、K、V三个向量，这三个向量的定义如下：
    * Q (Query)向量:  查询向量, 代表了这个词在寻找相关信息时提出的问题.
    * K (Key)向量:    键向量, 代表了一个单词的特征，或者说是这个单词如何"展示"自己，以便其它单词可以与它进行匹配.
    * V (Value)向量:  值向量, 携带的是这个单词的具体信息, 也就是当一个单词被"注意到"时, 它提供给关注者的内容.
    想象我们在图书馆寻找一本书（Q向量），我们知道要找的主题（Q向量），于是查询目录（K向量），目录告诉我哪本书涉及这个主题，最终我找到这本书并阅读内容（V向量），获取了我需要的信息。
    具体生成K、Q、V向量的方式主要通过线性变换：（其中W_Q，W_K和W_V是Transformer训练出的权重（每一层不同））
    Q = W_Q * (E1 + Pos1)
    K = W_K * (E2 + Pos2)
    V = W_V * (E3 + Pos3)

    """
    inputs = torch.tensor(
    [[0.43, 0.15, 0.89],    # Your     (x^1)
    [0.55, 0.87, 0.66],     # journey  (x^2)
    [0.57, 0.85, 0.64],     # starts   (x^3)
    [0.22, 0.58, 0.33],     # with     (x^4)
    [0.77, 0.25, 0.10],     # one      (x^5)
    [0.05, 0.80, 0.55]])    # step     (x^6)

    # 每个token与每个输入token之间的中间注意力得分计算如下
    query = inputs[1]  # 以第二个token "journey" 的嵌入向量作为查询向量 (q)
    attn_scores_2 = torch.empty(inputs.shape[0])
    for i, x_i in enumerate(inputs):
        attn_scores_2[i] = torch.dot(x_i, query)  # 计算查询向量与每个输入token的点积, 得到注意力得分, 点积越高表示越相关
    print("Attention Scores for 'journey':\n", attn_scores_2)
    
    # 归一化
    attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
    print("====Simple Normalized Attention====")
    print("Attention weights:", attn_weights_2_tmp)
    print("Sum:", attn_weights_2_tmp.sum())

    # softmax归一化更为常见, 这种方法更擅长处理极端值, 并且在训练过程中提供了更有利的梯度特性
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

    

def main():
    SimpleAttentionExaple();


if __name__ == "__main__":
    main()
