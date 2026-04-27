import torch
import tiktoken
from pathlib import Path
import re
from dataset_loader_test import create_dataloader_v1
from tokenizer_test import SimpleTokenizerV1, SimpleTokenizerV2

# 在处理文本数据时，连续向量表示（embedding）非常重要
# embedding 是将文本数据转换为连续向量的过程，这些向量可以捕捉文本的语义和上下文信息。通过使用 embedding，我们可以将文本数据表示为数值形式，使得机器学习模型能够更好地理解和处理文本。
# 整理一下Embedding部分的摘要：
# * LLM 需要将文本数据转换为数值向量，这称之为嵌入，因为它们无法处理原始文本。嵌入将离散数据（如单词或图像）转化为连续的向量空间，从而使其能够与神经网络操作兼容。
# * 作为第一步，原始文本被分解为token，这些token可以是单词或字符。然后，这些token被转换为整数表示，称为token ID。
# * 可以添加特殊token，如 <|unk|> 和 <|endoftext|>，以增强模型的理解能力，并处理各种上下文，例如未知单词或无关文本之间的边界分隔。
# * 用于像 GPT-2 和 GPT-3 这样的 LLM 的字节对编码（BPE）分词器，可以通过将未知单词分解为子词单元或单个字符，高效地处理这些单词。
# * 我们在分词后的文本数据上采用滑动窗口方法，以生成用于 LLM 训练的输入-目标对。
# * 在 PyTorch 中，嵌入层作为一种查找操作，用于检索与token ID 对应的向量。生成的嵌入向量提供了token的连续表示，这在训练像 LLM 这样的深度学习模型时至关重要。
# * 虽然token嵌入为每个token提供了一致的向量表示，但它们并没有考虑token在序列中的位置。为了解决这个问题，存在两种主要类型的位置嵌入：绝对位置嵌入和相对位置嵌入。
#   OpenAI 的 GPT 模型采用绝对位置嵌入，这些位置嵌入向量会与token嵌入向量相加，并在模型训练过程中进行优化。

# 处理文本数据，构建词汇表，并测试分词器的编码和解码功能
def step1(raw_text: str) -> None:
    # 文本正则分词
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    print("token count:", len(preprocessed))

    # 将preprocessed排序剔重，构建token到token_id的映射：vocab
    all_tokens  = sorted(list(preprocessed))
    # 我们可以修改分词器，以便在遇到不在词汇表中的单词时使用一个<|unk|> token。此外，我们还会在不相关的文本之间添加一个特殊的<|endoftext|> token。
    # 例如，在对多个独立文档或书籍进行GPT类大语言模型的训练时，通常会在每个文档或书籍之前插入一个token，以连接前一个文本源，这有助于大语言模型理解，
    # 尽管这些文本源在训练中是连接在一起的，但它们实际上是无关的。
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_tokens)}
    print("vocab size:", len(all_tokens))

    tokenizer1 = SimpleTokenizerV1(vocab)
    tokenizer2 = SimpleTokenizerV2(vocab)
    text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer1.encode(text)
    print("STV1 srcText:\t", text)
    print("STV1 encode:\t", ids)
    print("STV1 decode:\t", tokenizer1.decode(ids))

    # SimpleTokenizerV1 若包含没有的token会报错, SimpleTokenizerV1会用<|unk|>替换未知token
    text = "Hello, do you like tea?"
    print("STV2 srcText:\t", text)
    # print(tokenizer1.encode(text))
    print("STV2 encode:\t", tokenizer2.encode(text))
    print("STV2 decode:\t", tokenizer2.decode(tokenizer2.encode(text)))

    # --------------------
    # 这里不从0实现一个Byte pair encoding (BPE) 编码器, 使用tiktoken库的GPT-2编码器进行测试
    # tiktoken库的GPT-2编码器使用Byte Pair Encoding (BPE)算法来将文本分解成更小的单元（tokens），这些单元可以是单词、子词或字符，具体取决于文本的结构和词汇表的设计。
    # tiktoken.get_encoding("gpt2") 第一次运行时要去下载 GPT-2 的 BPE 词表，注意网络设置
    tokenizerBPE = tiktoken.get_encoding("gpt2")
    text = "Hello, do you like tea? <|endoftext|> In the sunlit terraces of someunknownPlace."
    integers = tokenizerBPE.encode(text, allowed_special={"<|endoftext|>"})
    print("tiktoken-gpt2 srcText:\t", text)
    print("tiktoken-gpt2 encode:\t", integers)
    print("tiktoken-gpt2 decode:\t", tokenizerBPE.decode(integers))

    text = "Akwirwier"
    integers = tokenizerBPE.encode(text)
    print("tiktoken-gpt2 srcText:\t", text)
    # 输出 [33901, 86, 343, 86, 959]
    # 对应 ['Ak' ,'w','ir','w','ier']
    # 字节对编码是一种基于统计的方法，它会先从整个语料库中找出最常见的字节对（byte pair），然后把这些字节对合并成一个新的单元。
    print("tiktoken-gpt2 encode:\t", integers)
    print("tiktoken-gpt2 decode:\t", tokenizerBPE.decode(integers))



# 使用滑动窗口进行数据采样
def step2(raw_text: str) -> None:
    tokenizer = tiktoken.get_encoding("gpt2")
    # BPE 分词器的 encode 方法将分词和转换为token ID 的过程合并为了一个步骤
    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    # 我们从数据集中移除前50个token以便演示，因为这会在接下来的步骤中产生稍微更有趣的文本段落
    enc_sample = enc_text[50:]
    context_size = 4
    for i in range(1, context_size + 1):
        context = enc_sample[:i]
        desired = enc_sample[i]
        print(context, "---->", desired)
        print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

def step3(raw_text: str) -> None:
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
    # 将数据加载器转换为 Python 迭代器，以便通过 Python 的内置 next() 函数获取下一个数据条目。
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    second_batch = next(data_iter)
    print(first_batch)
    print(second_batch)

# 构建嵌入层 (提供TokenID查询的功能，从一个TokenID到嵌入向量的映射)
def step4() -> None:
    vocab_size = 6  # 词汇表大小，通常是唯一token的数量, BPE分词器中有50,257个token
    output_dim = 3  # 每个token的嵌入向量维度, GPT-3模型中每个token的嵌入向量维度为12,288
    torch.manual_seed(123)
    # 创建一个包含TokenID（0-5）的嵌入层
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    # 打印嵌入层的权重矩阵（6x3），每一行代表每个token的唯一嵌入向量
    # 嵌入层的权重矩阵由比较小的随机值组成。这些值将在LLM训练过程中作为LLM优化的一部分被优化
    print(embedding_layer.weight)
    # 嵌入层实例化好了之后，可以通过它获取指定token ID的嵌入向量。例如，输入token ID 3会返回一个对应的嵌入向量，这个向量是一个3维的数值数组，代表了token ID 3在嵌入空间中的位置。
    print(embedding_layer(torch.tensor([3])))
    input_ids = torch.tensor([2, 3, 5, 1])
    # 通过输入一个token ID的序列，我们可以得到对应的嵌入向量序列。每个token ID都会被转换为一个3维的嵌入向量，这些向量可以捕捉token之间的语义关系和上下文信息。
    print(embedding_layer(input_ids))

    # 不在嵌入层的Token ID范围内的输入会报错：IndexError: index out of range in self
    # print(embedding_layer(torch.tensor(7)))

# 位置编码
def step5(raw_text: str) -> None:
    max_length = 4
    dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    # 使用滑动窗口采样, 每个输入序列的长度为max_length=4, 因此每个输入序列包含4个token ID, Batch为8, 因此输入的形状为(8, 4)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)
    
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)  # 实例化GGPT-3的BPE分词器的词汇表大小和一个较小(256)的嵌入维度
    token_embedding =  token_embedding_layer(inputs)
    # Token Embeddings shape: torch.Size([8, 4, 256])
    print("\nToken Embeddings shape:\n", token_embedding.shape)

    # 对于 GPT 模型所使用的绝对嵌入方法，我们只需创建另一个嵌入层，其维度与 token_embedding_layer 的维度相同
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    # pos_embeddings 的输入通常是一个占位符向量, torch.arange(context_length), 它包含一个从0到(最大输入长度-1)的数字序列
    # 输入文本很有可能超过支持的上下文长度，此时需要对文本进行截断
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    # Position Embeddings shape: torch.Size([4, 256])
    print("\nPosition Embeddings shape:\n", pos_embeddings.shape)

    # 我们创建的 input_embeddings, 现在可作为LLM的核心模块的输入嵌入
    input_embeddings = token_embedding + pos_embeddings
    # Input Embeddings shape: torch.Size([8, 4, 256])
    print("\nInput Embeddings shape:\n", input_embeddings.shape)

if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "the-verdict_test.txt"
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of characters:", len(raw_text))
    
    #step1(raw_text)
    #step2(raw_text)
    #step3(raw_text)
    #step4()
    step5(raw_text)
