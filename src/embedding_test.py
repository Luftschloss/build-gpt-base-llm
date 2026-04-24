
from pathlib import Path
import re
from dataset_loader_test import create_dataloader_v1
from tokenizer_test import SimpleTokenizerV1, SimpleTokenizerV2
import tiktoken

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


if __name__ == "__main__":
    dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "the-verdict_test.txt"
    with dataset_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of characters:", len(raw_text))
    
    #step1(raw_text)
    #step2(raw_text)
    step3(raw_text)
