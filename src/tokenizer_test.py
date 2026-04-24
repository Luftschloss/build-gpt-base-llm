# a simple text tokenizer
import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        # 将词汇表作为类属性存储，以方便在 encode 和 decode 方法中访问 
        self.str_to_int = vocab
        # 创建一个反向词汇表，将token ID 映射回原始的文本token 
        self.int_to_str = {i:s for s,i in vocab.items()}

    # 将输入文本转换为token ID
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # 将token ID 还原为文本
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 在指定的标点符号前去掉空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        #A 用 <|unk|> tokens替换未知词汇
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        # 在指定标点符号前替换空格
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
