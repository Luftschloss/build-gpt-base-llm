import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        # 将整个文本进行分词
        token_ids = tokenizer.encode(txt)

        # 使用滑动窗口将书籍分块为最大长度的重叠序列。
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    # 返回数据集的总行数
    def __len__(self):
        return len(self.input_ids)

    # 从数据集中返回指定行
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")
    # 创建GPTDatasetV1类
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
          dataset,
          batch_size=batch_size,
          shuffle=shuffle,
          drop_last=drop_last,                                        # drop_last=True会在最后一批次小于指定的batch_size时丢弃该批次，以防止训练期间的损失峰值
          num_workers=num_workers                                       # 用于预处理的CPU进程数量
    )
    return dataloader