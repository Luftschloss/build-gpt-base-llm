# LLM-GPT

从零实现类 GPT 的大语言模型学习项目，基于 PyTorch，逐步构建各个核心模块。

## 项目结构

```
├── datasets/
│   └── the-verdict_test.txt    # 训练示例文本
├── src/
│   ├── main.py                 # 统一测试入口（主入口）
│   ├── tokenizer_test.py       # 分词器（SimpleTokenizerV1/V2）
│   ├── dataset_loader_test.py  # 数据集与 DataLoader
│   ├── embedding_test.py       # Token 嵌入 + 位置编码
│   ├── attention_test.py       # 注意力机制
│   ├── self_attention.py       # 自注意力、因果注意力、多头注意力模块
│   └── DummyGPTModel.py        # Dummy GPT 模型结构测试
```

## 环境配置

```bash
# 首次创建 conda 环境
conda env create -f environment.yml

# 已创建环境后同步新增依赖
conda env update -f environment.yml --prune

# 激活环境
conda activate llm-gpt
```

## 使用方法

通过 `main.py` 统一管理各模块测试：

```bash
# 测试所有模块
python src/main.py

# 测试指定模块
python src/main.py tokenizer     # 分词器
python src/main.py dataloader    # 数据加载器
python src/main.py embedding     # 嵌入层
python src/main.py attention     # 注意力机制
python src/main.py dummy_gpt     # Dummy GPT 模型结构
```

## 模块说明

| 模块 | 功能 | 状态 |
|------|------|------|
| tokenizer_test | SimpleTokenizerV1/V2 + BPE (tiktoken) | ✅ |
| dataset_loader_test | GPTDatasetV1 + 滑动窗口 DataLoader | ✅ |
| embedding_test | Token Embedding + 绝对位置编码 | ✅ |
| attention_test / self_attention | 自注意力、因果注意力、多头注意力 | ✅ |
| DummyGPTModel | GPT 外层结构占位模型与 logits shape 测试 | ✅ |
