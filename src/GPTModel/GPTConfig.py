"""
GPT 模型配置。

本章默认配置对应 GPT-2 small（约 1.24 亿参数，若输出层不与 token
embedding 共享权重，实际可训练参数会更多）。`GPT_CONFIG_SMALL` 用于本项目的
快速演示，避免每次运行都实例化完整 124M 模型。
"""

GPT_CONFIG_124M = {
    "vocab_size": 50257,        # GPT-2 BPE 分词器的词表大小
    "context_length": 1024,     # 模型一次最多能看到的 token 数
    "emb_dim": 768,             # 每个 token/position 的向量维度
    "n_heads": 12,              # 多头注意力的 head 数，必须能整除 emb_dim
    "n_layers": 12,             # TransformerBlock 重复堆叠的层数
    "drop_rate": 0.1,           # 统一 dropout 默认值
    "qkv_bias": False,          # Query/Key/Value 线性层是否使用 bias
}


GPT_CONFIG_SMALL = {
    **GPT_CONFIG_124M,
    "context_length": 32,
    "emb_dim": 64,
    "n_heads": 4,
    "n_layers": 2,
    "drop_rate": 0.1,
    # 练习 4.3：三个位置可以独立控制 dropout；未设置时会回退到 drop_rate。
    "emb_drop_rate": 0.1,       # token embedding + position embedding 之后的 dropout
    "attn_drop_rate": 0.1,      # attention weights 上的 dropout
    "shortcut_drop_rate": 0.1,  # 残差分支相加前的 dropout
}
