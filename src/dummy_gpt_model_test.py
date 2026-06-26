import torch
import torch.nn as nn
import tiktoken
from GPTModel.LayerNorm import LayerNorm
from GPTModel.GELU import GELU
from GPTModel.FeedForwardGELU import FeedForwardGELU
from GPTModel.TransformerBlock import TransformerBlock

class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])      # 为 TransformerBlock 设置占位符
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])                        # 为 LayerNorm 设置占位符
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# 一个简单的占位类，后续将被真正的 TransformerBlock 替换
class DummyTransformerBlock(nn.Module):                                        
    def __init__(self, cfg):
        super().__init__()

    # 该模块无实际操作，仅原样返回输入
    def forward(self, x):                                                       
        return x

# 一个简单的占位类，后续将被真正的 DummyLayerNorm 替换
class DummyLayerNorm(nn.Module):  
    # 此处的参数仅用于模拟LayerNorm接口                                              
    def __init__(self, normalized_shape, eps=1e-5):                             
        super().__init__()

    def forward(self, x):
        return x

# 手动LayerNorm实现，计算均值和方差，并进行归一化
def run_dummy_gpt_model_with_layer_norm(config):
    
    # 创建gpt-2 tokenizer实例，构造输入batch
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = []
    txt1 = "Every effort moves you"
    txt2 = "Every day holds a"
    batch.append(torch.tensor(tokenizer.encode(txt1)))
    batch.append(torch.tensor(tokenizer.encode(txt2)))
    batch = torch.stack(batch, dim=0)
    print("Input batch:\n", batch)

    torch.manual_seed(123)
    model = DummyGPTModel(config)
    # 前向传播
    logits = model(batch)
    print("Dummy GPT Model output:\n", logits)
    print("Dummy GPT Model output shape:\n", logits.shape)

    #  使用层归一化对激活值进行标准化
    print("\nLayer Normalization Example:")
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5)
    layer = nn.Sequential(nn.Linear(5, 6), nn.ReLU())
    output_example = layer(batch_example)
    print("Example output:\n", output_example)
    # 归一化后的输出的均值和方差
    mean = output_example.mean(dim=-1, keepdim=True)
    var = output_example.var(dim=-1, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)
    out_norm = (output_example - mean) / torch.sqrt(var)
    # 归一化后的输出，均值接近0，方差接近1
    mean = out_norm.mean(dim=-1, keepdim=True)
    var = out_norm.var(dim=-1, keepdim=True)
    print("Normalized layer outputs:\n", out_norm)
    print("Normalized output mean:\n", mean)
    print("Normalized output variance:\n", var)

    torch.set_printoptions(sci_mode=False)
    print("Mean:\n", mean)
    print("Variance:\n", var)


# 运行LayerNorm测试，验证均值和方差是否接近0和1
def run_layer_norm_test():
    torch.manual_seed(123)
    batch_example = torch.randn(2, 5) 
    print("LayerNorm input:\n", batch_example)

    # 这里需要导入 LayerNorm 类本身，而不是 GPTModel.LayerNorm 这个模块
    ln = LayerNorm(emb_dim=5)
    out_ln = ln(batch_example)
    print("LayerNorm output:\n", out_ln)

    mean = out_ln.mean(dim=-1, keepdim=True)
    var = out_ln.var(dim=-1, unbiased=False, keepdim=True)
    print("Mean:\n", mean)
    print("Variance:\n", var)


# ReLU 是一个分段线性函数，输入为正时输出输入值本身，否则输出零。而 GELU 是一种平滑的非线性函数，它近似于 ReLU，但在负值上也具有非零梯度。
def run_GELU_test(testGELU, config):
    if testGELU:
        import matplotlib.pyplot as plt

        gelu, relu = GELU(), torch.nn.ReLU()

        # Some sample data
        x = torch.linspace(-3, 3, 100)
        y_gelu, y_relu = gelu(x), relu(x)
        plt.figure(figsize=(8, 3))
        for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
            plt.subplot(1, 2, i)
            plt.plot(x, y)
            plt.title(f"{label} activation function")
            plt.xlabel("x")
            plt.ylabel(f"{label}(x)")
            plt.grid(True)
        plt.tight_layout()
        plt.show()

    ffn = FeedForwardGELU(config)
    x = torch.rand(2, 3, 768)
    out = ffn(x)
    
    print("FeedForwardGELU output shape:", out.shape)
    print("FeedForwardGELU out:\n", out)

class ExampleDeepNeuralNetwork(nn.Module):
    # use_shortcut参数用于控制是否在每一层的输出中添加快捷连接（skip connection）。
    # 如果use_shortcut为True，则在每一层的输出中添加输入x与当前层输出的和，从而形成快捷连接；如果为False，则直接使用当前层的输出作为下一层的输入。
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            # Implement 5 layers
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU())
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute the output of the current layer
            layer_output = layer(x)
            # Check if shortcut can be applied
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output
        return x

def print_gradients(model, x):
    # Forward pass
    output = model(x)
    target = torch.tensor([[0.]])

    # Calculate loss based on how close the target
    # and output are
    loss = nn.MSELoss()
    loss = loss(output, target)

    # Backward pass to calculate the gradients
    loss.backward()
    for name, param in model.named_parameters():
        if 'weight' in name:
            # Print the mean absolute gradient of the weights
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

# 初始化一个没有快捷连接的神经网络，其中每一层都被初始化为接受 3 个输入值并返回 3 个输出值。最后一层则返回一个单一的输出值
# 快捷连接的两个重要的作用：
# 1. 保持信息（或者说是特征）流畅传递
# 2. 缓解梯度消失问题
def run_example_deep_neural_network_test():
    layer_sizes = [3, 3, 3, 3, 3, 1]
    sample_input = torch.tensor([[1., 0., -1.]])
    torch.manual_seed(123)
    model_without_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=False
    )
    print("Gradients for model without shortcut:")
    print_gradients(model_without_shortcut, sample_input)

    torch.manual_seed(123)
    model_with_shortcut = ExampleDeepNeuralNetwork(
        layer_sizes, use_shortcut=True
    )
    print("Gradients for model with shortcut:")
    print_gradients(model_with_shortcut, sample_input)


def run_dummy_gpt_model_connect_att_with_linear(config):
    torch.manual_seed(123)
    x = torch.rand(2, 4, 768)
    trf_block = TransformerBlock(cfg=config)
    output = trf_block(x)

    print("Input Shape:", x.shape)
    print("Output Shape:", output.shape)
    print("Output:\n", output)



def run_dummy_gpt_model_test():
    GPT_CONFIG_124M = {
        "vocab_size": 50257,        # Vocabulary size：BPE 分词器使用的 50,257 个词汇的词表大小
        "context_length": 1024,     # Context length：模型所能处理的最大输入 token 数
        "emb_dim": 768,             # Embedding dimension：表示嵌入维度，将每个 token 转换为 768 维的向量
        "n_heads": 12,              # Number of attention heads：指定多头注意力机制中并行注意力头的数量
        "n_layers": 12,             # Number of layers：指定模型中 Transformer 模块的层数
        "drop_rate": 0.1,           # Dropout rate：表示 dropout 机制的强度（例如，0.1 表示丢弃 10% 的隐藏单元），用于防止过拟合
        "qkv_bias": False           # Query-Key-Value bias：参数决定是否在多头注意力的查询、键和值的线性层中加入偏置向量。我们最初会禁用该选项，以遵循现代大语言模型的标准
    }


    # run_dummy_gpt_model_with_layer_norm(config=GPT_CONFIG_124M)
    # run_layer_norm_test()
    # run_GELU_test(testGELU=False, config=GPT_CONFIG_124M)
    # run_example_deep_neural_network_test()
    run_dummy_gpt_model_connect_att_with_linear(config=GPT_CONFIG_124M)
