import torch
import tiktoken

from GPTModel import (
    GPTModel,
    GPT_CONFIG_124M,
    GPT_CONFIG_SMALL,
    generate_text_simple,
)


def build_example_batch(tokenizer):
    """构造第 4 章使用的两个短文本 batch，形状为 (batch=2, tokens=4)。"""
    texts = [
        "Every effort moves you",
        "Every day holds a",
    ]
    token_ids = [torch.tensor(tokenizer.encode(text)) for text in texts]
    return torch.stack(token_ids, dim=0)


def summarize_model_parameters(model):
    """统计模型参数量，并给出书中讨论的 weight tying 对等参数量。"""
    total_params = sum(param.numel() for param in model.parameters())
    out_head_params = sum(param.numel() for param in model.out_head.parameters())
    total_size_mb = total_params * 4 / (1024 * 1024)  # float32 每个参数占 4 字节

    return {
        "total_params": total_params,
        # GPT-2 原论文统计常按输出层与 token embedding 共享权重计算。
        "weight_tying_equivalent_params": total_params - out_head_params,
        "total_size_mb": total_size_mb,
    }


def run_gpt_model_test(config=None):
    """
    第 4 章完整 GPTModel 演示。

    默认使用轻量配置，适合日常验证；传入 GPT_CONFIG_124M 可运行书中的 GPT-2 small
    结构，但会占用约 600MB 以上内存。
    """
    cfg = dict(config or GPT_CONFIG_SMALL)
    tokenizer = tiktoken.get_encoding("gpt2")
    batch = build_example_batch(tokenizer)

    torch.manual_seed(123)
    model = GPTModel(cfg)
    model.eval()  # 生成/验证阶段关闭 dropout，输出更稳定。

    with torch.no_grad():
        logits = model(batch)

    print("Input batch:\n", batch)
    print("\nOutput shape:", logits.shape)
    print("Last-token logits sample:", logits[0, -1, :8])

    stats = summarize_model_parameters(model)
    print(f"\nTotal number of parameters: {stats['total_params']:,}")
    print(
        "Parameters considering weight tying: "
        f"{stats['weight_tying_equivalent_params']:,}"
    )
    print(f"Estimated model size (float32): {stats['total_size_mb']:.2f} MB")

    start_context = "Hello, I am"
    encoded = tokenizer.encode(start_context)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    print("\nStart context:", start_context)
    print("Encoded:", encoded)

    out = generate_text_simple(
        model=model,
        idx=encoded_tensor,
        max_new_tokens=6,
        context_size=cfg["context_length"],
    )
    print("Generated token IDs:", out)
    print("Decoded text:", tokenizer.decode(out.squeeze(0).tolist()))


def run_gpt_model_124m_test():
    """运行书中 GPT-2 small 级别配置；该函数会占用明显更多内存和时间。"""
    run_gpt_model_test(GPT_CONFIG_124M)


if __name__ == "__main__":
    run_gpt_model_test()
