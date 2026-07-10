import torch


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """
    使用贪婪解码逐步生成 token ID。

    Args:
        model: GPTModel 实例。
        idx: 当前上下文 token ID，形状为 (batch, n_tokens)。
        max_new_tokens: 需要追加生成的 token 数。
        context_size: 模型支持的最大上下文长度。

    Returns:
        拼接了新 token 的 idx，形状为 (batch, n_tokens + max_new_tokens)。
    """
    for _ in range(max_new_tokens):
        # 若上下文过长，只保留最后 context_size 个 token 输入模型。
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)

        # 只使用最后一个时间步预测下一个 token。
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)

        # 将本轮预测结果接到上下文末尾，供下一轮继续生成。
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
