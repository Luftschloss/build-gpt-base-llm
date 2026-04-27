"""
LLM-GPT 模块测试入口
提供统一管理各个模块测试的功能。
"""

import argparse
import sys
from pathlib import Path

# 确保能找到同目录下的模块
_src_dir = Path(__file__).resolve().parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import torch
import tiktoken

from tokenizer_test import SimpleTokenizerV1, SimpleTokenizerV2
from dataset_loader_test import create_dataloader_v1
from embedding_test import step1, step2, step3, step4, step5
import attention_test as attention

# ---------- 数据集路径 ----------
_dataset_path = Path(__file__).resolve().parent.parent / "datasets" / "the-verdict_test.txt"

def _load_raw_text() -> str:
    with _dataset_path.open("r", encoding="utf-8") as f:
        return f.read()


# ========== 各模块测试入口 ==========

def run_tokenizer():
    print("=" * 60)
    print("[1/4] 分词器模块测试 (SimpleTokenizerV1 / V2 + BPE)")
    print("=" * 60)
    raw_text = _load_raw_text()
    step1(raw_text)


def run_dataloader():
    print("=" * 60)
    print("[2/4] 数据加载器模块测试 (GPTDatasetV1 + DataLoader)")
    print("=" * 60)
    raw_text = _load_raw_text()
    step3(raw_text)


def run_embedding():
    print("=" * 60)
    print("[3/4] 嵌入层模块测试 (Token Embedding + Position Embedding)")
    print("=" * 60)
    raw_text = _load_raw_text()
    step5(raw_text)


def run_attention():
    print("=" * 60)
    print("[4/4] 注意力机制模块测试")
    print("=" * 60)
    attention.main()


def run_all():
    """顺序运行所有模块测试"""
    raw_text = _load_raw_text()

    print("=" * 60)
    print("LLM-GPT 全部模块测试")
    print("=" * 60)

    print("\n========== 1/4  分词器 ==========")
    step1(raw_text)

    print("\n========== 2/4  数据加载器 ==========")
    step3(raw_text)

    print("\n========== 3/4  嵌入层 ==========")
    step5(raw_text)

    print("\n========== 4/4  注意力机制 ==========")
    attention.main(raw_text)

    print("\n" + "=" * 60)
    print("全部模块测试完成 !")
    print("=" * 60)


# ========== CLI 入口 ==========

def main():
    parser = argparse.ArgumentParser(description="LLM-GPT 模块测试管理器")
    parser.add_argument(
        "module",
        nargs="?",
        default="all",
        choices=["tokenizer", "dataloader", "embedding", "attention", "all"],
        help="要测试的模块（默认: all）",
    )
    args = parser.parse_args()

    runners = {
        "tokenizer": run_tokenizer,
        "dataloader": run_dataloader,
        "embedding": run_embedding,
        "attention": run_attention,
        "all": run_all,
    }
    runners[args.module]()


if __name__ == "__main__":
    main()
