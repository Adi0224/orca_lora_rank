import re
import random
from typing import Optional

import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def extract_gold(answer_text: str) -> Optional[int]:
    """Parse the number after #### in GSM8K answer field."""
    match = re.search(r"####\s*(-?[\d,]+)", answer_text)
    if match:
        return int(match.group(1).replace(",", ""))
    return None


def label_mod10(answer_text: str) -> int:
    """abs(gold) % 10 — pragmatic 10-bucket discrete class for OTDD."""
    gold = extract_gold(answer_text)
    if gold is None:
        return 0
    return abs(gold) % 10


def split_easy_hard(examples, threshold: int):
    """Partition examples by |extract_gold(answer)| < threshold."""
    easy, hard = [], []
    for ex in examples:
        gold = extract_gold(ex["answer"])
        if gold is not None and abs(gold) < threshold:
            easy.append(ex)
        else:
            hard.append(ex)
    return easy, hard


def make_splits(cfg):
    """
    Returns: easy_pool, hard_train, hard_val, test
    All deterministic via cfg.data_seed. Train/test are from different GSM8K publisher splits.
    """
    ds = load_dataset("gsm8k", "main")
    train_data = list(ds["train"])
    test_data = list(ds["test"])

    # Deterministic shuffle of train split
    rng = random.Random(cfg.data_seed)
    rng.shuffle(train_data)

    # Split into easy and hard
    easy_all, hard_all = split_easy_hard(train_data, cfg.easy_answer_threshold)

    # Take easy pool
    easy_pool = easy_all[: cfg.easy_pool_samples]

    # From hard, take train + val
    hard_train = hard_all[: cfg.hard_train_samples]
    hard_val = hard_all[cfg.hard_train_samples : cfg.hard_train_samples + cfg.val_samples]

    # Test split (cap if requested)
    test_limit = cfg.test_samples if cfg.test_samples > 0 else len(test_data)
    # Deterministic shuffle of test too for consistent subset
    rng2 = random.Random(cfg.data_seed)
    rng2.shuffle(test_data)
    test = test_data[:test_limit]

    return easy_pool, hard_train, hard_val, test


def tokenize_for_lm(example: dict, tokenizer, max_length: int) -> dict:
    """
    Format as "Question:\n{q}\n\nAnswer:\n{a}", mask prompt tokens as -100 in labels.
    """
    question = example["question"]
    answer = example["answer"]

    prompt = f"Question:\n{question}\n\nAnswer:\n"
    full_text = prompt + answer

    prompt_enc = tokenizer(prompt, add_special_tokens=False)
    full_enc = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )

    input_ids = full_enc["input_ids"]
    attention_mask = full_enc["attention_mask"]

    # Mask prompt tokens
    prompt_len = len(prompt_enc["input_ids"])
    labels = [-100] * prompt_len + input_ids[prompt_len:]

    # Ensure same length
    labels = labels[: len(input_ids)]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def tokenize_for_alignment(example: dict, tokenizer, max_length: int) -> dict:
    """Prompt-only tokenization for OTDD alignment, includes label10 field."""
    question = example["question"]
    prompt = f"Question:\n{question}\n\nAnswer:\n"

    enc = tokenizer(
        prompt,
        max_length=max_length,
        truncation=True,
        add_special_tokens=False,
    )

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "label10": label_mod10(example["answer"]),
    }


class LMDataset(Dataset):
    """Dataset for language model training."""

    def __init__(self, examples: list, tokenizer, max_length: int):
        self.items = []
        for ex in examples:
            self.items.append(tokenize_for_lm(ex, tokenizer, max_length))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


class AlignmentDataset(Dataset):
    """Dataset for OTDD alignment (prompt-only + label10)."""

    def __init__(self, examples: list, tokenizer, max_length: int):
        self.items = []
        for ex in examples:
            self.items.append(tokenize_for_alignment(ex, tokenizer, max_length))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch, pad_token_id: int = 0):
    """Pad input_ids, attention_mask, labels to same length."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        if "labels" in item:
            labels.append(item["labels"] + [-100] * pad_len)

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }
    if labels:
        result["labels"] = torch.tensor(labels, dtype=torch.long)

    return result


def collate_alignment(batch, pad_token_id: int = 0):
    """Collate for alignment dataset (includes label10)."""
    max_len = max(len(item["input_ids"]) for item in batch)

    input_ids = []
    attention_mask = []
    labels10 = []

    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels10.append(item["label10"])

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "label10": torch.tensor(labels10, dtype=torch.long),
    }
