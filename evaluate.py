import re
import torch
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from data import extract_gold, LMDataset, collate_fn


def extract_prediction(text: str):
    """Extract final number from generation. Look for #### <num> or last number."""
    # First try #### pattern
    match = re.search(r"####\s*(-?[\d,]+)", text)
    if match:
        return int(match.group(1).replace(",", ""))

    # Fall back to last number in text
    numbers = re.findall(r"-?\d+\.?\d*", text)
    if numbers:
        last = numbers[-1]
        num = float(last)
        # Return int if it's a whole number, float otherwise
        return int(num) if num == int(num) else num

    return None


def evaluate_em(model, examples: list, tokenizer, device, max_new_tokens: int = 128):
    """
    Greedy generation + exact match evaluation.

    Args:
        model: PythiaWithAdapter (possibly with LoRA)
        examples: list of raw GSM8K examples with 'question' and 'answer' fields
        tokenizer: tokenizer
        device: torch device
        max_new_tokens: max tokens to generate

    Returns:
        dict with accuracy, correct count, total, and per-sample predictions
    """
    model.eval()
    predictions = []
    correct = 0
    total = 0

    with torch.no_grad():
        for ex in examples:
            question = ex["question"]
            gold = extract_gold(ex["answer"])

            prompt = f"Question:\n{question}\n\nAnswer:\n"
            enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
            )

            # Decode only newly generated tokens
            generated_ids = output_ids[0, input_ids.shape[1]:]
            generation = tokenizer.decode(generated_ids, skip_special_tokens=True)

            predicted_num = extract_prediction(generation)
            is_correct = (predicted_num is not None and gold is not None and predicted_num == gold)

            if is_correct:
                correct += 1
            total += 1

            predictions.append({
                "prompt": prompt,
                "generation": generation[:300],
                "predicted_num": predicted_num,
                "gold": gold,
                "correct": is_correct,
            })

    accuracy = correct / max(total, 1)
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "predictions": predictions,
    }


def evaluate_ce_loss(model, examples: list, tokenizer, device, max_length: int = 256):
    """
    Compute test cross-entropy loss (teacher-forced, same as training loss but on test data).
    """
    model.eval()
    dataset = LMDataset(examples, tokenizer, max_length)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # Count non-masked tokens
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return {"ce_loss": avg_loss, "total_tokens": total_tokens}
