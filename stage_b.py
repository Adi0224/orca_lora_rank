import time
import torch
from torch.utils.data import DataLoader
from functools import partial

from data import LMDataset, collate_fn
from evaluate import evaluate_em


def run_stage_b(model, hard_train, hard_val, cfg):
    """
    LoRA Training loop with per-epoch validation.
    Optimizer groups: LoRA params at lora_lr, adapter params at lora_lr * 0.25.
    """
    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    # Build datasets
    train_ds = LMDataset(hard_train, tokenizer, cfg.max_length)
    val_ds = hard_val  # keep raw for evaluation

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    # Set up optimizer groups
    param_groups = []

    # LoRA params (from PEFT model)
    lora_params = []
    adapter_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "adapter" in name:
            adapter_params.append(param)
        else:
            lora_params.append(param)

    if lora_params:
        param_groups.append({"params": lora_params, "lr": cfg.lora_lr})
    if adapter_params:
        param_groups.append({"params": adapter_params, "lr": cfg.lora_lr * 0.25})

    if not param_groups:
        raise ValueError("No trainable parameters found")

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)

    # Training loop
    train_losses = []
    val_ems = []
    start_time = time.time()

    for epoch in range(cfg.lora_epochs):
        model.train()
        total_loss = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / cfg.grad_accum
            loss.backward()

            total_loss += outputs.loss.item()
            n_steps += 1

            if (step + 1) % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for g in param_groups for p in g["params"]],
                    cfg.gradient_clip,
                )
                optimizer.step()
                optimizer.zero_grad()

        # Final accumulation step if needed
        if n_steps % cfg.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in param_groups for p in g["params"]],
                cfg.gradient_clip,
            )
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = total_loss / max(n_steps, 1)
        train_losses.append(avg_loss)

        # Validation
        val_result = evaluate_em(model, val_ds, tokenizer, device, max_new_tokens=128)
        val_em = val_result["accuracy"]
        val_ems.append(val_em)

        if cfg.verbose:
            print(
                f"  Epoch {epoch + 1}/{cfg.lora_epochs} | "
                f"train_loss={avg_loss:.4f} | val_em={val_em:.4f}"
            )

    wall_seconds = time.time() - start_time

    return {
        "train_loss_per_epoch": train_losses,
        "val_em_per_epoch": val_ems,
        "wall_seconds": wall_seconds,
    }
