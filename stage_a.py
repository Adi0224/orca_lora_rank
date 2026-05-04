import time
import torch
from torch.utils.data import DataLoader

from data import AlignmentDataset, collate_alignment
from otdd_utils import compute_otdd, stabilize_features


def run_stage_a(model, easy_pool, hard_train, cfg):
    """
    OTDD Adapter Alignment (Method 2 only).
    Freeze all LM params, enable adapter gradients only.
    Align hard embeddings (through adapter) toward easy embeddings (frozen, no adapter).
    """
    device = next(model.parameters()).device
    tokenizer = model.tokenizer

    if model.adapter is None:
        raise ValueError("Stage A requires a model with adapter (method=orca_otdd)")

    # Freeze everything except adapter
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(model.adapter.parameters(), lr=cfg.alignment_lr)

    # Build alignment datasets
    easy_ds = AlignmentDataset(easy_pool, tokenizer, cfg.max_length)
    hard_ds = AlignmentDataset(hard_train, tokenizer, cfg.max_length)

    # Pre-compute source (easy) embeddings: raw Pythia embeddings (no adapter), mean-pooled, detached
    model.eval()
    source_feats_list = []
    source_labels_list = []

    easy_loader = DataLoader(
        easy_ds, batch_size=16, shuffle=False,
        collate_fn=lambda b: collate_alignment(b, tokenizer.pad_token_id),
    )

    with torch.no_grad():
        for batch in easy_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels10 = batch["label10"]

            # Raw embeddings without adapter
            embeds = model.get_embeddings(input_ids)
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

            source_feats_list.append(pooled.cpu())
            source_labels_list.append(labels10)

    source_feats = torch.cat(source_feats_list, dim=0)
    source_labels = torch.cat(source_labels_list, dim=0)

    # Hard data loader
    hard_loader = DataLoader(
        hard_ds, batch_size=16, shuffle=True,
        collate_fn=lambda b: collate_alignment(b, tokenizer.pad_token_id),
    )

    # Training loop
    loss_curve = []
    start_time = time.time()

    for epoch in range(cfg.embedder_epochs):
        model.train()
        epoch_loss = 0.0

        # Compute target (hard) embeddings through adapter
        target_feats_list = []
        target_labels_list = []

        for batch in hard_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels10 = batch["label10"]

            pooled = model.pooled_embed(input_ids, attention_mask)
            target_feats_list.append(pooled)
            target_labels_list.append(labels10.to(device))

        target_feats = torch.cat(target_feats_list, dim=0)
        target_labels = torch.cat(target_labels_list, dim=0)

        # Compute OTDD loss
        loss = compute_otdd(
            target_feats=target_feats,
            target_labels=target_labels,
            source_feats=source_feats.to(device),
            source_labels=source_labels.to(device),
            maxsamples_per_class=cfg.otdd_maxsamples_per_class,
            use_exact=cfg.otdd_use_exact,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss = loss.item()
        loss_curve.append(epoch_loss)

        if cfg.verbose:
            print(f"  Stage A epoch {epoch + 1}/{cfg.embedder_epochs} | otdd_loss={epoch_loss:.4f}")

    wall_seconds = time.time() - start_time

    # Unfreeze adapter params (they'll be used in Stage B at lower lr)
    # But re-enable requires_grad for all model params that will be used
    for param in model.parameters():
        param.requires_grad = False
    for param in model.adapter.parameters():
        param.requires_grad = True

    return {
        "loss_curve": loss_curve,
        "wall_seconds": wall_seconds,
    }
