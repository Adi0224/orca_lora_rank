import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


class InputAdapter(nn.Module):
    """Per-token transform applied to Pythia embeddings before transformer."""

    def __init__(self, hidden_size: int, bottleneck: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()
        return (x_float + self.net(x_float)).to(orig_dtype)


class PythiaWithAdapter(nn.Module):
    """Pythia wrapper with optional InputAdapter."""

    def __init__(self, model_name: str, adapter_bottleneck: int = 0, device: str = "cpu"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hidden_size = self.model.config.hidden_size
        self.adapter = None
        if adapter_bottleneck > 0:
            self.adapter = InputAdapter(hidden_size, adapter_bottleneck)

        self._device = device

    def get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings from the model's embedding layer."""
        return self.model.gpt_neox.embed_in(input_ids)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Embed → adapter (if exists) → LM forward."""
        embeds = self.get_embeddings(input_ids)
        if self.adapter is not None:
            embeds = self.adapter(embeds)

        outputs = self.model(
            inputs_embeds=embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    def pooled_embed(self, input_ids, attention_mask=None):
        """Embed → adapter → mean pool (for OTDD)."""
        embeds = self.get_embeddings(input_ids)
        if self.adapter is not None:
            embeds = self.adapter(embeds)

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (embeds * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            pooled = embeds.mean(dim=1)
        return pooled

    def generate(self, input_ids, attention_mask=None, max_new_tokens=128):
        """Greedy decode using embeddings through adapter."""
        device = input_ids.device

        for _ in range(max_new_tokens):
            embeds = self.get_embeddings(input_ids)
            if self.adapter is not None:
                embeds = self.adapter(embeds)

            outputs = self.model(
                inputs_embeds=embeds,
                attention_mask=attention_mask,
            )
            next_token_logits = outputs.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            # Stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            input_ids = torch.cat([input_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(1, 1, device=device, dtype=attention_mask.dtype)],
                    dim=-1,
                )

        return input_ids


def attach_lora(model: PythiaWithAdapter, cfg) -> nn.Module:
    """Attach LoRA via PEFT to the inner causal LM."""
    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model.model = get_peft_model(model.model, lora_config)
    return model


def load_model(cfg) -> PythiaWithAdapter:
    """Load Pythia with optional adapter based on method."""
    adapter_bottleneck = cfg.adapter_bottleneck if cfg.method == "orca_otdd" else 0
    model = PythiaWithAdapter(
        model_name=cfg.model_name,
        adapter_bottleneck=adapter_bottleneck,
        device=cfg.device,
    )
    return model
