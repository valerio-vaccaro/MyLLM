"""Load a checkpoint and sample text from it."""

from __future__ import annotations

import torch

from .model import GPT, GPTConfig
from .tokenizer import SPECIALS, WordTokenizer


def load_checkpoint(path, device="cpu"):
    """Return ``(model, tokenizer, info)``; the checkpoint carries its own config and vocabulary."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, WordTokenizer.from_dict(ckpt["tokenizer"]), ckpt.get("info", {})


def filter_logits(logits: torch.Tensor, top_k: int | None = None, top_p: float | None = None) -> torch.Tensor:
    """Mask (set to -inf) logits outside the top-k / nucleus top-p set. 1-D input."""
    logits = logits.clone()
    if top_k is not None and 0 < top_k < logits.numel():
        kth = torch.topk(logits, top_k).values[-1]
        logits[logits < kth] = float("-inf")
    if top_p is not None and 0 < top_p < 1:
        sorted_logits, order = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        # drop a token if the mass *before* it already reaches top_p (always keeps the best one)
        drop = (torch.cumsum(probs, dim=-1) - probs) >= top_p
        logits[order[drop]] = float("-inf")
    return logits


@torch.no_grad()
def generate(
    model: GPT,
    tokenizer: WordTokenizer,
    prompt: str = "",
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int | None = 40,
    top_p: float | None = 0.95,
    seed: int | None = None,
) -> str:
    """Sample a continuation of ``prompt``. ``temperature=0`` means greedy decoding."""
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)
    else:
        gen.seed()

    ids = [tokenizer.bos_id] + tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        ctx = torch.tensor([ids[-model.cfg.block_size :]], device=device)
        logits = model(ctx)[0][0, -1].float().cpu()
        logits[: len(SPECIALS)] = float("-inf")  # special tokens are never trained targets
        if temperature == 0:
            nxt = int(torch.argmax(logits))
        else:
            logits = filter_logits(logits / temperature, top_k, top_p)
            nxt = int(torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=gen))
        ids.append(nxt)
    return tokenizer.decode(ids[1:])
