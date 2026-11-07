"""Evaluate a checkpoint: loss, perplexity and next-token accuracy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .data import load_corpus, split_ids


@dataclass
class EvalResult:
    loss: float
    perplexity: float
    accuracy: float
    tokens: int


@torch.no_grad()
def evaluate_ids(model, ids, batch_size: int = 32, max_tokens: int | None = None) -> EvalResult:
    """Score a token stream with non-overlapping windows of ``block_size`` tokens."""
    model.eval()
    device = next(model.parameters()).device
    block = model.cfg.block_size
    ids = torch.as_tensor(ids, dtype=torch.long)
    if max_tokens:
        ids = ids[:max_tokens]
    n = (len(ids) - 1) // block
    if n < 1:
        raise ValueError(f"need at least {block + 1} tokens to evaluate, got {len(ids)}")
    data = ids[: n * block + 1].to(device)
    x, y = data[:-1].view(n, block), data[1:].view(n, block)

    total_loss = correct = 0.0
    for i in range(0, n, batch_size):
        xb, yb = x[i : i + batch_size], y[i : i + batch_size]
        logits, loss = model(xb, yb)
        total_loss += loss.item() * yb.numel()
        correct += (logits.argmax(-1) == yb).sum().item()
    count = n * block
    avg = total_loss / count
    return EvalResult(avg, math.exp(avg), correct / count, count)


def evaluate_text(model, tokenizer, text: str, **kw) -> EvalResult:
    return evaluate_ids(model, tokenizer.encode(text), **kw)


def evaluate_heldout(model, tokenizer, corpus, **kw) -> EvalResult:
    """Evaluate on the validation split that ``train`` held out from ``corpus``."""
    _, val = split_ids(tokenizer.encode(load_corpus(corpus)))
    return evaluate_ids(model, val, **kw)
