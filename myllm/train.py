"""Train a GPT on a text corpus under a wall-clock budget."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .data import get_batch, load_corpus, split_ids
from .model import GPT, GPTConfig
from .tokenizer import WordTokenizer

# name -> (model kwargs, default minutes). "default" is the ~10 minute recipe.
# Sized for a plain laptop CPU (~5k tokens/s for "default"); the output layer dominates cost, so vocab is small.
PRESETS = {
    "tiny": (dict(n_layer=2, n_head=2, n_embd=64, block_size=32), dict(vocab_size=2000, batch_size=16), 1.0),
    "fast": (dict(n_layer=2, n_head=4, n_embd=128, block_size=48), dict(vocab_size=3000, batch_size=16), 3.0),
    "default": (dict(n_layer=3, n_head=4, n_embd=128, block_size=64), dict(vocab_size=4000, batch_size=16), 10.0),
}


@dataclass
class TrainConfig:
    preset: str = "default"
    vocab_size: int | None = None  # None -> preset
    minutes: float | None = None  # wall-clock budget; None -> preset default
    max_steps: int | None = None  # optional hard step limit (used by tests)
    batch_size: int | None = None  # None -> preset
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 20
    seed: int = 1337
    device: str = "cpu"
    log: bool = True


@dataclass
class TrainResult:
    path: Path
    steps: int
    train_loss: float
    val_loss: float
    seconds: float
    n_params: int
    history: list = field(default_factory=list)
    timings: dict = field(default_factory=dict)


def lr_at(progress: float, step: int, cfg: TrainConfig) -> float:
    """Linear warmup then cosine decay. ``progress`` in [0, 1] is time/step fraction used."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = min(max(progress, 0.0), 1.0)
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cos)


@torch.no_grad()
def estimate_loss(model, data, cfg: TrainConfig, generator, batch_size) -> float:
    model.eval()
    losses = []
    for _ in range(cfg.eval_batches):
        x, y = get_batch(data, batch_size, model.cfg.block_size, generator, cfg.device)
        losses.append(model(x, y)[1].item())
    model.train()
    return sum(losses) / len(losses)


def save_checkpoint(path, model, tokenizer, info: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_config": asdict(model.cfg),
            "model_state": model.state_dict(),
            "tokenizer": tokenizer.to_dict(),
            "info": info,
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a crash never leaves a half-written checkpoint


def history_path_for(checkpoint) -> Path:
    """Where the training log (JSON) of a checkpoint is written."""
    return Path(checkpoint).with_suffix(".history.json")


def train(corpus, out, cfg: TrainConfig | None = None) -> TrainResult:
    """Tokenize ``corpus`` (file or directory), train, and save the best checkpoint to ``out``."""
    cfg = cfg or TrainConfig()
    if cfg.preset not in PRESETS:
        raise ValueError(f"unknown preset {cfg.preset!r}; choose from {sorted(PRESETS)}")
    model_kwargs, defaults, default_minutes = PRESETS[cfg.preset]
    vocab_size = cfg.vocab_size or defaults["vocab_size"]
    batch_size = cfg.batch_size or defaults["batch_size"]
    budget = (cfg.minutes if cfg.minutes is not None else default_minutes) * 60
    say = print if cfg.log else (lambda *a, **k: None)

    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)

    t0 = time.time()
    text = load_corpus(corpus)
    t_load = time.time() - t0
    tokenizer = WordTokenizer.train(text, vocab_size)
    train_ids, val_ids = split_ids(tokenizer.encode(text))
    block = model_kwargs["block_size"]
    if len(train_ids) <= block + 1 or len(val_ids) <= block + 1:
        raise ValueError(
            f"corpus too small: {len(train_ids)} train / {len(val_ids)} val tokens for block_size {block}"
        )
    t_prep = time.time() - t0 - t_load
    say(f"corpus: {len(train_ids) + len(val_ids):,} tokens, vocab {tokenizer.vocab_size:,}")

    model = GPT(GPTConfig(vocab_size=tokenizer.vocab_size, **model_kwargs)).to(cfg.device)
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )
    say(f"model: {model.num_params():,} parameters, budget {budget / 60:.1f} min on {cfg.device}")

    start = time.time()
    best_val = float("inf")
    history, step, tokens_seen = [], 0, 0
    train_loss = float("nan")  # exponential moving average of the batch loss
    model.train()
    while True:
        elapsed = time.time() - start
        progress = elapsed / budget if budget > 0 else 1.0
        if cfg.max_steps is not None:
            progress = max(progress, step / cfg.max_steps)
        done = progress >= 1.0 or (cfg.max_steps is not None and step >= cfg.max_steps)

        if done or step % cfg.eval_every == 0:
            val_loss = estimate_loss(model, val_ids, cfg, gen, batch_size)
            history.append({
                "step": step, "seconds": elapsed, "train_loss": None if math.isnan(train_loss) else train_loss,
                "val_loss": val_loss, "lr": lr_at(progress, step, cfg), "tokens": tokens_seen,
                "tokens_per_s": tokens_seen / elapsed if elapsed > 0 else 0.0,
            })
            say(f"step {step:5d} | {elapsed:6.1f}s | train {train_loss:.3f} | val {val_loss:.3f} | ppl {math.exp(val_loss):.1f}")
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(
                    out, model, tokenizer,
                    {"steps": step, "val_loss": val_loss, "preset": cfg.preset, "seed": cfg.seed},
                )
        if done:
            break

        lr = lr_at(progress, step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_ids, batch_size, block, gen, cfg.device)
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        train_loss = loss.item() if math.isnan(train_loss) else 0.9 * train_loss + 0.1 * loss.item()
        step += 1
        tokens_seen += batch_size * block

    seconds = time.time() - start
    say(f"done: {step} steps in {seconds:.0f}s, best val loss {best_val:.3f} (ppl {math.exp(best_val):.1f}) -> {out}")
    timings = {"load_corpus_s": t_load, "tokenize_s": t_prep, "train_s": seconds,
               "tokens_per_s": tokens_seen / seconds if seconds else 0.0}
    meta = {
        "preset": cfg.preset, "timings": timings, "history": history, "steps": step,
        "n_params": model.num_params(), "vocab_size": tokenizer.vocab_size, "batch_size": batch_size,
        "corpus_tokens": len(train_ids) + len(val_ids), "best_val_loss": best_val,
    }
    history_path = history_path_for(out)
    history_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return TrainResult(Path(out), step, train_loss, best_val, seconds, model.num_params(), history, timings)
