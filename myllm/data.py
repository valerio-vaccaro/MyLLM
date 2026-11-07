"""Corpus loading, train/validation split and batching."""

from __future__ import annotations

import re
from pathlib import Path

import torch

CHUNK = 256  # tokens per split unit; every VAL_EVERY-th chunk goes to validation
VAL_EVERY = 20  # -> ~5% validation, spread over the whole corpus (not just the last book)

_GUTENBERG_START = re.compile(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG[^\n]*\n")
_GUTENBERG_END = re.compile(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG")


def strip_gutenberg(text: str) -> str:
    """Remove the Project Gutenberg license header/footer, if present."""
    start = _GUTENBERG_START.search(text)
    if start:
        text = text[start.end():]
    end = _GUTENBERG_END.search(text)
    if end:
        text = text[: end.start()]
    return text


def clean_text(text: str) -> str:
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def load_corpus(path) -> str:
    """Read a ``.txt`` file or every top-level ``.txt`` file of a directory."""
    path = Path(path)
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(path.glob("*.txt"))
    else:
        raise FileNotFoundError(f"{path} does not exist")
    if not files:
        raise FileNotFoundError(
            f"no .txt files in {path}. Put some text there or run: python scripts/download_full_texts.py"
        )
    texts = [clean_text(strip_gutenberg(f.read_text(encoding="utf-8", errors="ignore"))) for f in files]
    return "\n\n".join(t for t in texts if t)


def split_ids(ids: list[int], chunk: int = CHUNK, val_every: int = VAL_EVERY):
    """Split a token stream into (train, val) tensors by interleaving fixed-size chunks."""
    train, val = [], []
    for n, start in enumerate(range(0, len(ids), chunk)):
        (val if n % val_every == val_every - 1 else train).extend(ids[start : start + chunk])
    return torch.tensor(train, dtype=torch.long), torch.tensor(val, dtype=torch.long)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, generator=None, device="cpu"):
    """Random (x, y) windows where y is x shifted one token to the left."""
    if len(data) <= block_size:
        raise ValueError(f"need more than block_size={block_size} tokens, got {len(data)}")
    starts = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    offsets = torch.arange(block_size)
    x = data[starts[:, None] + offsets]
    y = data[starts[:, None] + offsets + 1]
    return x.to(device), y.to(device)
