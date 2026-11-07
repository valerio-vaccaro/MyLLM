"""Charts: corpus statistics and training curves / timings (matplotlib, headless)."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt  # headless machines fall back to a non-GUI backend automatically

from .data import load_corpus  
from .tokenizer import split_words  


def plot_corpus(corpus, out_png, vocab_size: int = 4000, top: int = 20) -> Path:
    """Save :func:`corpus_figure` as a PNG."""
    return _save(corpus_figure(corpus, vocab_size, top), out_png)


def corpus_figure(corpus, vocab_size: int = 4000, top: int = 20):
    """Top words, Zipf's law, and how much text a vocabulary of a given size covers."""
    counts = Counter(split_words(load_corpus(corpus)))
    total = sum(counts.values())
    ranked = counts.most_common()
    freqs = [c for _, c in ranked]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    words, vals = zip(*ranked[:top])
    ax[0].barh(range(top), vals[::-1], color="#3b6ea5")
    ax[0].set_yticks(range(top), words[::-1])
    ax[0].set_title(f"Top {top} tokens")
    ax[0].set_xlabel("occurrences")

    ax[1].loglog(range(1, len(freqs) + 1), freqs, color="#c0504d")
    ax[1].set_title("Zipf's law: frequency vs rank")
    ax[1].set_xlabel("rank")
    ax[1].set_ylabel("count")

    cum, running = [], 0
    for f in freqs:
        running += f
        cum.append(running / total)
    ax[2].semilogx(range(1, len(cum) + 1), cum, color="#4f9d4f")
    k = min(vocab_size, len(cum))
    ax[2].axvline(k, ls="--", color="gray")
    ax[2].annotate(f"vocab={k:,}\ncovers {cum[k - 1]:.1%}", (k, cum[k - 1]), textcoords="offset points", xytext=(-90, -40))
    ax[2].set_title("Text coverage by vocabulary size")
    ax[2].set_xlabel("vocabulary size")
    ax[2].set_ylim(0, 1.02)
    fig.suptitle(f"Corpus: {total:,} tokens, {len(counts):,} distinct")
    fig.tight_layout()
    return fig


def plot_training(history_json, out_png) -> Path:
    """Save :func:`training_figure` as a PNG."""
    return _save(training_figure(history_json), out_png)


def training_figure(history_json):
    """Loss and perplexity over wall-clock time, learning rate, throughput and phase timings."""
    meta = json.loads(Path(history_json).read_text(encoding="utf-8"))
    h = meta["history"]
    t = [r["seconds"] for r in h]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))

    tr = [(r["seconds"], r["train_loss"]) for r in h if r["train_loss"] is not None]
    if tr:
        ax[0, 0].plot(*zip(*tr), label="train (smoothed)", color="#3b6ea5")
    ax[0, 0].plot(t, [r["val_loss"] for r in h], label="validation", color="#c0504d", marker="o", ms=3)
    ax[0, 0].set(title="Loss vs training time", xlabel="seconds", ylabel="cross-entropy")
    ax[0, 0].legend()

    ax[0, 1].plot(t, [math.exp(r["val_loss"]) for r in h], color="#c0504d", marker="o", ms=3)
    ax[0, 1].set_yscale("log")
    ax[0, 1].set(title="Validation perplexity (lower is better)", xlabel="seconds", ylabel="perplexity")

    ax[1, 0].plot(t, [r["lr"] for r in h], color="#7e57c2", label="learning rate")
    ax[1, 0].set(title="Learning rate and throughput", xlabel="seconds", ylabel="lr")
    twin = ax[1, 0].twinx()
    twin.plot(t[1:], [r["tokens_per_s"] for r in h[1:]], color="#f39c12")
    twin.set_ylabel("tokens / s", color="#f39c12")

    tm = meta["timings"]
    names = ["load corpus", "tokenize", "train"]
    secs = [tm["load_corpus_s"], tm["tokenize_s"], tm["train_s"]]
    bars = ax[1, 1].bar(names, secs, color=["#999", "#bbb", "#3b6ea5"])
    ax[1, 1].bar_label(bars, labels=[f"{s:.1f}s" for s in secs])
    ax[1, 1].set(title="Time per phase", ylabel="seconds")

    fig.suptitle(
        f"{meta['preset']} preset | {meta['n_params']:,} params | {meta['steps']:,} steps | "
        f"{tm['tokens_per_s']:,.0f} tok/s | best val ppl {math.exp(meta['best_val_loss']):.1f}"
    )
    fig.tight_layout()
    return fig


def _save(fig, out_png) -> Path:
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=110)
    plt.close(fig)
    return out_png
