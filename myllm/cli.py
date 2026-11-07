"""Command line: ``python -m myllm {train,generate,eval,info}``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .evaluate import evaluate_heldout, evaluate_text
from .generate import generate, load_checkpoint
from .plots import plot_corpus, plot_training
from .train import PRESETS, TrainConfig, history_path_for, train

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "raw"
DEFAULT_MODEL = ROOT / "models" / "model.pt"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="myllm", description="Train and sample a tiny GPT on your CPU.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="tokenize a corpus and train a model")
    t.add_argument("--data", default=str(DEFAULT_DATA), help="text file or directory of .txt files")
    t.add_argument("--out", default=str(DEFAULT_MODEL), help="checkpoint path")
    t.add_argument("--preset", choices=sorted(PRESETS), default="default")
    t.add_argument("--minutes", type=float, help="time budget (default: preset, 10 for 'default')")
    t.add_argument("--max-steps", type=int, help="stop after this many steps")
    t.add_argument("--vocab", type=int, help="vocabulary size (default: preset)")
    t.add_argument("--batch", type=int, help="batch size (default: preset)")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--device", default="cpu")

    g = sub.add_parser("generate", help="sample text from a checkpoint")
    g.add_argument("prompt", nargs="?", default="")
    g.add_argument("--model", default=str(DEFAULT_MODEL))
    g.add_argument("--length", type=int, default=60, help="tokens to generate")
    g.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int)

    e = sub.add_parser("eval", help="perplexity on held-out data (or on --text-file)")
    e.add_argument("--model", default=str(DEFAULT_MODEL))
    e.add_argument("--data", default=str(DEFAULT_DATA), help="corpus used for training (held-out split)")
    e.add_argument("--text-file", help="evaluate on this file instead")

    pl = sub.add_parser("plot", help="write charts (training curves, timings, corpus stats) as PNGs")
    pl.add_argument("--model", default=str(DEFAULT_MODEL))
    pl.add_argument("--data", default=str(DEFAULT_DATA))
    pl.add_argument("--out-dir", default=str(ROOT / "outputs"))

    i = sub.add_parser("info", help="show checkpoint details")
    i.add_argument("--model", default=str(DEFAULT_MODEL))
    return p


def _need_model(path) -> bool:
    if Path(path).is_file():
        return True
    print(f"no model at {path}. Train one first: python -m myllm train", file=sys.stderr)
    return False


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)

    if a.cmd == "train":
        cfg = TrainConfig(preset=a.preset, vocab_size=a.vocab, minutes=a.minutes, max_steps=a.max_steps,
                          batch_size=a.batch, lr=a.lr, seed=a.seed, device=a.device)
        try:
            train(a.data, a.out, cfg)
        except (FileNotFoundError, ValueError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        return 0

    if a.cmd == "plot":
        out = Path(a.out_dir)
        hist = history_path_for(a.model)
        if not hist.is_file():
            print(f"no training log at {hist}. Train first: python -m myllm train", file=sys.stderr)
            return 1
        print(plot_training(hist, out / "training.png"))
        try:
            print(plot_corpus(a.data, out / "corpus.png"))
        except FileNotFoundError as err:
            print(f"skipping corpus chart: {err}", file=sys.stderr)
        return 0

    if not _need_model(a.model):
        return 1
    model, tok, info = load_checkpoint(a.model)

    if a.cmd == "generate":
        print(generate(model, tok, a.prompt, a.length, a.temperature, a.top_k, a.top_p, a.seed))
    elif a.cmd == "eval":
        try:
            if a.text_file:
                r = evaluate_text(model, tok, Path(a.text_file).read_text(encoding="utf-8", errors="ignore"))
            else:
                r = evaluate_heldout(model, tok, a.data)
        except (FileNotFoundError, ValueError) as err:
            print(f"error: {err}", file=sys.stderr)
            return 1
        print(f"tokens {r.tokens:,} | loss {r.loss:.3f} | perplexity {r.perplexity:.1f} | accuracy {r.accuracy:.1%}")
    elif a.cmd == "info":
        c = model.cfg
        print(f"{a.model}: {model.num_params():,} params, {c.n_layer} layers, {c.n_head} heads, "
              f"dim {c.n_embd}, context {c.block_size}, vocab {c.vocab_size}")
        print(f"training info: {info}")
    return 0
