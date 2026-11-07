"""myllm - a tiny GPT language model you can train from scratch in ~10 minutes on a CPU."""

from .generate import generate, load_checkpoint
from .model import GPT, GPTConfig
from .tokenizer import WordTokenizer

__all__ = ["GPT", "GPTConfig", "WordTokenizer", "generate", "load_checkpoint"]
__version__ = "0.2.0"
