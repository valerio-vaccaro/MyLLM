"""Word-level tokenizer: lowercases text and splits words and punctuation."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
SPECIALS = (PAD, UNK, BOS, EOS)
_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)?|[^\s\w]")
_NO_SPACE_BEFORE = set(".,;:!?)]}%’'")
_NO_SPACE_AFTER = set("([{")


def split_words(text: str) -> list[str]:
    """Split text into lowercase word and punctuation tokens."""
    return _TOKEN_RE.findall(text.lower())


class WordTokenizer:
    """Maps words to ids. Ids 0-3 are reserved for <pad>, <unk>, <bos>, <eos>."""

    def __init__(self, itos: list[str]):
        if tuple(itos[: len(SPECIALS)]) != SPECIALS:
            raise ValueError(f"vocabulary must start with {SPECIALS}")
        if len(set(itos)) != len(itos):
            raise ValueError("vocabulary contains duplicate tokens")
        self.itos = list(itos)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    @classmethod
    def train(cls, text: str, vocab_size: int = 8000) -> "WordTokenizer":
        """Build a vocabulary from the ``vocab_size - 4`` most frequent tokens."""
        if vocab_size <= len(SPECIALS):
            raise ValueError(f"vocab_size must be > {len(SPECIALS)}")
        counts = Counter(split_words(text))
        for special in SPECIALS:  # literal "<pad>" in the corpus must not collide
            counts.pop(special, None)
        words = [w for w, _ in counts.most_common(vocab_size - len(SPECIALS))]
        return cls(list(SPECIALS) + words)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    pad_id = property(lambda self: 0)
    unk_id = property(lambda self: 1)
    bos_id = property(lambda self: 2)
    eos_id = property(lambda self: 3)

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(w, unk) for w in split_words(text)]

    def decode(self, ids, skip_special: bool = True) -> str:
        words = []
        for i in ids:
            i = int(i)
            if not 0 <= i < len(self.itos):
                raise ValueError(f"token id {i} out of range")
            if skip_special and i < len(SPECIALS):
                continue
            words.append(self.itos[i])
        return _join(words)

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @classmethod
    def from_dict(cls, data: dict) -> "WordTokenizer":
        return cls(data["itos"])

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "WordTokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _join(words: list[str]) -> str:
    """Join tokens with spaces, attaching punctuation to the neighbouring word."""
    out = ""
    for w in words:
        if not out or w in _NO_SPACE_BEFORE or out[-1] in _NO_SPACE_AFTER:
            out += w
        else:
            out += " " + w
    return out
