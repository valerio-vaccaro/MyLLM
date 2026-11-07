import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
torch.set_num_threads(2)

SENTENCES = [
    "the king said unto the people, go forth and multiply.",
    "bitcoin is a peer-to-peer electronic cash system.",
    "and god said, let there be light: and there was light.",
    "a transaction spends the outputs of earlier transactions.",
]


@pytest.fixture(scope="session")
def corpus_text():
    return "\n".join(SENTENCES * 400)


@pytest.fixture()
def corpus_dir(tmp_path, corpus_text):
    d = tmp_path / "raw"
    d.mkdir()
    (d / "a.txt").write_text(corpus_text, encoding="utf-8")
    return d


@pytest.fixture(scope="session")
def trained(tmp_path_factory, corpus_text):
    """One tiny trained checkpoint shared by the tests that need a real model."""
    from myllm.train import TrainConfig, train

    d = tmp_path_factory.mktemp("trained")
    (d / "c.txt").write_text(corpus_text, encoding="utf-8")
    cfg = TrainConfig(preset="tiny", max_steps=150, minutes=5, eval_every=50, eval_batches=3,
                      warmup_steps=10, lr=3e-3, log=False)
    res = train(d / "c.txt", d / "m.pt", cfg)
    return d, res
