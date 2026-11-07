import json
import math
import time

import pytest
import torch

from myllm.generate import load_checkpoint
from myllm.train import PRESETS, TrainConfig, history_path_for, lr_at, train


def test_lr_schedule_warmup_then_decay():
    cfg = TrainConfig(lr=1.0, warmup_steps=10, min_lr_ratio=0.1)
    assert lr_at(0.0, 0, cfg) == pytest.approx(0.1)
    assert lr_at(0.0, 9, cfg) == pytest.approx(1.0)
    assert lr_at(0.0, 10, cfg) == pytest.approx(1.0)
    assert lr_at(0.5, 50, cfg) < lr_at(0.1, 50, cfg)
    assert lr_at(1.0, 50, cfg) == pytest.approx(0.1)
    assert lr_at(5.0, 50, cfg) == pytest.approx(0.1)  # clamped


def test_presets_are_valid():
    for kwargs, defaults, minutes in PRESETS.values():
        assert kwargs["n_embd"] % kwargs["n_head"] == 0 and minutes > 0 and defaults["vocab_size"] > 4
    assert PRESETS["default"][2] == 10.0  # the "10 minute LLM"


def test_training_reduces_loss_and_writes_artifacts(trained):
    d, res = trained
    first = res.history[0]["val_loss"]
    assert res.steps == 150
    assert res.val_loss < first - 1.0, (first, res.val_loss)
    assert res.path.is_file() and history_path_for(res.path).is_file()
    assert res.n_params > 0


def test_history_records_times(trained):
    _, res = trained
    log = json.loads(history_path_for(res.path).read_text())
    secs = [r["seconds"] for r in log["history"]]
    assert secs == sorted(secs) and len(secs) >= 3
    t = log["timings"]
    assert set(t) == {"load_corpus_s", "tokenize_s", "train_s", "tokens_per_s"}
    assert t["train_s"] > 0 and t["tokens_per_s"] > 0
    assert log["history"][0]["train_loss"] is None  # nothing trained at step 0


def test_checkpoint_roundtrip_is_self_contained(trained):
    _, res = trained
    model, tok, info = load_checkpoint(res.path)
    assert model.cfg.vocab_size == tok.vocab_size
    assert info["steps"] > 0 and not model.training


def test_same_seed_reproduces(tmp_path, corpus_text):
    (tmp_path / "c.txt").write_text(corpus_text, encoding="utf-8")
    cfg = lambda: TrainConfig(preset="tiny", max_steps=20, minutes=5, eval_every=10, eval_batches=2, log=False)
    a = train(tmp_path / "c.txt", tmp_path / "a.pt", cfg())
    b = train(tmp_path / "c.txt", tmp_path / "b.pt", cfg())
    assert a.val_loss == pytest.approx(b.val_loss)


def test_time_budget_is_respected(tmp_path, corpus_text):
    (tmp_path / "c.txt").write_text(corpus_text, encoding="utf-8")
    t0 = time.time()
    res = train(tmp_path / "c.txt", tmp_path / "m.pt", TrainConfig(preset="tiny", minutes=0.05, log=False))
    assert time.time() - t0 < 3 + 20  # 3s budget + setup/eval slack
    assert res.seconds >= 3 and res.steps > 0


def test_errors(tmp_path):
    (tmp_path / "s.txt").write_text("too small", encoding="utf-8")
    with pytest.raises(ValueError, match="too small"):
        train(tmp_path / "s.txt", tmp_path / "m.pt", TrainConfig(preset="tiny", max_steps=1, log=False))
    with pytest.raises(ValueError, match="unknown preset"):
        train(tmp_path, tmp_path / "m.pt", TrainConfig(preset="nope", log=False))
    with pytest.raises(FileNotFoundError):
        train(tmp_path / "missing", tmp_path / "m.pt", TrainConfig(log=False))


def test_no_partial_checkpoint_left(trained):
    d, _ = trained
    assert not list(d.glob("*.tmp"))
