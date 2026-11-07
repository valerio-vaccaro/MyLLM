import math

import pytest
import torch

from myllm.evaluate import evaluate_heldout, evaluate_ids, evaluate_text
from myllm.generate import filter_logits, generate, load_checkpoint
from myllm.model import GPT, GPTConfig


@pytest.fixture(scope="module")
def loaded(trained):
    return load_checkpoint(trained[1].path)


def test_filter_top_k():
    out = filter_logits(torch.tensor([1.0, 4.0, 3.0, 2.0]), top_k=2)
    assert torch.isinf(out[[0, 3]]).all() and torch.isfinite(out[[1, 2]]).all()


def test_filter_top_p_keeps_minimal_nucleus_and_best():
    logits = torch.log(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = filter_logits(logits, top_p=0.7)
    assert torch.isfinite(out[:2]).all() and torch.isinf(out[2:]).all()
    assert torch.isfinite(filter_logits(logits, top_p=1e-6)[0])  # never empty
    assert torch.equal(filter_logits(logits, None, None), logits)


def test_filter_does_not_mutate_input():
    logits = torch.tensor([1.0, 2.0, 3.0])
    filter_logits(logits, top_k=1)
    assert torch.isfinite(logits).all()


def test_generate_is_seed_deterministic(loaded):
    model, tok, _ = loaded
    kw = dict(temperature=3.0, top_k=None, top_p=None)  # flat distribution: seeds must matter
    a = generate(model, tok, "the king", 30, seed=7, **kw)
    assert a == generate(model, tok, "the king", 30, seed=7, **kw)
    assert a != generate(model, tok, "the king", 30, seed=8, **kw)


def test_generate_length_and_prompt(loaded):
    model, tok, _ = loaded
    out = generate(model, tok, "the king said", 12, seed=0)
    assert out.startswith("the king said")
    assert len(tok.encode(out)) == 3 + 12


def test_greedy_equals_top_k_1_and_zero_new_tokens(loaded):
    model, tok, _ = loaded
    assert generate(model, tok, "and god", 10, temperature=0) == generate(
        model, tok, "and god", 10, temperature=1.0, top_k=1, top_p=None, seed=1)
    assert generate(model, tok, "and god", 0) == "and god"


def test_generate_handles_long_prompt_and_empty_prompt(loaded):
    model, tok, _ = loaded
    long_prompt = "the king said unto the people " * 20  # longer than block_size
    assert generate(model, tok, long_prompt, 3, seed=0)
    assert generate(model, tok, "", 5, seed=0)


def test_generate_never_emits_special_tokens(loaded):
    model, tok, _ = loaded
    out = generate(model, tok, "", 40, temperature=1.5, top_k=None, top_p=None, seed=3)
    assert "<" not in out


def test_negative_temperature_rejected(loaded):
    with pytest.raises(ValueError):
        generate(*loaded[:2], "x", 1, temperature=-1)


def test_untrained_model_perplexity_near_vocab_size():
    torch.manual_seed(0)
    m = GPT(GPTConfig(vocab_size=100, block_size=16, n_layer=1, n_head=2, n_embd=32, dropout=0.0))
    r = evaluate_ids(m, torch.randint(0, 100, (1000,)))
    assert 70 < r.perplexity < 140
    assert r.tokens == (999 // 16) * 16
    assert r.loss == pytest.approx(math.log(r.perplexity))


def test_trained_model_beats_untrained(trained, loaded, corpus_text):
    model, tok, _ = loaded
    r = evaluate_text(model, tok, corpus_text, max_tokens=2000)
    assert r.perplexity < tok.vocab_size / 5
    assert r.accuracy > 0.2


def test_heldout_eval_and_too_short_input(trained, loaded):
    model, tok, _ = loaded
    r = evaluate_heldout(model, tok, trained[0] / "c.txt")
    assert r.tokens > 0 and r.perplexity > 1
    with pytest.raises(ValueError):
        evaluate_ids(model, [1, 2, 3])
