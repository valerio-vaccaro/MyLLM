import pytest
import torch

from myllm.model import GPT, GPTConfig


@pytest.fixture()
def model():
    torch.manual_seed(0)
    return GPT(GPTConfig(vocab_size=50, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0))


def test_output_shapes_and_loss(model):
    x = torch.randint(0, 50, (3, 10))
    logits, loss = model(x, x)
    assert logits.shape == (3, 10, 50)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_initial_loss_near_uniform(model):
    x = torch.randint(0, 50, (8, 16))
    _, loss = model(x, x)
    assert abs(loss.item() - torch.log(torch.tensor(50.0)).item()) < 0.5


def test_is_causal(model):
    model.eval()
    a = torch.randint(0, 50, (1, 12))
    b = a.clone()
    b[0, 8:] = (b[0, 8:] + 1) % 50  # change only the future
    la, lb = model(a)[0], model(b)[0]
    assert torch.allclose(la[:, :8], lb[:, :8], atol=1e-5)
    assert not torch.allclose(la[:, 8:], lb[:, 8:])


def test_rejects_too_long_sequence(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, 17, dtype=torch.long))


def test_bad_head_config():
    with pytest.raises(ValueError):
        GPTConfig(n_embd=30, n_head=4)


def test_weights_tied_and_counted_once(model):
    assert model.head.weight is model.tok_emb.weight
    assert model.num_params() == sum(p.numel() for p in set(model.parameters()))


def test_can_overfit_one_batch(model):
    torch.manual_seed(1)
    x = torch.randint(0, 50, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model(x, x)[1].item()
    for _ in range(60):
        loss = model(x, x)[1]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < first * 0.5
