import pytest
import torch

from myllm.data import clean_text, get_batch, load_corpus, split_ids, strip_gutenberg


def test_strip_gutenberg():
    raw = "license\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nreal text\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nfooter"
    assert strip_gutenberg(raw).strip() == "real text"
    assert strip_gutenberg("plain") == "plain"


def test_clean_text_removes_urls_and_emails():
    assert "http" not in clean_text("see https://x.org/a now") and "@" not in clean_text("a b@c.com d")


def test_load_corpus_dir_and_file(tmp_path):
    (tmp_path / "b.txt").write_text("two", encoding="utf-8")
    (tmp_path / "a.txt").write_text("one", encoding="utf-8")
    (tmp_path / "skip.md").write_text("nope", encoding="utf-8")
    assert load_corpus(tmp_path) == "one\n\ntwo"
    assert load_corpus(tmp_path / "a.txt") == "one"


def test_load_corpus_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="download_full_texts"):
        load_corpus(tmp_path)


def test_split_is_disjoint_complete_and_spread():
    ids = list(range(256 * 100))
    train, val = split_ids(ids)
    assert len(train) + len(val) == len(ids)
    assert set(train.tolist()).isdisjoint(val.tolist())
    assert len(val) == 5 * 256
    assert val.min() < len(ids) // 4 and val.max() > len(ids) * 3 // 4  # spread across corpus


def test_get_batch_shift_and_bounds():
    data = torch.arange(100)
    x, y = get_batch(data, 8, 10, torch.Generator().manual_seed(0))
    assert x.shape == y.shape == (8, 10)
    assert torch.equal(y, x + 1)
    assert y.max() <= 99


def test_get_batch_deterministic_and_validates():
    data = torch.arange(100)
    a = get_batch(data, 4, 5, torch.Generator().manual_seed(3))[0]
    b = get_batch(data, 4, 5, torch.Generator().manual_seed(3))[0]
    assert torch.equal(a, b)
    with pytest.raises(ValueError):
        get_batch(torch.arange(5), 2, 5)
