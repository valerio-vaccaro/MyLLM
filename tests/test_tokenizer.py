import pytest

from myllm.tokenizer import BOS, EOS, PAD, UNK, WordTokenizer, split_words


def test_split_words_lowercases_and_separates_punctuation():
    assert split_words("Hello, World!") == ["hello", ",", "world", "!"]
    assert split_words("don't stop") == ["don't", "stop"]


def test_special_ids_are_fixed():
    tok = WordTokenizer.train("a b c")
    assert tok.itos[:4] == [PAD, UNK, BOS, EOS]
    assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (0, 1, 2, 3)


def test_vocab_keeps_most_frequent_and_respects_size():
    tok = WordTokenizer.train("a a a b b c", vocab_size=6)
    assert tok.vocab_size == 6
    assert tok.itos[4:] == ["a", "b"]


def test_unknown_words_map_to_unk():
    tok = WordTokenizer.train("a b")
    assert tok.encode("a zzz") == [tok.stoi["a"], tok.unk_id]


def test_roundtrip_and_detokenize_punctuation():
    tok = WordTokenizer.train("the king said, go (now).")
    assert tok.decode(tok.encode("The king said, go (now).")) == "the king said, go (now)."


def test_decode_skips_specials_and_validates_range():
    tok = WordTokenizer.train("a")
    assert tok.decode([tok.bos_id, tok.stoi["a"], tok.pad_id]) == "a"
    with pytest.raises(ValueError):
        tok.decode([999])


def test_save_load(tmp_path):
    tok = WordTokenizer.train("x y z z")
    tok.save(tmp_path / "t.json")
    assert WordTokenizer.load(tmp_path / "t.json").itos == tok.itos


def test_invalid_vocab_rejected():
    with pytest.raises(ValueError):
        WordTokenizer(["a", "b"])
    with pytest.raises(ValueError):
        WordTokenizer.train("a", vocab_size=4)
