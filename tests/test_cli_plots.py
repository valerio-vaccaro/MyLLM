import pytest

from myllm.cli import main
from myllm.plots import plot_corpus, plot_training
from myllm.train import history_path_for


def test_plots_written(trained, tmp_path, corpus_dir):
    png = plot_training(history_path_for(trained[1].path), tmp_path / "t.png")
    assert png.stat().st_size > 5_000
    assert plot_corpus(corpus_dir, tmp_path / "c.png", vocab_size=20).stat().st_size > 5_000


def test_cli_end_to_end(tmp_path, corpus_dir, capsys):
    m = str(tmp_path / "m.pt")
    assert main(["train", "--data", str(corpus_dir), "--out", m, "--preset", "tiny", "--max-steps", "30"]) == 0
    assert main(["info", "--model", m]) == 0
    assert main(["generate", "the king", "--model", m, "--length", "5", "--seed", "1"]) == 0
    assert main(["eval", "--model", m, "--data", str(corpus_dir)]) == 0
    assert main(["plot", "--model", m, "--data", str(corpus_dir), "--out-dir", str(tmp_path / "o")]) == 0
    out = capsys.readouterr().out
    assert "perplexity" in out and "params" in out
    assert (tmp_path / "o" / "training.png").is_file() and (tmp_path / "o" / "corpus.png").is_file()


def test_cli_friendly_errors(tmp_path, capsys):
    missing = str(tmp_path / "none.pt")
    for cmd in (["generate", "x"], ["eval"], ["info"], ["plot"]):
        assert main(cmd + ["--model", missing]) == 1
    assert main(["train", "--data", str(tmp_path / "empty"), "--out", missing]) == 1
    assert "error" in capsys.readouterr().err


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        main([])
