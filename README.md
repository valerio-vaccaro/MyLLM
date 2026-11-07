# MyLLM — a tiny GPT you can train in 10 minutes on a CPU

A small, readable word-level GPT (decoder-only transformer) in plain PyTorch.
No GPU and no pretrained weights: it trains from scratch on the King James Bible
and *Mastering Bitcoin* (~1.2M tokens) inside a local venv.

```
myllm/
  tokenizer.py   word-level tokenizer (lowercase, punctuation split, <pad>/<unk>/<bos>/<eos>)
  model.py       GPT: pre-LN blocks, causal attention, tied embeddings
  data.py        corpus loading, Gutenberg header stripping, train/val split, batching
  train.py       wall-clock-budgeted training, cosine LR, best-checkpoint saving, timing log
  generate.py    temperature / top-k / top-p sampling; checkpoints are self-contained
  evaluate.py    loss, perplexity, next-token accuracy
  plots.py       charts: training curves, throughput, phase timings, corpus statistics
  cli.py         python -m myllm {train,generate,eval,plot,info}
scripts/download_full_texts.py   fetches the corpus into data/raw
tests/                           pytest suite (~25 s)
```

## Quick start

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python scripts/download_full_texts.py     # once (or drop any .txt files into data/raw)
python -m myllm train                     # ~10 minutes ("default" preset)
python -m myllm generate "In the beginning" --length 60
python -m myllm eval                      # perplexity on held-out text
python -m myllm plot                      # PNG charts -> outputs/
```

| preset    | time budget | params | use it for        |
|-----------|-------------|--------|-------------------|
| `tiny`    | 1 min       | ~0.2M  | smoke test        |
| `fast`    | 3 min       | ~0.6M  | quick experiment  |
| `default` | 10 min      | ~1.1M  | the 10-minute LLM |

`--minutes N` overrides the budget. Training always stops on time and the
learning-rate schedule follows the elapsed fraction of the budget. Throughput
depends on your CPU (about 5k tokens/s for `default` on the dev machine).

## Charts and recorded times

Every `train` writes `models/model.history.json`: train/validation loss, learning
rate, tokens/s and elapsed seconds at each evaluation, plus the time spent loading,
tokenizing and training. `python -m myllm plot` turns it into:

* `outputs/training.png` — loss and perplexity vs. time, LR and throughput, time per phase
* `outputs/corpus.png` — top tokens, Zipf's law, vocabulary coverage

## Tests

```bash
python -m pytest
```

They cover the tokenizer, model (shapes, causality, overfitting one batch), data
split and batching, training (loss decreases, time budget, reproducibility, checkpoint
round trip, timing log), sampling filters, evaluation, charts and the CLI end to end.
They use a tiny synthetic corpus, so no data download is needed.

## Sampling options

| flag            | effect                                                        |
|-----------------|---------------------------------------------------------------|
| `--temperature` | lower = safer/repetitive, higher = more random; `0` = greedy  |
| `--top-k`       | sample only among the k most likely tokens (default 40)       |
| `--top-p`       | sample only from the smallest set with probability mass p (default 0.95) |
| `--seed`        | reproducible output                                           |

## Lessons (notebooks)

`notebooks/` is a course, one lesson per notebook, each with objectives, short explanations,
runnable cells, key takeaways, exercises and discussion questions. Run them in order with
`jupyter lab notebooks/`; the saved outputs let you preview them without running anything.

| # | notebook | topic |
|---|----------|-------|
| 0 | `00_setup` | big picture, environment check, CPU speed benchmark |
| 1 | `01_data_prep` | corpus, cleaning, train/val split, batches |
| 2 | `02_tokenization` | words to ids, vocabulary size trade-off |
| 3 | `03_model_architecture` | embeddings, attention from scratch, causal mask, parameter count |
| 4 | `04_training` | hand-written loop, LR schedule, time-budgeted `train()` (1 min) |
| 5 | `05_inference` | temperature, top-k, top-p, seeds |
| 6 | `06_evaluation` | perplexity, baselines, loss by position |
| 7 | `07_improvement` | controlled experiments under a fixed time budget (~2 min) |
| 8 | `08_ten_minute_llm` | the full recipe and recorded times |
| 9 | `09_toolbox` | CLI reference, checkpoints, troubleshooting |
| 10 | `10_wrapup_and_exercises` | glossary, quiz, project ideas |
