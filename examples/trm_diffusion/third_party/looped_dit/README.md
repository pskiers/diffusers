# Looped-DiT data code (vendored, unmodified)

Verbatim copies of the data-preparation and data-loading files from
[OpenSenseNova/Looped-DiT](https://github.com/OpenSenseNova/Looped-DiT) at commit
`65a7705ad2954fa127721bd1131ea8f865688848` (MIT License, see `LICENSE`):

- `tools/prepare_finetune_data.py`: builds the Std-120K fine-tuning shards
  (BLIP3o-60K, DALL-E 3, ShareGPT-4o-Image text-to-image).
- `looped_dit/data.py` (+ its `config.py` / `utils.py` imports): the
  WebDataset loader that decodes, resizes, center-crops and tokenizes them.
- `looped_dit/diffusion.py`: the flow-matching training loss and Euler sampler
  (reference for `models/minit2i.py`, checked in `tests/test_minit2i_thinker.py`).

Only the two empty `__init__.py` files at this level and in `tools/` were added
(to make the files importable). `datasets/std120k_dataset.py` reproduces this
pipeline in a space-efficient packed format, and
`tests/test_std120k_dataset.py` checks it against these files sample for sample.
