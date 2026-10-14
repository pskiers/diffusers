# MiniT2I model code (vendored, unmodified)

`mmdit.py` is a verbatim copy of the model definition shipped with the
[MiniT2I/MiniT2I](https://huggingface.co/MiniT2I/MiniT2I) checkpoints, at Hub
revision `5b4da2870d1554992670d1005567e9d55b1b37bb` (MIT License, see `LICENSE`, from
[minit2i-torch](https://github.com/Hope7Happiness/minit2i-torch)).

`models/minit2i.py` builds the frozen MiniT2I-B/16 painter from these classes and
the released weights. It swaps in fused attention and exposes the blocks in
two halves so a thinker can read early hidden states and steer later ones;
`tests/test_minit2i_thinker.py` checks it against `MMJiT.forward` here.
