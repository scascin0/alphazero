# AlphaZero

This repo aims to provide a working AlphaZero implementation that's simple enough to be able to understand what's going on at a quick glance, without sacrificing too much.

The `connect2.py` example gives a rough idea of what's needed to train an agent. Being this basic, the game does not require everything a complete AlphaZero training loop requires but the main things should be there.

The main reference for this implementation can be found at [this link](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphazero-shedding-new-light-on-chess-shogi-and-go/alphazero_preprint.pdf), which is the preprint of the Nature paper published by DeepMind in 2017.

**Requirements**:
- `torch`
- `matplotlib` (for `connect2.py` example)
- `tqdm` (for progress bar)
