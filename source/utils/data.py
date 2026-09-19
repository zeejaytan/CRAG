"""Dataloader worker-seeding helpers (missing from the code release).

`source/datasets/crag.py` imports `make_seed`, `pytorch_worker_seed` and
`worker_init_fn` from this module. Semantics follow the call sites: the
iterable dataset builds a worker-aware RNG from several entropy sources, and
the module is passed as DataLoader `worker_init_fn`.
"""

import hashlib
import random

import numpy as np
import torch


def pytorch_worker_seed() -> int:
    """32-bit seed unique to this DataLoader worker (main process included)."""
    base = torch.initial_seed() % 2**32
    info = torch.utils.data.get_worker_info()
    if info is None:
        return base
    return (base + info.id) % 2**32


def make_seed(*args) -> int:
    """Fold arbitrary entropy (ints, pids, timestamps, bytes) into one seed."""
    h = hashlib.sha256()
    for a in args:
        h.update(a if isinstance(a, (bytes, bytearray)) else str(a).encode())
    return int.from_bytes(h.digest()[:4], "little")


def worker_init_fn(worker_id: int) -> None:
    """Seed numpy/random per worker. Standard DataLoader `worker_init_fn`."""
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
