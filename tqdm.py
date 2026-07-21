from __future__ import annotations

def tqdm(iterable=None, *args, **kwargs):
    return iterable if iterable is not None else range(0)

def trange(*args, **kwargs):
    return range(*args)
