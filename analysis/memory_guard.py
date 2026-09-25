"""Choose a batch size that keeps DeepDTA-iBAM inference inside available RAM.

The model itself is small: 41.5 M parameters, about 166 MB. What exhausts memory
is batching, because the protein adapter's self-attention materializes a
(batch x heads x L x L) tensor, where L is the padded length of the longest
target in the batch. That grows with the square of target length, so a single
long protein in a large batch can demand tens of gigabytes:

    64 pairs x 8 heads x 4128 residues^2 x 4 bytes = 34.9 GB

which is what failed on a 64 GB machine. Target length, not library size,
governs memory.

The loader batches by token budget rather than by a fixed batch size, so the two
knobs that matter are `max_pairs_per_batch` and `protein_token_budget`. This
module sets both from the longest sequence actually being scored and the memory
the machine can spare, so a script runs unchanged on a laptop or a large node.

Usage::

    from memory_guard import apply_batch_limit
    info = apply_batch_limit(config, sequences)     # mutates config in place
    print(info["max_pairs_per_batch"], info["reason"])
"""

from __future__ import annotations

import math
import os
from typing import Iterable

# Per-pair attention cost model, in bytes:
#   heads x L^2 x 4 bytes, times a factor covering the several attention and
#   residual tensors alive at once in the adapter and the fusion stack.
ATTENTION_HEADS = 8
BYTES_PER_FLOAT = 4
LIVE_TENSOR_FACTOR = 6.0

# Never hand the model more than this share of free memory, and never drop
# below one pair per batch.
MEMORY_FRACTION = 0.35
MIN_PAIRS = 1
MAX_PAIRS = 64
# Used when the machine's free memory cannot be determined.
FALLBACK_BUDGET_BYTES = 6 * 1024 ** 3


def available_bytes() -> tuple[int, str]:
    """Best estimate of memory we may use, with the source of the estimate."""
    try:
        import psutil  # optional dependency
        return int(psutil.virtual_memory().available), "psutil.available"
    except Exception:
        pass
    try:
        # Windows: ask the OS directly rather than adding a dependency.
        import ctypes

        class _Status(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        st = _Status()
        st.dwLength = ctypes.sizeof(_Status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return int(st.ullAvailPhys), "GlobalMemoryStatusEx"
    except Exception:
        pass
    try:
        page = os.sysconf("SC_PAGE_SIZE")
        avail = os.sysconf("SC_AVPHYS_PAGES")
        return int(page * avail), "sysconf"
    except Exception:
        pass
    return FALLBACK_BUDGET_BYTES, "fallback"


def pairs_that_fit(max_len: int, budget_bytes: int | None = None) -> tuple[int, dict]:
    """Largest pairs-per-batch whose attention tensors fit in the budget."""
    if budget_bytes is None:
        avail, source = available_bytes()
        budget_bytes = int(avail * MEMORY_FRACTION)
    else:
        source = "caller"

    per_pair = (ATTENTION_HEADS * (max_len ** 2) * BYTES_PER_FLOAT
                * LIVE_TENSOR_FACTOR)
    n = int(budget_bytes // max(per_pair, 1))
    n = max(MIN_PAIRS, min(MAX_PAIRS, n))
    return n, {
        "max_protein_length": int(max_len),
        "estimated_bytes_per_pair": int(per_pair),
        "memory_budget_bytes": int(budget_bytes),
        "memory_source": source,
    }


def apply_batch_limit(config, sequences: Iterable[str], *,
                      budget_bytes: int | None = None, verbose: bool = True):
    """Set the loader's batching knobs on `config` so inference fits in RAM."""
    seqs = list(sequences)
    max_len = max((len(s) for s in seqs), default=1) + 8
    n, info = pairs_that_fit(max_len, budget_bytes)

    config.max_pairs_per_batch = n
    config.protein_token_budget = n * max_len
    # The loader also reads these; keep them consistent.
    for attr in ("batch_size", "eval_batch_size"):
        if hasattr(config, attr):
            setattr(config, attr, n)

    info["max_pairs_per_batch"] = n
    info["protein_token_budget"] = n * max_len
    info["reason"] = (
        f"longest target {max_len} residues, "
        f"{info['estimated_bytes_per_pair'] / 1024 ** 3:.2f} GB per pair, "
        f"budget {info['memory_budget_bytes'] / 1024 ** 3:.1f} GB "
        f"({info['memory_source']})"
    )
    if verbose:
        print(f"[memory] {n} pairs per batch: {info['reason']}", flush=True)
    return info
