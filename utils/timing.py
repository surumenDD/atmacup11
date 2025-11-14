# utils/timing.py
import math
import os
import sys
import time
from contextlib import contextmanager

import psutil


@contextmanager
def trace(title: str):
    """
    処理時間とメモリ使用量の変化を表示するコンテキストマネージャ。

    例:
        >>> with trace("train_fold_0"):
        ...     train_one_fold(...)
    """
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info().rss / 2.0**30
    yield
    m1 = p.memory_info().rss / 2.0**30
    delta = m1 - m0
    sign = "+" if delta >= 0 else "-"
    delta = math.fabs(delta)
    print(
        f"[{m1:.1f}GB({sign}{delta:.1f}GB):{time.time() - t0:.1f}sec] {title} ",
        file=sys.stderr,
    )


@contextmanager
def timer(name: str):
    """
    処理時間だけを計測したいときに使う簡易版。

    例:
        >>> with timer("load_data"):
        ...     load_data(...)
    """
    t0 = time.time()
    yield
    elapsed_time = time.time() - t0
    print(f"[{name}] done in {elapsed_time:.1f} s")
