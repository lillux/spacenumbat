#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 13 13:45:52 2025

@author: lillux
"""

import contextlib
import joblib
from tqdm.auto import tqdm as tqdm_auto


@contextlib.contextmanager
def tqdm_joblib(*args, **kwargs):
    """
    Patch joblib to report into a single tqdm progress bar.

    Yields
    ------
    tqdm object
        You can use it as a normal tqdm instance if needed.
    """
    tqdm_object = tqdm_auto(*args, **kwargs)

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *a, **k):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*a, **k)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def ParallelPbar(desc: str | None = None, total: int | None = None, **tqdm_kwargs):
    """
    Return a joblib.Parallel subclass that shows a tqdm progress bar.

    Parameters
    ----------
    desc : str or None
        tqdm description.
    total : int or None
        Total number of tasks. If None, we try len(iterable); if that fails we
        materialize the iterable to a list as a fallback.
    **tqdm_kwargs :
        Any extra tqdm kwargs (leave, position, disable, etc.).
    """
    class Parallel(joblib.Parallel):
        def __call__(self, iterable):
            local_iter = iterable
            local_total = total
            if local_total is None:
                try:
                    local_total = len(iterable)  # works for sequences
                except TypeError:
                    # last-resort: make a list only if needed
                    local_iter = list(iterable)
                    local_total = len(local_iter)

            with tqdm_joblib(total=local_total, desc=desc, **tqdm_kwargs):
                return super().__call__(local_iter)

    return Parallel







