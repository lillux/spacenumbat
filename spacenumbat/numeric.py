#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 12:47:06 2026

@author: ccarlino
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit


@njit(cache=True)
def log_sum_exp(vals: np.ndarray) -> float:
    """
    Compute log(sum(exp(vals))) safely in log space.

    Parameters
    ----------
    vals
        One-dimensional numeric NumPy array. Internal callers should pass
        float64 arrays.

    Returns
    -------
    float
        Stable scalar log-sum-exp.

    Notes
    -----
    Special-value behavior:

    - empty input -> -inf
    - any NaN -> NaN
    - any +inf, in the absence of NaN -> +inf
    - all -inf -> -inf

    The explicit loop avoids allocating ``exp(vals - max_val)`` and can be
    called both from Python and from other Numba-compiled functions.
    """
    n = vals.shape[0]

    if n == 0:
        return -np.inf

    max_val = -np.inf
    has_pos_inf = False

    for i in range(n):
        value = vals[i]

        if np.isnan(value):
            return np.nan

        if value == np.inf:
            has_pos_inf = True
        elif value > max_val:
            max_val = value

    if has_pos_inf:
        return np.inf

    # Every element is -inf.
    if max_val == -np.inf:
        return -np.inf

    exp_sum = 0.0
    for i in range(n):
        exp_sum += math.exp(vals[i] - max_val)

    return max_val + math.log(exp_sum)


@njit(cache=True)
def safe_add(left: float, right: float) -> float:
    """
    Add two floating-point values without evaluating inf + (-inf).
    """
    if np.isnan(left) or np.isnan(right):
        return np.nan

    if (
        (left == np.inf and right == -np.inf)
        or (left == -np.inf and right == np.inf)
    ):
        return np.nan

    return left + right


@njit(cache=True)
def safe_subtract(left: float, right: float) -> float:
    """
    Subtract two floating-point values without evaluating inf - inf.
    """
    if np.isnan(left) or np.isnan(right):
        return np.nan

    if (
        (left == np.inf and right == np.inf)
        or (left == -np.inf and right == -np.inf)
    ):
        return np.nan

    return left - right


@njit(cache=True)
def safe_exp_difference(
    numerator_log: float,
    denominator_log: float,
) -> float:
    """
    Compute exp(numerator_log - denominator_log) safely.

    Undefined differences return NaN. Valid finite calculations are unchanged.
    """
    difference = safe_subtract(
        numerator_log,
        denominator_log,
    )

    if np.isnan(difference):
        return np.nan

    if difference == -np.inf:
        return 0.0

    if difference == np.inf:
        return np.inf

    return math.exp(difference)