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