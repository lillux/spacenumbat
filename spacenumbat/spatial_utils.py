#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:39:54 2025

@author: carlino.calogero
"""

from typing import Optional, Literal
import numpy as np
import scipy.sparse as sp


def build_distance_weights(
    A: sp.spmatrix,
    D: sp.spmatrix,
    kind: Literal["gaussian", "exp", "invdist", "cauchy"] = "gaussian",
    sigma: Optional[float] = None,
    ell: Optional[float] = None,
    p: float = 1.0,
    radius: Optional[float] = None,
    include_A_weight: bool = True,
    ) -> sp.csr_matrix:
    """
    Construct an edge-weight matrix from pairwise distances on the same sparsity pattern as an adjacency.

    The function takes a sparse adjacency matrix A (assumed symmetric) and a sparse matrix D holding
    distances for the same edges, aligns D to A, applies a decay kernel to convert distances into
    similarity weights, optionally applies a hard cutoff radius, optionally multiplies by A's existing
    weights, and returns a symmetric CSR matrix W.

    Parameters
    ----------
    A : scipy.sparse.spmatrix
        Sparse adjacency on a graph (n x n), ideally symmetric. Nonzeros indicate edges; values
        are preserved if include_A_weight is True.
    D : scipy.sparse.spmatrix
        Sparse distances (n x n) on the same edge set as A. Only distances on edges are used; others
        are zeroed out.
    kind : {"gaussian", "exp", "invdist", "cauchy"}, default "gaussian"
        Decay kernel mapping distance d to weight w:
          - "gaussian": exp(-(d^2) / sigma^2)
          - "exp":      exp(-d / ell)
          - "invdist":  1 / (d + 1e-6)^p
          - "cauchy":   1 / (1 + (d / sigma)^2)
    sigma : float, optional
        Scale parameter for gaussian/cauchy kernels. If None, defaults to median of positive distances
        in D (fallback 1.0 if no positives).
    ell : float, optional
        Scale parameter for exponential kernel. If None, defaults to median of positive distances
        in D (fallback 1.0 if no positives).
    p : float, default 1.0
        Exponent for the inverse-distance kernel (only used when kind="invdist").
    radius : float, optional
        If provided, sets weights to zero for edges with distance > radius (hard cutoff).
    include_A_weight : bool, default True
        If True, multiply the computed weights elementwise by A, preserving any edge weighting in A.
        If False, use only the kernel-derived weights.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric CSR matrix W (n x n) with the same sparsity pattern as D∩A, containing
        nonnegative edge weights derived from distances.

    Notes
    -----
    - Complexity is O(E), where E is the number of stored distances (nonzeros in D on A's edges).
    - Sigma/ell defaults use the median of positive distances to provide a robust scale.
    - Small epsilons (1e-12, 1e-6) are used for numerical stability and to avoid division by zero.
    - The result is explicitly symmetrized: W := 0.5 * (W + W.T).

    """    
    # A: sparse adjacency (symmetric). D: sparse distances on the same edges.
    # Align distances to edges (zero elsewhere)
    D = D.tocsr().multiply(A > 0)
    d = D.data  # all edge distances
    # Choose decay kernel. Defaults is median neighbor distance
    if kind == "gaussian":
        sigma = float(np.median(d[d>0])) if sigma is None and np.any(d>0) else (sigma or 1.0)
        w = np.exp(-(d**2) / (sigma**2 + 1e-12))
    elif kind == "exp":
        ell = float(np.median(d[d>0])) if ell is None and np.any(d>0) else (ell or 1.0)
        w = np.exp(-d / (ell+1e-12))
    elif kind == "invdist":
        w = 1.0 / np.power(d + 1e-6, p)
    elif kind == "cauchy":
        sigma = float(np.median(d[d>0])) if sigma is None and np.any(d>0) else (sigma or 1.0)
        w = 1.0 / (1.0 + (d/sigma)**2)
    else:
        raise ValueError

    # Hard cutoff optional
    if radius is not None:
        w *= (d <= radius)

    # Build W with same sparsity as D (and A); optionally merge weights with A
    w[np.isnan(w)] = 0
    W = sp.csr_matrix((w, D.indices, D.indptr), shape=D.shape)
    if include_A_weight:
        W = W.multiply(A)

    # Symmetrize
    if (W - W.T).nnz:
        W = (W + W.T) * 0.5
    return W



