#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  6 14:39:54 2025

@author: carlino.calogero
"""

from typing import Optional, Literal, Tuple, Sequence, List, Dict, Any
import numpy as np
import scipy.sparse as sp
import pandas as pd
import anndata as ad


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

    The function takes a sparse adjacency matrix A (symmetric) and a sparse matrix D holding
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
        Symmetric CSR matrix W (n x n) with the same sparsity pattern as D and A, containing
        nonnegative edge weights derived from distances.

    """    
    # A: sparse adjacency.
    # D: sparse distances on the same edges.
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

    # optional cutoff
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


#  utilities 
def _get_graph(
    adata: ad.AnnData,
    cells: Sequence[str],
    connectivity_key: str,
    distance_key: str,
    ) -> Tuple[sp.csr_matrix, Optional[sp.csr_matrix]]:
    """
    Extract a subgraph for a given set of cells from an AnnData object.

    The function subsets adata to the provided cells, fetches the connectivity
    matrix from .obsp[connectivity_key], symmetrizes it defensively, and
    optionally fetches the distance matrix from .obsp[distance_key] if present.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing the full graph in .obsp.
    cells : sequence of str
        Cell identifiers (must be present in adata.obs_names) defining the subgraph.
    connectivity_key : str
        Key in adata.obsp holding the sparse connectivity matrix.
    distance_key : str
        Key in adata.obsp holding the sparse distance matrix. If the key is not
        present, the returned distance matrix is None.

    Returns
    -------
    (csr_matrix, Optional[csr_matrix])
        Tuple of (A, D) where A is the symmetrized connectivity submatrix in CSR
        format and D is the distance submatrix in CSR format or None if missing.
    """
    view = adata[cells, :]
    A = view.obsp[connectivity_key].tocsr()
    # symmetrize
    if (A - A.T).nnz:
        A = (A + A.T) * 0.5
    D = view.obsp[distance_key].tocsr() if distance_key in view.obsp else None
    return A, D


def _random_walk_diffuse(
    X: np.ndarray,
    A: sp.spmatrix,
    alpha: float = 0.7,
    steps: int = 15,
    ) -> np.ndarray:
    """
    Diffuse features over a graph via an iterated random-walk update.

    The update is Z <- alpha * (D^{-1} A Z) + (1 - alpha) * X for the given
    number of steps, where D is the degree matrix of A. This acts as a simple
    low-pass smoother on graph signals.

    Parameters
    ----------
    X : ndarray, shape (n_nodes, n_features)
        Input features to diffuse.
    A : sparse matrix, shape (n_nodes, n_nodes)
        Sparse adjacency (weights allowed). Need not be row-stochastic.
    alpha : float, default 0.7
        Mixing parameter between propagated features and the original signal.
    steps : int, default 15
        Number of diffusion iterations.

    Returns
    -------
    ndarray, shape (n_nodes, n_features)
        Diffused features after the specified number of steps.
    """
    deg = np.asarray(A.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        dinv = 1.0 / np.maximum(deg, 1e-12)
    RW = sp.diags(dinv) @ A  # inverse distance * Adj
    Z = X.copy()
    for _ in range(steps):
        Z = alpha * (RW @ Z) + (1 - alpha) * X
    return Z


def _pagerank_diffuse(
    X: np.ndarray,
    A: sp.spmatrix,
    alpha: float = 0.75,
    *,
    coifman_alpha: float = 0.5,
    lazy: float = 0.1,
    steps: int | None = None,
    tol: float = 1e-6,
    max_iter: int = 30,
    ) -> np.ndarray:
    """
    Personalized PageRank diffusion with Coifman density correction.
    
    Reference:
    1) R.R. Coifman, S. Lafon, A.B. Lee, M. Maggioni, B. Nadler, F. Warner, & S.W. Zucker, 
    Geometric diffusions as a tool for harmonic analysis and structure definition of data: Diffusion maps.
    Proc. Natl. Acad. Sci. U.S.A. 102 (21) 7426-7431, https://doi.org/10.1073/pnas.0500334102 (2005)
    
    2) Page, L., Brin, S., Motwani, R. & Winograd, T. (1998).
    The PageRank Citation Ranking: Bringing Order to the Web. 
    Stanford Digital Library Technologies Project 

    Parameters
    ----------
    X : (n, d) array
        Input features/signals for n nodes and d channels.
        Teleportation pushes the diffusion back to X at every step.
    A : sparse matrix (n, n)
        Symmetric, nonnegative affinity/adjacency (e.g., kNN with weights).
    alpha : float, default 0.75
        PageRank continuation (walk) probability. (1 - alpha) is the teleport
        probability to X. Larger alpha -> stronger smoothing; smaller -> sharp boundaries.
    coifman_alpha : float, default 0.5
        Coifman density correction exponent. 0 -> random-walk on raw graph;
        0.5 -> normalized Laplacian geometry; 1 -> strong de-biasing against density.
    lazy : float, default 0.1
        Laziness parameter for boundary preservation. P_lazy = (1 - lazy) P + lazy I
        reduces “leakage” across weak boundaries and stabilizes the iteration.
    steps : int or None, default None
        If provided, run exactly this many iterations (power method). If None,
        iterate until ||Z_{t+1}-Z_t||_F / ||Z_t||_F < tol or max_iter reached.
    tol : float, default 1e-6
        Relative tolerance for early stopping when steps is None.
    max_iter : int, default 30
        Safety cap on iterations when steps is None.

    Returns
    -------
    Z : (n, d) array
        Diffused signals. Same shape as X.
    """

    n = A.shape[0]
    X = np.asarray(X)
    A = A.tocsr()

    # Coifman diffusion: W = D^{-cf} A D^{-cf}
    deg = np.asarray(A.sum(axis=1)).ravel()
    eps = 1e-12
    d_pow = np.power(np.maximum(deg, eps), -coifman_alpha)
    Dcf_inv = sp.diags(d_pow)
    W = Dcf_inv @ A @ Dcf_inv

    # Row-normalize to Markov matrix P
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        inv_row = 1.0 / np.maximum(row_sum, eps)
    P = sp.diags(inv_row) @ W

    # Lazy walk for boundary preservation
    if lazy > 0.0:
        I = sp.eye(n, format="csr")
        P = (1.0 - lazy) * P + lazy * I

    # Iterative personalized PageRank with teleportation to X
    Z = X.copy()
    walk = float(alpha)
    tele = 1.0 - walk

    if steps is not None:
        for _ in range(int(steps)):
            Z = walk * (P @ Z) + tele * X
        return Z

    # Converge by tol
    denom_floor = np.linalg.norm(Z, ord="fro")
    denom_floor = denom_floor if denom_floor > 0 else 1.0
    for _ in range(max_iter):
        Z_next = walk * (P @ Z) + tele * X
        rel = np.linalg.norm(Z_next - Z, ord="fro") / max(np.linalg.norm(Z, ord="fro"), denom_floor)
        Z = Z_next
        if rel < tol:
            break
    return Z #if X.ndim == 2 else Z.ravel()



def neighbors_average(
    df: pd.DataFrame,
    adata: ad.AnnData,
    columns: Optional[List[str]] = None,
    by: Optional[List[str]] = None,
    method: Literal["degree", "weighted", "diffuse", "cpr"] = "cpr",
    connectivity_key: str = "spatial_connectivities",
    distance_key: str = "spatial_distances",
    method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
    """
    Smooth selected columns by averaging over each cell's neighborhood, optionally per group.

    The function groups rows by `by` (if provided), extracts the corresponding subgraph
    for those cells from `adata.obsp`, and applies one of several smoothing schemes to
    each column listed in `columns` independently:

      - "degree": unweighted neighbor mean using the connectivity matrix (adds self-loops).
      - "weighted": inverse-distance weighted mean using the distance matrix (adds self-loops).
      - "diffuse": iterative random-walk diffusion (uses _random_walk_diffuse).
      - "cpr": personalized PageRank–style diffusion (uses _pagerank_diffuse).

    The output preserves row order within each group and aligns to the input index.
    If the input contains {cell, CHROM, seg, cnv_state}, those columns are prepended
    to the returned DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input table. Must contain a column "cell" (string-like IDs present in adata.obs_names),
        and all columns listed in `columns`. If grouping is used, must also contain all `by` keys.
    adata : anndata.AnnData
        AnnData object providing the global graphs in .obsp[connectivity_key]
        and optionally .obsp[distance_key]. Must include all cells referenced in df["cell"].
    columns : list of str, optional
        Names of numeric columns in `df` to smooth. Required (must not be None or empty).
    by : list of str, optional
        Grouping keys. If None, all rows are smoothed as a single group.
    method : {"degree", "weighted", "diffuse", "cpr"}, default "cpr"
        Smoothing strategy (see above).
    connectivity_key : str, default "spatial_connectivities"
        Key in adata.obsp containing the sparse connectivity matrix.
    distance_key : str, default "spatial_distances"
        Key in adata.obsp containing the sparse distance matrix. Required when method="weighted".
    method_kwargs : dict, optional
        Extra method-specific parameters. Examples:
          • diffuse: {"alpha": float, 
                      "steps": int}
          • cpr: {"alpha": float, 
                  "coifman_alpha":float, 
                  "lazy":float,
                  "steps": int, 
                  "tol":float, 
                  "max_iter":int} 

    Returns
    -------
    pandas.DataFrame
        Smoothed values with the same number of rows as `df` (per group), indexed as `df`.
        If {cell, CHROM, seg, cnv_state} are present in `df`, they are concatenated in front
        of the smoothed columns; otherwise only smoothed columns are returned.

    Raises
    ------
    ValueError
        If `method` is unknown, or `method=="weighted"` and the distance matrix is unavailable.

    Notes
    -----
    - Self-loops are added for "degree" and "weighted" to include each cell's own value.
    - Denominators of zero are set to 1.0 to avoid division by zero.

    Examples
    --------
    >>> neighbors_average(
    ...     df=table, adata=adata, columns=["score1", "score2"],
    ...     by=["sample"], method="degree"
    ... )
    """
    
    if method_kwargs is None: method_kwargs = {}

    collector = []

    # grouping
    group_iter = df.groupby(by, observed=True, sort=False)[['cell'] + by + columns]
    for _, group in group_iter:
        cells = group['cell'].to_numpy()
        A, D = _get_graph(adata, cells, connectivity_key, distance_key)
        X = group.loc[:, columns].to_numpy(dtype=float)

        # compute smoothing
        m = method.lower()
        if m == "weighted":
            # Inverse-distance average with self loops
            A_sl = (A + sp.eye(A.shape[0], format='csr'))
            if D is None:
                raise ValueError("distance_key is required for 'weighted'.")
            D1 = D.copy()
            D1.setdiag(1.0)  # avoid division by 0
            Dinv = D1.copy()
            Dinv.data = 1.0 / np.maximum(D1.data, 1e-12)
            W = A_sl.multiply(Dinv).tocsr()
            num = W @ X
            den = np.asarray(W.sum(axis=1)).ravel()[:, None]
            den[den == 0] = 1.0
            Z = num / den

        elif m == "degree":
            A_sl = (A + sp.eye(A.shape[0], format='csr'))
            num = A_sl @ X
            den = np.asarray(A_sl.sum(axis=1)).ravel()[:, None]
            den[den == 0] = 1.0
            Z = num / den

        elif m == "diffuse":
            Z = _random_walk_diffuse(X, A, **{"alpha": 0.75, "steps": 15, **method_kwargs})

        elif m == "cpr":
            Z = _pagerank_diffuse(X, A, **method_kwargs)

        else:
            raise ValueError(f"Unknown method: {method}")

        collector.append(pd.DataFrame(Z, columns=columns, index=group.index))

    if {'cell','CHROM','seg','cnv_state'}.issubset(df.columns):
        return pd.concat((df.loc[:, ['cell','CHROM','seg','cnv_state']],
                          pd.concat(collector)), axis=1)
    else:
        return pd.concat(collector)
