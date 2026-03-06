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
from sklearn.metrics import pairwise_distances

import anndata as ad
import squidpy as sq

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("Test operations")

def get_spatial_info(
    counts_mat: ad.AnnData,
    ncores: int = 1,
    kind: str = "gaussian",
    distance_key: str = "weighted_adjacency",
    connectivity_key: str = "spatial_connectivities"
    ) -> ad.AnnData:
    """
    Compute spatial neighbors, Euclidean distances, and a weighted adjacency matrix.

    Parameters
    ----------
    counts_mat : AnnData
        Spatial count matrix. Modified in place by adding entries to obsp.
    ncores : int, optional
        Number of CPU cores for pairwise distance computation, by default 1.
    kind : str, optional
        Weighting scheme passed to build_distance_weights, by default gaussian.

    Returns
    -------
    AnnData
        The input object with euclidean_distances and weighted_adj stored in obsp.
    """
    if counts_mat.is_view:
        counts_mat = counts_mat.copy()

    # Ensure spatial_connectivities exists
    if connectivity_key not in counts_mat.obsp:
        sq.gr.spatial_neighbors(counts_mat)
    if connectivity_key not in counts_mat.obsp:
        raise KeyError(
            f"{connectivity_key} is not found in counts_mat.obsp "
            f"after calling sq.gr.spatial_neighbors.\nCurrent keys are: "
            f"{counts_mat.obsp.keys()}"
        )

    if sp.issparse(counts_mat.X):
        gene_sums = np.asarray(counts_mat.X.sum(axis=0)).ravel()
    else:
        gene_sums = counts_mat.X.sum(axis=0)

    keep_vars = counts_mat.var_names[gene_sums > 0]
    dist_test = pairwise_distances(counts_mat[:, keep_vars].X, n_jobs=ncores)
    counts_mat.obsp[distance_key] = sp.csr_matrix(dist_test)

    W = build_distance_weights(
        counts_mat.obsp[connectivity_key],
        counts_mat.obsp[distance_key],
        kind=kind,
    )
    counts_mat.obsp[distance_key] = W
    dist_nans = int(np.isnan(dist_test).sum())
    w_nans = int(np.isnan(W.data).sum()) if W.nnz else 0
    log.info(
        "[sanity] get_spatial_info: dist_nans=%s weight_nans=%s obsp_keys=%s",
        dist_nans,
        w_nans,
        list(counts_mat.obsp.keys()),
    )

    return counts_mat


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

    # Build W with same sparsity as D (and A)
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
    matrix from .obsp[connectivity_key], symmetrizes it, and
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
    steps: int = 5,
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
        Sparse adjacency (weights allowed).
    alpha : float, default 0.7
        Mixing parameter between propagated features and the original signal.
    steps : int, default 8
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
    lazy: float = 0.0,
    steps: int | None = 4,
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
    lazy : float, default 0.0
        Laziness parameter for boundary preservation. P_lazy = (1 - lazy) P + lazy I
        reduces “leakage” across weak boundaries and stabilizes the iteration.
        Not required in regular lattice.
    steps : int or None, default 4
        If provided, run exactly this many iterations (power method).
        If None, return X unaltered.

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

    # Row-normalize W
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    with np.errstate(divide='ignore'):
        inv_row = 1.0 / np.maximum(row_sum, eps)
    P = sp.diags(inv_row) @ W

    # Lazy walk (attempt to boundary preservation)
    # Not required in regular lattice
    if lazy > 0.0:
        I = sp.eye(n, format="csr")
        P = (1.0 - lazy) * P + lazy * I

    # Iterative personalized PageRank with teleportation rate to X
    Z = X.copy()
    walk = float(alpha)
    tele = 1.0 - walk

    if steps is not None:
        for _ in range(int(steps)):
            Z = walk * (P @ Z) + tele * X
        return Z
    
    else:
        msg = f"Spatial algorithm have not been applied because steps value was: {steps}\n"
        log.warning(msg)

    return Z



def neighbors_average(
    df: pd.DataFrame,
    adata: ad.AnnData,
    columns: Optional[List[str]] = None,
    by: Optional[List[str]] = None,
    method: Literal["degree", "weighted", "diffuse", "cpr"] = "cpr",
    connectivity_key: str = "spatial_connectivities",
    distance_key: str = "weighted_adjacency",
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
                  "steps": int} 

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
    ...     by=["sample"], method="cpr"
    ... )
    """
    
    if method_kwargs is None:
        method_kwargs = {}
    log.info(
        "[sanity] neighbors_average: method=%s columns=%s by=%s",
        method,
        columns,
        by,
    )

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
            Z = _random_walk_diffuse(X, A, **method_kwargs)

        elif m == "cpr":
            Z = _pagerank_diffuse(X, A, **method_kwargs)

        else:
            msg = (f'Unknown method: {method}. Accepted methods are:\n'
                   f'"degree", "weighted", "diffuse", "cpr"')
            raise ValueError(msg)

        collector.append(pd.DataFrame(Z, columns=columns, index=group.index))

    if {'cell','CHROM','seg','cnv_state'}.issubset(df.columns):
        return pd.concat((df.loc[:, ['cell','CHROM','seg','cnv_state']],
                          pd.concat(collector)), axis=1)
    else:
        return pd.concat(collector)
    
    
    
    
    
