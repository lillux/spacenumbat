#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:08:57 2026

@author: ccarlino
"""

import numpy as np
import scipy.sparse as sp
import pandas as pd

import anndata as ad

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("Test operations")


_HMRF_STATES = ("neu", "loh", "del", "amp", "bamp", "bdel")

_HMRF_SCORE_COLUMNS = (
    "Z_n",
    "Z_loh",
    "Z_del",
    "Z_amp",
    "Z_bamp",
    "Z_bdel",
)

_HMRF_PROB_COLUMNS = (
    "p_neu",
    "p_loh",
    "p_del",
    "p_amp",
    "p_bamp",
    "p_bdel",
)


def _row_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute a numerically stable row-wise softmax."""
    logits = np.asarray(logits, dtype=float)
    logits = np.where(np.isnan(logits), -np.inf, logits)

    row_max = np.max(logits, axis=1, keepdims=True)
    if not np.all(np.isfinite(row_max)):
        bad_rows = np.flatnonzero(~np.isfinite(row_max.ravel()))
        raise ValueError("At least one HMRF node has no finite state score. "
                         f"Invalid row indices: {bad_rows[:10].tolist()}")

    probabilities = np.exp(logits - row_max)
    denominator = probabilities.sum(axis=1, keepdims=True)

    if np.any(denominator <= 0):
        raise ValueError("HMRF softmax produced a zero denominator.")

    return probabilities / denominator


def _mean_field_potts(
    log_scores: np.ndarray,
    adjacency: sp.spmatrix,
    beta: float = 0.25,
    max_iter: int = 15,
    tol: float = 1e-5,
    damping: float = 0.5,
    ) -> tuple[np.ndarray, int, bool]:
    """
    Approximate Potts-HMRF marginal probabilities using mean-field inference.

    Parameters
    ----------
    log_scores
        Local log scores with shape ``(n_nodes, n_states)``.
    adjacency
        Binary or weighted adjacency matrix with shape ``(n_nodes, n_nodes)``.
    beta
        Spatial coupling strength per neighboring edge.
    max_iter
        Maximum number of mean-field updates.
    tol
        Convergence tolerance based on the maximum probability change.
    damping
        Fraction of each new update to accept. Values below 1 help stabilize
        synchronous graph updates.

    Returns
    -------
    probabilities, n_iter, converged
        Approximate state probabilities and convergence information.
    """
    if beta < 0:
        raise ValueError("beta must be non-negative.")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if not 0 < damping <= 1:
        raise ValueError("damping must be in the interval (0, 1].")

    log_scores = np.asarray(log_scores, dtype=float)
    A = adjacency.tocsr().astype(float)

    if A.shape[0] != log_scores.shape[0]:
        raise ValueError("Adjacency and unary score matrices have incompatible shapes: "
                         f"{adjacency.shape} and {log_scores.shape}.")

    D = np.asarray(A.sum(0)).ravel()
    D_inv_sqrt = sp.diags(np.divide(1,
                                    np.sqrt(D),
                                    where=D>0, 
                                    out=np.zeros_like(D)))
    A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    
    probabilities = _row_softmax(log_scores)
    converged = False

    for iteration in range(1, max_iter + 1):
        neighbor_support = A_norm @ probabilities

        proposal = _row_softmax(log_scores + beta * neighbor_support)

        updated = ((1.0 - damping) * probabilities + damping * proposal)

        difference = np.max(np.abs(updated - probabilities))
        probabilities = updated

        if difference < tol:
            converged = True
            break

    return probabilities, iteration, converged


def hmrf_regularize_joint_post(
    joint_post: pd.DataFrame,
    adata: ad.AnnData,
    connectivity_key: str = "spatial_connectivities",
    beta: float = 0.25,
    max_iter: int = 15,
    tol: float = 1e-5,
    damping: float = 0.5,
    ) -> pd.DataFrame:
    """
    Apply an independent Potts HMRF to each genomic segment.

    The canonical posterior columns ``p_*``, ``p_cnv``, ``p_n``,
    ``Z_*``, ``Z``, ``Z_cnv``, ``logBF``, and ``cnv_state_map``
    are replaced by the spatially regularized HMRF values.
    
    Local posterior copies are not retained. Raw expression and allele
    likelihood columns remain available in the joint table.


    Parameters
    ----------
    joint_post
        Per-spot joint posterior table after ``compute_posterior``.
    adata
        AnnData containing the spatial adjacency matrix.
    connectivity_key
        Key in ``adata.obsp`` containing the spatial adjacency.
    beta
        Potts coupling strength per graph edge.
    max_iter
        Maximum number of mean-field iterations per segment.
    tol
        Mean-field convergence tolerance.
    damping
        Update damping in the interval ``(0, 1]``.

    Returns
    -------
    pandas.DataFrame
        Joint posterior table containing local and HMRF probabilities.
    """
    if connectivity_key not in adata.obsp:
        raise KeyError(f"{connectivity_key!r} is not present in adata.obsp. "
                       f"Available keys: {list(adata.obsp.keys())}")

    required_columns = {
        "cell",
        "seg",
        *_HMRF_SCORE_COLUMNS,
        *_HMRF_PROB_COLUMNS,
    }
    missing = required_columns.difference(joint_post.columns)
    if missing:
        raise KeyError("joint_post is missing HMRF input columns: "
                       f"{sorted(missing)}")

    result = joint_post.copy()
    state_array = np.asarray(_HMRF_STATES)
    result["hmrf_iterations"] = 0
    result["hmrf_converged"] = False

    grouped = result.groupby("seg", observed=True, sort=False)

    for segment, group in grouped:
        cells = group["cell"].astype(str).tolist()

        if len(cells) != len(set(cells)):
            raise ValueError(f"Segment {segment!r} contains duplicated cell identifiers.")

        missing_cells = [cell for cell in cells if cell not in adata.obs_names]
        if missing_cells:
            raise KeyError(f"Segment {segment!r} contains cells absent from AnnData: "
                           f"{missing_cells[:10]}")

        # AnnData preserves the requested observation order.
        view = adata[cells, :]

        adjacency = view.obsp[connectivity_key].tocsr()

        # Simple Potts graph: only presence/absence of an edge matters.
        #adjacency = (adjacency > 0).astype(float)
        adjacency.setdiag(0)
        adjacency.eliminate_zeros()

        # Ensure an undirected adjacency.
        adjacency = ((adjacency + adjacency.T) > 0).astype(float).tocsr()

        log_scores = group.loc[:, _HMRF_SCORE_COLUMNS].to_numpy(dtype=float)

        probabilities, n_iter, converged = _mean_field_potts(log_scores=log_scores,
                                                             adjacency=adjacency,
                                                             beta=beta,
                                                             max_iter=max_iter,
                                                             tol=tol,
                                                             damping=damping)

        result.loc[group.index, list(_HMRF_PROB_COLUMNS)] = probabilities
        result.loc[group.index, "hmrf_iterations"] = n_iter
        result.loc[group.index, "hmrf_converged"] = converged

    # CNV probability is the complement of neutral state.
    result["p_cnv"] = 1.0 - result["p_neu"]
    result["p_n"] = result["p_neu"]

    probability_matrix = result.loc[:, _HMRF_PROB_COLUMNS].to_numpy(dtype=float)
    result["cnv_state_map"] = state_array[np.argmax(probability_matrix, axis=1)]

    # HMRF posterior log odds.
    eps = 1e-15
    
    log_probability_matrix = np.log(np.clip(probability_matrix, eps, 1.0))

    result.loc[:, list(_HMRF_SCORE_COLUMNS)] = log_probability_matrix

    result["Z"] = 0.0
    result["Z_cnv"] = np.log(np.clip(result["p_cnv"].to_numpy(dtype=float), eps, 1.0))
    result["Z_n"] = np.log(np.clip(result["p_n"].to_numpy(dtype=float), eps, 1.0,))
    result["Z_neu"] = result["Z_n"]

    result["logBF"] = result["Z_cnv"] - result["Z_n"]
    
    return result