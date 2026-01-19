#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:57:25 2024

@author: lillux
"""
import numpy as np
import pandas as pd
import scipy
import anndata as ad
import natsort

from scipy.cluster.hierarchy import ClusterNode, linkage, fcluster, to_tree
from sklearn.metrics import pairwise_distances

from joblib import Parallel, delayed, cpu_count

from spacenumbat.utils import filter_genes, get_bulk, check_anndata

from functools import partial
from typing import Any, Dict, Union, Optional, List

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("Test clustering")


def scale_counts(x: scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    """
    Row-normalize a sparse count matrix to sum to 1 per row.

    Parameters
    ----------
    x : scipy.sparse.spmatrix
        Sparse count matrix (e.g., from AnnData.X) with cells as rows and genes as columns.

    Returns
    -------
    scipy.sparse.spmatrix
        Sparse matrix with the same shape as `x`, where each row is divided by its row sum.
    """
    return x / x.sum(1)


def choose_ref_cor(
    count_mat: ad.AnnData,
    lambdas_ref: pd.DataFrame,
    gtf: pd.DataFrame
    ) -> pd.Series:
    """
    Assign to each cell the most correlated reference expression profile using annotated genes.

    Parameters
    ----------
    count_mat : anndata.AnnData
        AnnData object containing cell-by-gene count matrix.
    lambdas_ref : pd.DataFrame
        Reference expression profile DataFrame (genes x references).
    gtf : pd.DataFrame
        Genome annotation DataFrame with at least a 'gene' column.

    Returns
    -------
    pd.Series
        Series of best-matching reference names per cell, indexed by cell barcode.

    Raises
    ------
    ValueError
        If `lambdas_ref` contains duplicated columns.
    Warning
        Emits a warning if some cells have zero coverage after filtering.
    
    Notes
    -----
    - If only one reference is present, all cells are assigned to that reference.
    - Log-transformed counts and references are used for correlation calculation.
    - Only genes annotated in all inputs are used for reference assignment.
    - Cells with zero coverage after filtering are excluded from reference assignment.
    """
    if len(lambdas_ref.columns.unique()) != len(lambdas_ref.columns):
        msg = 'Duplicated genes in lambdas_ref'
        raise ValueError(msg)

    if len(lambdas_ref.columns) == 1:
        ref_name = lambdas_ref.columns
        cells = count_mat.obs.index
        best_refs = pd.Series(np.repeat(ref_name, len(cells)), index=cells)
        return best_refs

    genes_annotated = set(gtf.gene).intersection(set(count_mat.var.index)).intersection(set(lambdas_ref.index))
    genes_annotated = [i for i in gtf.gene if i in genes_annotated]

    count_mat = count_mat[:, genes_annotated].copy()
    lambdas_ref_annot = lambdas_ref.loc[genes_annotated, :].copy()

    count_mat.layers['X_norm'] = scale_counts(count_mat.X)
    count_mat.layers['X'] = count_mat.X.copy()
    count_mat.X = count_mat.layers['X_norm']
    count_mat.X = scipy.sparse.csr_matrix(count_mat.X)
    count_mat = count_mat[:, lambdas_ref_annot[np.sum(lambdas_ref_annot * 1e6 > 2, axis=1) > 0].index]

    zero_cov = count_mat.obs.loc[count_mat.X.sum(1).A == 0, :].index

    if len(zero_cov) != 0:
        log.warning(f'Cannot choose reference for {len(zero_cov)} cells due to low coverage')
        count_mat = count_mat[[i for i in count_mat.obs.index if i not in set(zero_cov)], :]

    # homemade vectorized correlation
    c_mat = np.log1p(count_mat.X.toarray() * 1e6)
    ref_mat = np.log1p(lambdas_ref_annot * 1e6).loc[count_mat.var_names, :].values

    c_mat_centered = c_mat - c_mat.mean(axis=1, keepdims=True)
    c_mat_std = c_mat_centered / c_mat.std(axis=1, ddof=0, keepdims=True)

    ref_mat_centered = ref_mat - ref_mat.mean(axis=0)
    ref_mat_std = ref_mat_centered / ref_mat.std(axis=0, ddof=0)

    cor_mat_dot = np.dot(c_mat_std, ref_mat_std) / c_mat.shape[1]
    cors_dot = pd.DataFrame(cor_mat_dot, columns=lambdas_ref.columns, index=count_mat.obs_names)

    best_refs = cors_dot.idxmax(axis=1)

    return best_refs


def get_lambdas_bar(
    lambdas_ref: pd.DataFrame,
    sc_refs: pd.Series,
    verbose: bool = True
    ) -> np.ndarray:
    """
    Compute the weighted average reference expression profile (lambdas_bar)
    by combining reference profiles with proportions estimated from single-cell labels.

    Parameters
    ----------
    lambdas_ref : pd.DataFrame
        Reference expression profiles with genes as index and cell types as columns.
    sc_refs : pd.Series
        Single-cell reference labels indexed by cell barcodes, values are cell type names.
    verbose : bool, optional
        If True, prints fitted proportions for each reference cell type. Default is True.

    Returns
    -------
    np.ndarray
        Weighted average reference expression profile (genes x 1),
        computed as matrix product of lambdas_ref and proportions.
    """
    ref_counts = sc_refs.value_counts(normalize=True)
    lambdas_bar = {}

    for cell_type in lambdas_ref.columns:
        try:
            lambdas_bar[cell_type] = ref_counts[cell_type]
        except KeyError:
            lambdas_bar[cell_type] = 0    

    lambdas_bar = pd.Series(lambdas_bar)
    if verbose:
        log.info(
            "Fitted reference proportions: " +
            ", ".join([f"{name}: {round(val, 3)}" for name, val in lambdas_bar.items()])
        )
    return np.matmul(lambdas_ref, lambdas_bar)


def smooth_expression(
    count_mat: ad.AnnData,
    lambdas_ref: pd.DataFrame,
    gtf: pd.DataFrame,
    window: int = 101,
    cap: int = 3,
    filter_hla: bool = True,
    filter_segments = None,
    ncores: int = 1,
    verbose: bool = True
    ) -> ad.AnnData:
    """
    Smooth gene expression counts using a rolling window after normalization and log-transformation.

    Parameters
    ----------
    count_mat : anndata.AnnData
        Single-cell count matrix with raw counts in `.X`.
    lambdas_ref : pd.DataFrame
        Reference expression profiles (genes x cell types).
    gtf : pd.DataFrame
        Genome annotation with gene information.
    window : int, optional
        Window size for rolling mean smoothing (default 101).
    cap : int, optional
        Cap for expression deviations after normalization (default 3).
    filter_hla : bool, optional
        Whether to exclude genes in the HLA region (default True).

    Returns
    -------
    anndata.AnnData
        Input AnnData with `.layers['X_smooth']` containing the smoothed expression matrix,
        and `.layers['X']` preserving the original counts.
    """
    # Filter mutually expressed genes
    mut_expressed = filter_genes(count_mat, lambdas_ref, gtf, filter_hla=filter_hla, filter_segments=filter_segments)
    count_mat = count_mat[:, mut_expressed].copy()
    lambdas_ref = lambdas_ref.loc[mut_expressed]

    # Normalize counts
    exp_mat = scale_counts(count_mat.X).toarray()

    # Log transform and normalize by reference
    exp_mat_norm = np.log2(exp_mat * 1e6 + 1) - np.log2(lambdas_ref * 1e6 + 1).values

    # Cap expression values
    exp_mat_norm = np.clip(exp_mat_norm, -cap, cap)

    # Center by cell (subtract mean expression)
    row_means = exp_mat_norm.mean(axis=1, keepdims=True)
    exp_mat_norm -= row_means
    exp_mat_norm = pd.DataFrame(exp_mat_norm)

    # Rolling window smoothing along genes (transpose, smooth, transpose back)
    exp_mat_smooth = exp_mat_norm.T.rolling(window=window, center=True, min_periods=1).mean()
    count_mat.layers['X_smooth'] = scipy.sparse.csr_matrix(exp_mat_smooth.values.T)
    count_mat.layers['X'] = count_mat.X

    return count_mat


def exp_hclust(
    count_mat: ad.AnnData,
    lambdas_ref: pd.DataFrame,
    gtf: pd.DataFrame,
    sc_refs: Optional[pd.Series] = None,
    window: int = 101,
    ncores: int = 1,
    verbose: bool = True,
    filter_hla: bool = True,
    filter_segments = None
    ) -> Dict[str, Any]:
    """
    Perform hierarchical clustering on smoothed gene expression profiles.

    This function estimates reference proportions, smooths expression profiles,
    computes pairwise Euclidean distances in parallel, and performs Ward hierarchical clustering.

    Parameters
    ----------
    count_mat : anndata.AnnData
        AnnData containing single-cell barcode x gene ".X" matrix.
    lambdas_ref : pd.DataFrame
        Reference expression profile (genes x cell types).
    gtf : pd.DataFrame
        Genome annotation DataFrame with gene information.
    sc_refs : pd.Series, optional
        Single-cell reference labels indexed by cell barcodes; if None, inferred by `choose_ref_cor`.
    window : int, optional
        Window size for smoothing gene expression (default 101).
    ncores : int, optional
        Number of CPU cores to use for parallel distance computation (default 1).
    verbose : bool, optional
        Whether to print progress messages (default True).
    filter_hla : bool, optional
        Whether to exclude genes in the HLA region during smoothing (default True).

    Returns
    -------
    Dict[str, Any]
        Dictionary containing:
        - 'gexp_roll_wide': smoothed AnnData object with expression profiles.
        - 'hc': hierarchical clustering linkage matrix.
    """
    count_mat = check_anndata(count_mat.copy())
    if sc_refs is None:
        sc_refs = choose_ref_cor(count_mat, lambdas_ref, gtf)
    lambdas_bar = get_lambdas_bar(lambdas_ref, sc_refs, verbose=verbose)
    gexp_roll_wide = smooth_expression(
        count_mat,
        lambdas_bar,
        gtf,
        window=window,
        verbose=verbose,
        filter_hla=filter_hla,
        filter_segments=filter_segments
    )

    # Compute parallel pairwise Euclidean distances
    dist_mat = pairwise_distances(
        gexp_roll_wide.layers['X_smooth'], metric='euclidean', n_jobs=ncores
    )

    # Handle NaNs by replacing with zero
    dist_mat[np.isnan(dist_mat)] = 0

    # Symmetrize distance matrix (make it strictly symmetric)
    dist_mat = (dist_mat + dist_mat.T) * 0.5

    # Convert to condensed distance matrix for linkage
    dist_mat_condensed = scipy.spatial.distance.squareform(dist_mat)

    if verbose:
        log.info('Running hierarchical clustering...')

    hc = scipy.cluster.hierarchy.linkage(dist_mat_condensed, method='ward')

    if verbose:
        log.info('Ended hierarchical clustering')

    return {'gexp_roll_wide': gexp_roll_wide, 'hc': hc}


def get_internal_nodes(
    node: ClusterNode,
    node_id: Union[str, int],
    labels: List[str],
    clusters_dict: Dict[str, Union[str, int]]
    ) -> pd.DataFrame:
    """
    Recursively extract membership information for internal nodes in a hierarchical clustering tree.

    Parameters
    ----------
    node : scipy.cluster.hierarchy.ClusterNode
        Node object from scipy's linkage tree representing a cluster node.
        Must have `.pre_order()` method returning leaf indices, and `.left` and `.right` attributes.
    node_id : str or int
        Unique identifier for the current node.
    labels : List[str]
        List of leaf labels (e.g., cell names) ordered according to the dendrogram leaf order.
    clusters_dict : Dict[str, Union[str, int]]
        Mapping from leaf label to cluster assignment.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - 'cell': leaf labels under internal nodes where clusters are mixed.
        - 'node': node identifiers for these memberships.

        Returns an empty DataFrame if the node is a leaf node (all leaves belong to the same cluster).
    """
    indices = node.pre_order()  # indices of leaves under this node
    cell_labels = [labels[i] for i in indices]
    cluster_labels = [clusters_dict[label] for label in cell_labels]

    membership = pd.DataFrame({
        'cell': cell_labels,
        'node': [str(node_id)] * len(cell_labels)
    })

    is_leaf = len(np.unique(cluster_labels)) == 1

    if is_leaf:
        return pd.DataFrame()

    memberships = [membership]
    if getattr(node, "left", None) is not None:
        memberships.append(get_internal_nodes(node.left, f"{node_id}.1", labels, clusters_dict))
    if getattr(node, "right", None) is not None:
        memberships.append(get_internal_nodes(node.right, f"{node_id}.2", labels, clusters_dict))

    return pd.concat(memberships, ignore_index=True)


def get_nodes_celltree(
    hclust: Dict[str, Union[pd.DataFrame, np.ndarray]],
    k: int,
    debug_sort:bool=True
    ) -> Dict[str, Dict[str, Union[str, List[str], int]]]:
    """
    Generate a dictionary describing internal and terminal nodes in a hierarchical clustering tree,
    grouped by cluster assignments from a dendrogram cut.

    Parameters
    ----------
    hclust : dict
        Dictionary containing:
        - 'hc': linkage matrix as a NumPy ndarray from scipy.cluster.hierarchy.linkage.
        - 'gexp_roll_wide': AnnData-like object with `.obs_names` attribute listing cell labels.
    k : int
        Number of clusters to cut the dendrogram into.

    Returns
    -------
    Dict[str, Dict[str, Union[str, List[str], int]]]
        Dictionary keyed by node identifiers containing:
        - 'sample': node identifier (str)
        - 'members': list of unique cluster ids in the node
        - 'cells': list of cell labels in the node
        - 'size': number of cells in the node
    """
    # Cut dendrogram into k clusters
    clusters_array = fcluster(hclust['hc'], k, criterion='maxclust')
    
    if debug_sort:
        group_lab, group_count = np.unique(clusters_array, return_counts=True)
        sorted_count_index = np.argsort(group_count)[::-1]
        
        new_label = np.zeros(clusters_array.shape[0], dtype=np.int64)
        for idx, e in enumerate(group_lab):
            group_index = np.where(clusters_array == e)[0]
            new_label[group_index] = group_lab[sorted_count_index][idx]
        clusters_array = new_label

    # Cell labels in dendrogram leaf order
    cell_labels = list(hclust['gexp_roll_wide'].obs_names)

    # Map cells to clusters
    clusters_dict = dict(zip(cell_labels, clusters_array))

    # Convert linkage matrix to tree
    tree, labels = to_tree(hclust['hc'], rd=True)

    # Get internal node memberships
    nodes = get_internal_nodes(tree, '0', cell_labels, clusters_dict)

    # Prepare terminal nodes (leaves)
    terminal_nodes = pd.DataFrame({
        'cell': cell_labels,
        'node': [str(clusters_dict[label]) for label in cell_labels],
        'cluster': [clusters_dict[label] for label in cell_labels]
    })

    # Combine internal and terminal nodes
    nodes = pd.concat(
        [nodes.assign(cluster=[clusters_dict[cell] for cell in nodes['cell']]), terminal_nodes],
        ignore_index=True)

    # Group by node and summarize membership
    nodes_list = []
    grouped_nodes = nodes.groupby('node', sort=False, observed=True)
    for node_name, group in grouped_nodes:
        node_info = {
            'sample': node_name,
            'members': group['cluster'].unique().tolist(),
            'cells': group['cell'].tolist(),
            'size': len(group)
        }
        nodes_list.append(node_info)

    # Convert to dictionary keyed by node/sample id
    nodes_dict = {node['sample']: node for node in nodes_list}

    return nodes_dict

    
    
    
    