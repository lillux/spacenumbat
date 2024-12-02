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

from scipy.cluster.hierarchy import linkage, fcluster, to_tree
from sklearn.metrics import pairwise_distances

from spacenumbat.utils import filter_genes

import logging


def scale_counts(x:scipy.sparse.spmatrix) -> scipy.sparse.spmatrix:
    return x / x.sum(1)

def choose_ref_cor(count_mat:ad.AnnData, lambdas_ref:pd.DataFrame, gtf:pd.DataFrame):
    
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

    count_mat = count_mat[:,genes_annotated].copy()
    lambdas_ref_annot = lambdas_ref.loc[genes_annotated,:]

    count_mat.layers['X_norm'] = scale_counts(count_mat.X)
    count_mat.layers['X'] = count_mat.X.copy()
    count_mat.X = count_mat.layers['X_norm']
    count_mat.X = scipy.sparse.csr_matrix(count_mat.X)
    count_mat = count_mat[:,lambdas_ref_annot[np.sum(lambdas_ref_annot*1e6 > 2, axis=1) > 0].index]

    zero_cov = count_mat.obs.loc[count_mat.X.sum(1).A == 0,:].index # CORRECT

    if len(zero_cov) == 0:
        logging.warning(f'Cannot choose reference for {len(zero_cov)} cells due to low coverage')
        count_mat = count_mat[[i for i in count_mat.obs.index if i not in set(zero_cov)],:]

    # homemade vectorized correlation
    c_mat = np.log1p(count_mat.X.toarray() * 1e6)
    ref_mat = np.log1p(lambdas_ref_annot * 1e6).loc[count_mat.var_names,:].values

    c_mat_centered = c_mat - c_mat.mean(axis=1, keepdims=True)
    c_mat_std = c_mat_centered / c_mat.std(axis=1, ddof=0, keepdims=True)
    
    ref_mat_centered = ref_mat - ref_mat.mean(axis=0)
    ref_mat_std = ref_mat_centered / ref_mat.std(axis=0, ddof=0)
    
    cor_mat_dot = np.dot(c_mat_std, ref_mat_std) / c_mat.shape[1]
    cors_dot = pd.DataFrame(cor_mat_dot, columns=lambdas_ref.columns, index=count_mat.obs_names)

    best_refs = cors_dot.idxmax(axis=1)

    return best_refs


def get_lambdas_bar(lambdas_ref, sc_refs, verbose=True):
    ref_counts = sc_refs.value_counts(normalize=True)
    lambdas_bar = {}
    
    for cell_type in lambdas_ref.columns:
        try:
            lambdas_bar[cell_type] = ref_counts[cell_type]
        except KeyError:
            lambdas_bar[cell_type] = 0    
            
    lambdas_bar = pd.Series(lambdas_bar)
    if verbose:
        print(f"Fitted reference proportions: {', '.join([f'{name}: {round(val, 3)}' for name, val in lambdas_bar.items()])}")
    
    return np.matmul(lambdas_ref, lambdas_bar)


def smooth_expression(count_mat, lambdas_ref, gtf, window=101, cap=3, verbose=False, filter_hla=True):
    # Filter mutually expressed genes
    mut_expressed = filter_genes(count_mat, lambdas_ref, gtf, filter_hla=filter_hla)
    count_mat = count_mat[:,mut_expressed].copy()
    lambdas_ref = lambdas_ref.loc[mut_expressed]
    
    # Normalize counts
    exp_mat = scale_counts(count_mat.X).toarray()
    
    # Log transformation and normalization
    exp_mat_norm = np.log2(exp_mat * 1e6 + 1) - np.log2(lambdas_ref * 1e6 + 1).values
    
    # Cap values to a range of [-cap, cap]
    exp_mat_norm = np.clip(exp_mat_norm, -cap, cap)
    
    # Centering by cell (subtract row means)
    row_means = exp_mat_norm.mean(axis=1, keepdims=True)
    exp_mat_norm -= row_means
    exp_mat_norm = pd.DataFrame(exp_mat_norm)

    # Apply rolling window smoothing
    exp_mat_smooth = exp_mat_norm.T.rolling(window=window, center=True, min_periods=1).mean()
    count_mat.layers['X_smooth'] = scipy.sparse.csr_matrix(exp_mat_smooth.values.T)
    count_mat.layers['X'] = count_mat.X
    count_mat.X = count_mat.layers['X_smooth']

    # return AnnData with a smoothed X matrix. The original X is stored in layers.
    return count_mat


def exp_hclust(count_mat, lambdas_ref, gtf, sc_refs=None, window=101, ncores=1, verbose=True):
    # count_mat = check_matrix(count_mat)
    
    if sc_refs is None:
        sc_refs = choose_ref_cor(count_mat, lambdas_ref, gtf)
    
    lambdas_bar = get_lambdas_bar(lambdas_ref, sc_refs, verbose=False)

    gexp_roll_wide = smooth_expression(count_mat, lambdas_bar, gtf, window=window, verbose=verbose)

    # Use parallel pairwise distance computation (mimicking `parallelDist::parDist`)
    dist_mat = pairwise_distances(gexp_roll_wide, metric='euclidean', n_jobs=ncores)

    # Fill NaNs with zeros (as in R code if there are missing values)
    dist_mat[np.isnan(dist_mat)] = 0

    print('Running hierarchical clustering...')
    hc = linkage(dist_mat, method='ward')  # equivalent to 'ward.D2' in R
    
    return {
        'gexp_roll_wide': gexp_roll_wide,
        'hc': hc
    }


def get_internal_nodes(node, node_id, labels, clusters_dict):
    indices = node.pre_order()  # Get indices of leaves under this node
    cell_labels = [labels[i] for i in indices]
    cluster_labels = [clusters_dict[label] for label in cell_labels]

    # Create a DataFrame with cell memberships for this node
    membership = pd.DataFrame({
        'cell': cell_labels,
        'node': [node_id] * len(cell_labels)
    })

    # Check if all leaves under this node belong to the same cluster
    is_leaf = len(np.unique(cluster_labels)) == 1

    if is_leaf:
        return pd.DataFrame()

    # Recursively get memberships from child nodes
    memberships = [membership]
    if node.left is not None:
        memberships.append(get_internal_nodes(node.left, f"{node_id}.1", labels, clusters_dict))
    if node.right is not None:
        memberships.append(get_internal_nodes(node.right, f"{node_id}.2", labels, clusters_dict))

    return pd.concat(memberships, ignore_index=True)


def get_nodes_celltree(hclust, k):
    # Cut the dendrogram into 'k' clusters
    clusters_array = fcluster(hclust['hc'], k, criterion='maxclust')
    
    # Assuming you have the cell labels in the same order as the observations used to compute 'hc'
    # For example, if 'gexp_roll_wide' was used to compute 'hc', and its index contains the cell labels
    cell_labels = list(hclust['gexp_roll_wide'].obs_names)
    
    # Map cell labels to cluster assignments
    clusters_dict = dict(zip(cell_labels, clusters_array))
    
    # Convert the linkage matrix to a tree
    tree, labels = to_tree(hclust['hc'], rd=True)
    
    # Now, get internal nodes
    nodes = get_internal_nodes(tree, '0', cell_labels, clusters_dict)
    
    # Add terminal nodes (leaves)
    terminal_nodes = pd.DataFrame({
        'cell': cell_labels,
        'node': [str(clusters_dict[label]) for label in cell_labels],
        'cluster': [clusters_dict[label] for label in cell_labels]
    })
    
    # Combine internal nodes and terminal nodes
    nodes = pd.concat(
        [nodes.assign(cluster=[clusters_dict[cell] for cell in nodes['cell']]), terminal_nodes],
        ignore_index=True
    )
    
    # Group by 'node' and create a list of dictionaries
    nodes_list = []
    grouped_nodes = nodes.groupby('node')
    for node_name, group in grouped_nodes:
        node_info = {
            'sample': node_name,
            'members': group['cluster'].unique().tolist(),
            'cells': group['cell'].tolist(),
            'size': len(group)
        }
        nodes_list.append(node_info)
    
    # Convert to a dictionary
    nodes_dict = {node['sample']: node for node in nodes_list}
    
    return nodes_dict