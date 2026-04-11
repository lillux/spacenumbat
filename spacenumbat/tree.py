#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  3 17:21:12 2026

@author: carlino.calogero

"""

from dataclasses import dataclass
from typing import Tuple, List, Dict, Optional

import pandas as pd
from numba import njit, prange
from sklearn.metrics import pairwise_distances
import numpy as np
import skbio
from skbio import DistanceMatrix
from skbio.tree import TreeNode

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("This is an info message.")

def _subtree_key(node: TreeNode) -> Tuple[int, str]:
    """
    Compute a sortable key for a subtree.

    Parameters
    ----------
    node : TreeNode
        Input tree node.

    Returns
    -------
    tuple[int, str]
        Tuple containing subtree tip count and smallest tip name.
    """
    tips = list(node.tips()) if not node.is_tip() else [node]
    n = len(tips)
    min_tip = min((t.name or "" for t in tips), default="")
    return (n, min_tip)


def canonicalize_tree_inplace(tree: TreeNode, right: bool = True) -> TreeNode:
    """
    Canonicalize child ordering in a tree in place.

    Parameters
    ----------
    tree : TreeNode
        Input tree.
    right : bool, default=True
        Whether to reverse the sorted child order at each internal node.

    Returns
    -------
    TreeNode
        Input tree with reordered children.
    """
    for n in tree.postorder():
        if n.is_tip():
            continue
        n.children.sort(key=_subtree_key)
        if right:
            n.children.reverse()
    return tree


def build_distance_matrix_from_P(
    P_with_outgroup: pd.DataFrame,
    metric: str = "euclidean",
    n_jobs: int = 1,
    ) -> DistanceMatrix:
    """
    Build a symmetric distance matrix from a probability matrix.

    Parameters
    ----------
    P_with_outgroup : pd.DataFrame
        Matrix with samples as rows and features as columns.
    metric : str, default="euclidean"
        Distance metric passed to pairwise distance computation.
    n_jobs : int, default=1
        Number of parallel jobs used for distance computation.

    Returns
    -------
    DistanceMatrix
        Pairwise distance matrix with row labels from the input index.
    """
    labels = list(P_with_outgroup.index)
    D = pairwise_distances(P_with_outgroup.values, metric=metric, n_jobs=n_jobs)
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return DistanceMatrix(D, labels)


def build_upgma_tree(dm) -> TreeNode:
    """
    Build a canonicalized UPGMA tree from a distance matrix.

    Parameters
    ----------
    dm : DistanceMatrix
        Input distance matrix.

    Returns
    -------
    TreeNode
        UPGMA tree with canonicalized child ordering.
    """
    t = skbio.tree.upgma(dm)
    canonicalize_tree_inplace(t, right=True)  # to mimic ladderize/reorder
    return t


def build_nj_tree(dm: DistanceMatrix) -> TreeNode:
    """
    Build a canonicalized neighbor-joining tree from a distance matrix.

    Parameters
    ----------
    dm : DistanceMatrix
        Input distance matrix.

    Returns
    -------
    TreeNode
        Neighbor-joining tree with canonicalized child ordering.
    """
    t = skbio.tree.nj(dm)
    canonicalize_tree_inplace(t, right=True)
    return t


def root_and_prune_outgroup(tree: TreeNode, outgroup: str) -> TreeNode:
    """
    Root a tree on an outgroup and remove the outgroup tip.

    Parameters
    ----------
    tree : TreeNode
        Input unrooted or rooted tree.
    outgroup : str
        Tip name used as the outgroup.

    Returns
    -------
    TreeNode
        Rooted ingroup tree after outgroup removal and pruning.
    """
    # Re-root "above" the outgroup, inserting
    # a new root between outgroup and ingroup.
    rooted = tree.root_by_outgroup([outgroup], above=True, reset=True, inplace=False)

    # Drop the outgroup tip
    keep = [t.name for t in rooted.tips() if t.name != outgroup]
    ingroup = rooted.shear(keep)

    # Clean up any unary nodes after pruning
    ingroup.prune()

    return ingroup


def assert_P_invariants(P_df: pd.DataFrame) -> None:
    """
    Validate structural invariants of a probability matrix.

    Parameters
    ----------
    P_df : pd.DataFrame
        Probability matrix with cells as rows and segments as columns.

    Returns
    -------
    None
    """
    if not isinstance(P_df, pd.DataFrame):
        raise TypeError("P_df must be a pandas DataFrame.")
    if P_df.shape[0] == 0 or P_df.shape[1] == 0:
        raise ValueError("P_df must be non-empty (n_cells x n_segments).")
    if not P_df.index.is_unique:
        raise ValueError("P_df.index (cell barcodes) must be unique.")
    if P_df.index.isnull().any():
        raise ValueError("P_df.index contains null barcodes.")
    if P_df.columns.has_duplicates:
        raise ValueError("P_df.columns (segments) must be unique.")
    if P_df.columns.isnull().any():
        raise ValueError("P_df.columns contains null segment IDs.")


def assert_tree_matches_P(tree: TreeNode, P_df: pd.DataFrame) -> None:
    """
    Validate that tree tip names match the probability matrix index.

    Parameters
    ----------
    tree : TreeNode
        Input tree.
    P_df : pd.DataFrame
        Probability matrix with tip identifiers in the index.

    Returns
    -------
    None
    """
    tips = [t.name for t in tree.tips()]
    if any(x is None or x == "" for x in tips):
        raise ValueError("Tree contains unnamed tips; barcode names were lost.")
    if len(tips) != len(set(tips)):
        raise ValueError("Tree contains duplicate tip names.")
    missing = set(P_df.index) - set(tips)
    extra = set(tips) - set(P_df.index)
    if missing or extra:
        raise ValueError(
            f"Tree tips do not match P_df.index.\n"
            f"  missing_in_tree={len(missing)}\n"
            f"  extra_in_tree={len(extra)}"
        )
    return

@dataclass
class ScorePlan:
    n_tips: int
    n_int: int
    child1: np.ndarray   # (n_int,) global row ids
    child2: np.ndarray   # (n_int,) global row ids
    row_labels: List[str]


def build_score_plan(tree: TreeNode, P_index: List[str]) -> Tuple[ScorePlan, Dict[TreeNode, int]]:
    """
    Build the tree scoring plan and node-to-row mapping.

    Parameters
    ----------
    tree : TreeNode
        Input bifurcating tree.
    P_index : list[str]
        Tip labels in the order used by the probability matrix.

    Returns
    -------
    tuple[ScorePlan, dict[TreeNode, int]]
        Scoring plan and mapping from tree nodes to row indices.
    """
    tip_to_id = {name: i for i, name in enumerate(P_index)}

    internal_post = [u for u in tree.postorder() if not u.is_tip()]
    n = len(P_index)
    n_int = len(internal_post)

    node_to_row: Dict[TreeNode, int] = {}

    # tips: map by name
    for u in tree.tips():
        if u.name not in tip_to_id:
            raise ValueError(f"Tip '{u.name}' not found in P_df.index")
        node_to_row[u] = tip_to_id[u.name]

    # internals: appended in postorder
    for i, u in enumerate(internal_post):
        node_to_row[u] = n + i

    child1 = np.empty(n_int, dtype=np.int64)
    child2 = np.empty(n_int, dtype=np.int64)

    for i, u in enumerate(internal_post):
        if len(u.children) != 2:
            raise ValueError("Non-binary internal node encountered; expected bifurcating tree.")
        a, b = u.children
        child1[i] = node_to_row[a]
        child2[i] = node_to_row[b]

    row_labels = list(P_index) + [f"Node{i}" for i in range(n_int)]
    return ScorePlan(n_tips=n, n_int=n_int, child1=child1, child2=child2, row_labels=row_labels), node_to_row


@njit(cache=True)
def _propagate_logQ_numba(logQ: np.ndarray, child1: np.ndarray, child2: np.ndarray, n_tips: int) -> None:
    """
    Propagate tip log-scores to internal nodes in postorder.

    Parameters
    ----------
    logQ : np.ndarray
        Score matrix with tip rows already initialized.
    child1 : np.ndarray
        First child row index for each internal node.
    child2 : np.ndarray
        Second child row index for each internal node.
    n_tips : int
        Number of tip rows.

    Returns
    -------
    None
    """
    n_int = child1.shape[0]
    for i in range(n_int):
        uid = n_tips + i
        c1 = child1[i]
        c2 = child2[i]
        logQ[uid, :] = logQ[c1, :] + logQ[c2, :]
        
    return


def compute_logQ_for_tree(
    plan: ScorePlan,
    P_df: pd.DataFrame,
    clip_eps: float = 1e-10,
    ) -> Tuple[np.ndarray, float]:
    """
    Compute the tree log-score matrix and baseline log-likelihood term.

    Parameters
    ----------
    plan : ScorePlan
        Tree scoring plan.
    P_df : pd.DataFrame
        Probability matrix with tips as rows and sites as columns.
    clip_eps : float, default=1e-10
        Lower bound used for numerical stability.

    Returns
    -------
    tuple[np.ndarray, float]
        Log-score matrix and baseline log-likelihood term.
    """
    P = np.clip(P_df.values.astype(np.float64, copy=False), clip_eps, 1.0 - clip_eps)
    logP0 = np.log1p(-P)
    logP1 = np.log(P)
    L0 = float(logP0.sum())

    n, m = P.shape
    logQ = np.empty((plan.n_tips + plan.n_int, m), dtype=np.float64)

    logQ[:n, :] = logP1 - logP0
    _propagate_logQ_numba(logQ, plan.child1, plan.child2, plan.n_tips)

    return logQ, L0


@dataclass(frozen=True)
class ScoreTreeResult:
    l_tree: float
    logQ: np.ndarray
    l_matrix: Optional[np.ndarray]
    row_labels: List[str]
    
    
def score_tree_treenode_fast(
    tree: TreeNode,
    P_df: pd.DataFrame,
    get_l_matrix: bool = False,
    clip_eps: float = 1e-10,
    ) -> ScoreTreeResult:
    """
    Score a tree against a probability matrix by evaluating node-wise mutation likelihoods.

    This function validates that the tree tips match the rows of "P_df", builds a
    postorder scoring plan for the tree, and computes a node-by-site log-score
    matrix "logQ". Tip rows are initialized from the log-likelihood ratio
    "log(P) - log(1 - P)", and internal rows are obtained by summing child rows
    in postorder. The resulting matrix represents, for each node and site, the
    relative support for assigning that site to that node.

    The total tree score is then computed independently for each site by taking
    the maximum supported node assignment and summing across sites. When
    "get_l_matrix" is False, the score is obtained as:

    - max over rows of "logQ" for each site
    - plus the baseline term sum(log(1 - P)) across all cells and sites

    When "get_l_matrix" is True, the baseline term is added back to each column
    of "logQ" to produce "l_matrix", a node-by-site log-likelihood matrix on the
    original scale, and the tree score is computed by summing the per-site column
    maxima of "l_matrix".

    Parameters
    ----------
    tree : TreeNode
        Input bifurcating tree whose tip names must match "P_df.index".
    P_df : pd.DataFrame
        Probability matrix with tips as rows and sites as columns.
    get_l_matrix : bool, default=False
        Whether to return the node-by-site log-likelihood matrix.
    clip_eps : float, default=1e-10
        Lower bound used to clip probabilities away from 0 and 1 for numerical
        stability.

    Returns
    -------
    ScoreTreeResult
        Tree scoring result containing the total tree score, the node-by-site
        relative log-score matrix "logQ", the optional node-by-site
        log-likelihood matrix "l_matrix", and row labels for tips and internal
        nodes.
    """
    assert_P_invariants(P_df)
    assert_tree_matches_P(tree, P_df)

    plan, _ = build_score_plan(tree, list(P_df.index))
    logQ, L0 = compute_logQ_for_tree(plan, P_df, clip_eps=clip_eps)

    if get_l_matrix:
        P = np.clip(P_df.values.astype(np.float64, copy=False), clip_eps, 1.0 - clip_eps)
        logP0 = np.log1p(-P)
        col_add = logP0.sum(axis=0)
        l_matrix = logQ + col_add[None, :]
        l_tree = float(l_matrix.max(axis=0).sum())
    else:
        l_matrix = None
        l_tree = float(logQ.max(axis=0).sum() + L0)

    return ScoreTreeResult(
        l_tree=l_tree,
        logQ=logQ,
        l_matrix=l_matrix,
        row_labels=plan.row_labels)


@njit(cache=True, parallel=True)
def _col_max1_max2_argmax(logQ: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the top two row maxima and argmax for each column of a score matrix.

    Parameters
    ----------
    logQ : np.ndarray
        Input score matrix with rows as candidate nodes and columns as sites.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        First maximum, argmax row index, and second maximum for each column.
    """
    nrows, m = logQ.shape
    max1 = np.empty(m, dtype=np.float64)
    max2 = np.empty(m, dtype=np.float64)
    argm = np.empty(m, dtype=np.int64)

    neg_inf = -1.0e300

    for j in prange(m):
        m1 = neg_inf
        m2 = neg_inf
        a = -1
        for i in range(nrows):
            v = logQ[i, j]
            if v > m1:
                m2 = m1
                m1 = v
                a = i
            elif v > m2:
                m2 = v
        max1[j] = m1
        max2[j] = m2
        argm[j] = a

    return max1, argm, max2


def internal_edges_for_rooted_nni(tree: TreeNode) -> List[Tuple[TreeNode, TreeNode]]:
    """
    Collect internal parent-child edges eligible for rooted NNI.

    Parameters
    ----------
    tree : TreeNode
        Input rooted tree.

    Returns
    -------
    list[tuple[TreeNode, TreeNode]]
        Internal edges with both parent and child restricted to non-tip nodes.
    """
    edges: List[Tuple[TreeNode, TreeNode]] = []
    for child in tree.postorder():
        parent = child.parent
        if parent is None:
            continue
        if parent.is_tip() or child.is_tip():
            continue
        edges.append((parent, child))
    return edges


def build_nni_edge_arrays(
    tree: TreeNode,
    node_to_row: Dict[TreeNode, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build row-index arrays used to score rooted NNI moves.

    For each internal edge, this function identifies the child internal node,
    its sibling under the parent, and the two children of the internal node,
    then maps them to score-matrix row indices.

    Parameters
    ----------
    tree : TreeNode
        Input rooted bifurcating tree.
    node_to_row : dict[TreeNode, int]
        Mapping from tree nodes to score-matrix row indices.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Arrays of row indices for the internal child, its sibling, and its two
        children for each internal edge.
    """
    edges = internal_edges_for_rooted_nni(tree)

    p2_rows = np.empty(len(edges), dtype=np.int64)
    e1_rows = np.empty(len(edges), dtype=np.int64)
    e2_rows = np.empty(len(edges), dtype=np.int64)
    e3_rows = np.empty(len(edges), dtype=np.int64)

    for i, (p1, p2) in enumerate(edges):
        if len(p1.children) != 2 or len(p2.children) != 2:
            raise ValueError("Non-binary node encountered; expected bifurcating tree.")

        # sibling e1
        s = p1.children[0] if p1.children[1] is p2 else p1.children[1]
        a, b = p2.children[0], p2.children[1]

        p2_rows[i] = node_to_row[p2]
        e1_rows[i] = node_to_row[s]
        e2_rows[i] = node_to_row[a]
        e3_rows[i] = node_to_row[b]

    return p2_rows, e1_rows, e2_rows, e3_rows


@njit(cache=True, parallel=True)
def _nni_scores_from_logQ(
    logQ: np.ndarray,
    L0: float,
    max1: np.ndarray,
    argmax: np.ndarray,
    max2: np.ndarray,
    p2_rows: np.ndarray,
    e1_rows: np.ndarray,
    e2_rows: np.ndarray,
    e3_rows: np.ndarray,
    ) -> np.ndarray:
    """
    Score all rooted NNI candidates from a precomputed log-score matrix.

    For each internal edge, this function evaluates the two possible rooted NNI
    rearrangements by updating only the affected internal-node row and
    recomputing the tree score column-wise from the best available row score.

    Parameters
    ----------
    logQ : np.ndarray
        Node-by-site relative log-score matrix.
    L0 : float
        Baseline log-likelihood term.
    max1 : np.ndarray
        Per-column maximum over rows in "logQ".
    argmax : np.ndarray
        Per-column row index achieving "max1".
    max2 : np.ndarray
        Per-column second-largest row value in "logQ".
    p2_rows : np.ndarray
        Row indices of the internal child on each scored edge.
    e1_rows : np.ndarray
        Row indices of the sibling of the internal child.
    e2_rows : np.ndarray
        Row indices of the first child of the internal child.
    e3_rows : np.ndarray
        Row indices of the second child of the internal child.

    Returns
    -------
    np.ndarray
        Array of shape "(n_edges, 2)" containing the two NNI scores for each
        internal edge.
    """
    E = p2_rows.shape[0]
    m = logQ.shape[1]
    out = np.empty((E, 2), dtype=np.float64)

    for i in prange(E):
        p2 = p2_rows[i]
        e1 = e1_rows[i]
        e2 = e2_rows[i]
        e3 = e3_rows[i]

        s0 = L0
        s1 = L0

        for j in range(m):
            base = max2[j] if argmax[j] == p2 else max1[j]

            v0 = logQ[e1, j] + logQ[e3, j]
            v1 = logQ[e1, j] + logQ[e2, j]

            s0 += v0 if v0 > base else base
            s1 += v1 if v1 > base else base

        out[i, 0] = s0
        out[i, 1] = s1

    return out
    

def _swap_subtrees_inplace(P: TreeNode, C: TreeNode, S: TreeNode, X: TreeNode) -> None:
    """
    Apply a rooted NNI subtree swap in place.

    Parameters
    ----------
    P : TreeNode
        Parent node.
    C : TreeNode
        Internal child of "P".
    S : TreeNode
        Sibling of "C" under "P".
    X : TreeNode
        Child of "C" to swap with "S".

    Returns
    -------
    None
    """
    if C.parent is not P or S.parent is not P or X.parent is not C:
        raise RuntimeError("Invalid rooted NNI swap configuration.")
    P.remove(S)
    C.remove(X)
    P.append(X)
    C.append(S)
    return
    

def perform_nni_ml_greedy_local(
    tree_init: TreeNode,
    P_df: pd.DataFrame,
    eps: float = 1e-5,
    max_iter: int = 100,
    verbose: bool = True,
    clip_eps: float = 1e-10,
    ) -> List[TreeNode]:
    """
    Perform greedy local maximum-likelihood tree search using rooted NNI moves.

    Starting from an initial tree, this function repeatedly computes the
    node-by-site score matrix, evaluates all rooted NNI neighbors with fast
    row-update scoring, applies the best improving move, and stores each
    accepted tree until convergence or the iteration limit is reached.

    Parameters
    ----------
    tree_init : TreeNode
        Initial rooted tree.
    P_df : pd.DataFrame
        Probability matrix with tips as rows and sites as columns.
    eps : float, default=1e-5
        Minimum score improvement required to accept a move.
    max_iter : int, default=100
        Maximum number of NNI iterations.
    verbose : bool, default=True
        Whether to log search progress.
    clip_eps : float, default=1e-10
        Lower bound used for numerical stability in score computation.

    Returns
    -------
    list[TreeNode]
        Sequence of accepted trees, including the initial tree and final tree.
    """
    # Make copy
    cur = tree_init.copy()
    canonicalize_tree_inplace(cur, right=True)  # keep deterministic

    history = [cur.copy()]

    for it in range(1, max_iter + 1):
        # plan + mapping aligned to P_df.index
        plan, node_to_row = build_score_plan(cur, list(P_df.index))

        # Compute logQ and L0
        logQ, L0 = compute_logQ_for_tree(plan, P_df, clip_eps=clip_eps)

        # Current score (max over rows per column + L0)
        max1, argmax, max2 = _col_max1_max2_argmax(logQ)
        cur_score = float(max1.sum() + L0)

        # Build NNI edge arrays
        edges = internal_edges_for_rooted_nni(cur)
        if not edges:
            if verbose:
                log.info(f"[NNI-local] no internal edges at iter={it}, score={cur_score:.6g}")
            break

        p2_rows, e1_rows, e2_rows, e3_rows = build_nni_edge_arrays(cur, node_to_row)

        # Score all candidates fast (E x 2)
        scores = _nni_scores_from_logQ(logQ, L0, max1, argmax, max2, p2_rows, e1_rows, e2_rows, e3_rows)

        # Find best move
        flat = scores.reshape(-1)
        best_flat = int(np.argmax(flat))
        best_edge = best_flat // 2
        best_cand = best_flat % 2
        best_score = float(flat[best_flat])

        # Convergence check
        if (best_score - cur_score) <= eps:
            if verbose:
                log.info(f"[NNI-local] converge at iter={it}, score={cur_score:.6g}")
            break

        # Apply the best move IN PLACE (no copies)
        p1, p2 = edges[best_edge]

        # sibling S
        if len(p1.children) != 2 or len(p2.children) != 2:
            raise ValueError("Non-binary node encountered at move application.")
        S = p1.children[0] if p1.children[1] is p2 else p1.children[1]
        A, B = p2.children[0], p2.children[1]

        # cand0: swap S with A ; cand1: swap S with B
        X = A if best_cand == 0 else B
        _swap_subtrees_inplace(p1, p2, S, X)

        # Canonicalize once after acceptance
        canonicalize_tree_inplace(cur, right=True)

        history.append(cur.copy())

        if verbose:
            log.info(f"[NNI-local] iter={it}, score={best_score:.6g}, move=edge#{best_edge}, cand={best_cand}")

    return history


def P_to_candidate_tree(
    P_df: pd.DataFrame,
    outgroup_name: str = "outgroup",
    skip_nj: bool = False,
    n_jobs: int = 1,
    eps_nni: float = 1e-5,
    max_nni: int = 100,
    verbose: bool = True,
    ) -> pd.DataFrame:
    """
    Build a candidate maximum-likelihood tree from a probability matrix.

    This function adds an outgroup to the probability matrix, builds guide
    trees from pairwise distances, roots and prunes the outgroup, selects the
    better initial topology when both UPGMA and neighbor joining are used, and
    refines the selected tree by greedy rooted NNI search.

    Parameters
    ----------
    P_df : pd.DataFrame
        Probability matrix with tips as rows and sites as columns.
    outgroup_name : str, default="outgroup"
        Name assigned to the synthetic outgroup.
    skip_nj : bool, default=False
        Whether to skip neighbor-joining initialization and use only UPGMA.
    n_jobs : int, default=1
        Number of parallel jobs used for distance computation.
    eps_nni : float, default=1e-5
        Minimum score improvement required to accept an NNI move.
    max_nni : int, default=100
        Maximum number of NNI iterations.
    verbose : bool, default=True
        Whether to log search progress.

    Returns
    -------
    TreeNode
        Final candidate tree after guide-tree selection and local NNI
        optimization.
    """
    
    # Build guide trees (UPGMA/NJ, root+prune outgroup)
    P_with_out = P_df.copy()
    P_with_out.loc[outgroup_name, :] = 0.0

    dm = build_distance_matrix_from_P(P_with_out, metric="euclidean", n_jobs=n_jobs)

    t_upgma = root_and_prune_outgroup(build_upgma_tree(dm), outgroup=outgroup_name)

    if skip_nj:
        t_init = t_upgma
    else:
        t_nj = root_and_prune_outgroup(build_nj_tree(dm), outgroup=outgroup_name)
        up = score_tree_treenode_fast(t_upgma, P_df, get_l_matrix=False).l_tree
        nj = score_tree_treenode_fast(t_nj, P_df, get_l_matrix=False).l_tree
        t_init = t_upgma if up >= nj else t_nj

    canonicalize_tree_inplace(t_init, right=True)

    # ML tree search with fast local NNI
    tree_list = perform_nni_ml_greedy_local(
        tree_init=t_init,
        P_df=P_df,
        eps=eps_nni,
        max_iter=max_nni,
        verbose=verbose,
    )
    treeML = tree_list[-1]
    
    return treeML
