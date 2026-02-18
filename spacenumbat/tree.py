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



def _subtree_key(node: TreeNode) -> Tuple[int, str]:
    tips = list(node.tips()) if not node.is_tip() else [node]
    n = len(tips)
    min_tip = min((t.name or "" for t in tips), default="")
    return (n, min_tip)


def canonicalize_tree_inplace(tree: TreeNode, right: bool = True) -> TreeNode:
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
    
    labels = list(P_with_outgroup.index)
    D = pairwise_distances(P_with_outgroup.values, metric=metric, n_jobs=n_jobs)
    D = (D + D.T) * 0.5
    np.fill_diagonal(D, 0.0)
    return DistanceMatrix(D, labels)


def build_upgma_tree(dm) -> TreeNode:
    t = skbio.tree.upgma(dm)
    canonicalize_tree_inplace(t, right=True)  # to mimic ladderize/reorder
    return t


def build_nj_tree(dm: DistanceMatrix) -> TreeNode:
    t = skbio.tree.nj(dm)
    canonicalize_tree_inplace(t, right=True)
    return t


def root_and_prune_outgroup(tree: TreeNode, outgroup: str) -> TreeNode:
    """
    - Reroot on the edge leading to outgroup
    - Remove the outgroup tip
    - Prune unary nodes
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
    Returns:
      - ScorePlan for scoring (postorder internals)
      - node_to_row mapping for TreeNode -> global row id
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
    logQ[0:n_tips] already filled for tips.
    Fill internal rows in postorder: row = n_tips+i, children = child1[i], child2[i].
    """
    n_int = child1.shape[0]
    for i in range(n_int):
        uid = n_tips + i
        c1 = child1[i]
        c2 = child2[i]
        logQ[uid, :] = logQ[c1, :] + logQ[c2, :]


def compute_logQ_for_tree(
    plan: ScorePlan,
    P_df: pd.DataFrame,
    *,
    clip_eps: float = 1e-10,
    ) -> Tuple[np.ndarray, float]:
    """
    Returns:
      logQ (n_tips+n_int, m)
      L0 = sum(log(1-P))
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
    Fast scorer using the ScorePlan + numba propagation.
    Returns l_tree.
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
    For each column j:
      max1[j] = max over rows
      argmax[j] = row index achieving max1
      max2[j] = 2nd max over rows
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
    Rooted internal edges: parent and child are both internal.
    Equivalent to your _internal_edges_for_nni but on TreeNode objects.
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
    For each internal edge (p1->p2):
      e1 = sibling of p2 under p1
      e2,e3 = children of p2

    Returns arrays (length E):
      p2_rows, e1_rows, e2_rows, e3_rows
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
    NNI scoring:
      Only row p2 changes:
        cand0: logQ[p2] = logQ[e1] + logQ[e3]
        cand1: logQ[p2] = logQ[e1] + logQ[e2]
      Score per candidate = L0 + sum_j max( max_except_p2[j], new_row[j] )

    Returns scores of shape (E, 2) for E internal edges.
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
    Rooted NNI swap:
      P has children {C, S}; C has children {X, other}
      swap S and X:
        P children -> {C, X}
        C children -> {S, other}
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
    Greedy ML search using rooted NNI neighbors, but scoring all candidates
    without copying trees:
      - build logQ once for current tree
      - compute per-column max1/argmax/max2
      - for each internal edge, score 2 NNI alternatives by updating only row p2
      - pick best move; apply swap once; repeat
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
                print(f"[NNI-local] no internal edges at iter={it}, score={cur_score:.6g}")
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
                print(f"[NNI-local] converge at iter={it}, score={cur_score:.6g}")
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
            print(f"[NNI-local] iter={it}, score={best_score:.6g}, move=edge#{best_edge}, cand={best_cand}")

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
