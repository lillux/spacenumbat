#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 20:15:57 2026

@author: lillux
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Any

import numpy as np
import pandas as pd
import networkx as nx

from skbio.tree import TreeNode

from spacenumbat.operations import _log_sum_exp
from spacenumbat.tree import score_tree_treenode_fast

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("This is an info message.")


def _split_muts(s: Optional[str]) -> List[str]:
    if s is None:
        return []
    s = str(s)
    if s == "" or s.lower() == "nan":
        return []
    return [x for x in (t.strip() for t in s.split(",")) if x != ""]


def _join_muts(muts: Iterable[str]) -> str:
    muts = [m for m in muts if m != ""]
    return ",".join(muts)


def _tree_root(tree: TreeNode) -> TreeNode:
    # skbio TreeNode root has parent None
    r = tree
    while r.parent is not None:
        r = r.parent
    return r


def tree_to_gtree_nx(tree: TreeNode) -> nx.DiGraph:
    """
    Build a deterministic node-id assignment from the current TreeNode topology.
    Names are taken from TreeNode.name for tips; internals are expected to have
    been named consistently (or will be assigned later from l_matrix indexing).
    """
    root = _tree_root(tree)

    # Deterministic traversal: BFS from root, children order as stored in tree
    order: List[TreeNode] = []
    q = [root]
    seen = set()
    while q:
        u = q.pop(0)
        if id(u) in seen:
            continue
        seen.add(id(u))
        order.append(u)
        for c in u.children:
            q.append(c)

    node_to_id: Dict[TreeNode, int] = {u: i for i, u in enumerate(order)}

    G = nx.DiGraph()
    for u, uid in node_to_id.items():
        leaf = u.is_tip()
        is_root = (u is root)

        # Name: tips must have barcodes; internals may be None at this stage
        nm = u.name if (u.name is not None and u.name != "") else None

        G.add_node(
            uid,
            name=nm,
            leaf=bool(leaf),
            root=bool(is_root),
            depth=None,   # computed next
            site=None,
            n_mut=0,
            GT="",
            last_mut="",
            clone=None,
            compartment=None,
            is_tumor_root=False,
        )

    # edges
    for u in order:
        uid = node_to_id[u]
        for c in u.children:
            cid = node_to_id[c]
            G.add_edge(uid, cid, leaf=bool(c.is_tip()), length=0.0)

    # depth from root
    depths = nx.single_source_shortest_path_length(G, node_to_id[root])
    nx.set_node_attributes(G, {nid: int(d) for nid, d in depths.items()}, "depth")

    return G


def _set_gtree_names_from_lmatrix_rows(gtree: nx.DiGraph, tree: TreeNode, row_labels: List[str]) -> None:
    """
    Enforce consistent naming:
      - tips: barcode names from tree tips
      - internals: Node0..Node{k-1} in the same internal postorder convention
        used by your build_score_plan() / row_labels

    """
    
    # This gives Node0..Node{k-1}.
    root = _tree_root(tree)

    # map TreeNode -> gtree node id: rebuild by BFS order used in tree_to_gtree_nx
    q = [root]
    order: List[TreeNode] = []
    seen = set()
    while q:
        u = q.pop(0)
        if id(u) in seen:
            continue
        seen.add(id(u))
        order.append(u)
        for c in u.children:
            q.append(c)
    tnode_to_gid = {u: i for i, u in enumerate(order)}

    # tips
    for tip in tree.tips():
        gid = tnode_to_gid[tip]
        gtree.nodes[gid]["name"] = tip.name

    # internals: postorder
    internals = [u for u in tree.postorder() if not u.is_tip()]
    for i, u in enumerate(internals):
        gid = tnode_to_gid[u]
        gtree.nodes[gid]["name"] = f"Node{i}"

    # all names non-null and unique
    names = [gtree.nodes[n]["name"] for n in gtree.nodes]
    if any(x is None or x == "" for x in names):
        raise ValueError("Some gtree nodes remained unnamed; check tree internal naming.")
    if len(names) != len(set(names)):
        # This is fatal for join logic downstream
        raise ValueError("gtree node names are not unique; check tree / naming pipeline.")

    return


def annotate_tree(
    tree: TreeNode,
    P_df: pd.DataFrame,
    clip_eps: float = 1e-10,
    ) -> nx.DiGraph:
    """
      - computes l_matrix
      - assigns each mutation (segment) to max-likelihood node
      - builds a gtree (nx.DiGraph) with node/edge annotations
      - calls mut_to_tree() to compute edge lengths and GT/last_mut fields

    """
    # score and l_matrix (rows: tips in P_df.index + internals Node0..)
    tree_stats = score_tree_treenode_fast(tree, P_df, get_l_matrix=True, clip_eps=clip_eps)
    l_matrix = tree_stats.l_matrix
    if l_matrix is None:
        raise RuntimeError("Expected l_matrix from score_tree_treenode_fast(get_l_matrix=True).")

    sites = list(P_df.columns)
    #n = P_df.shape[0]

    l_df = pd.DataFrame(l_matrix, index=tree_stats.row_labels, columns=sites)

    # mutation assignment on nodes (per column argmax)
    node_phylo = l_df.values.argmax(axis=0)
    lmax = l_df.values.max(axis=0)

    # map row index -> node name (barcodes for tips, Node{i} for internals)
    name_assigned = [tree_stats.row_labels[i] for i in node_phylo]

    mut_nodes = (pd.DataFrame({"site": sites, "name": name_assigned, "l": lmax})
                 .groupby("name", as_index=False)
                 .agg(site=("site", lambda x: ",".join(sorted(map(str, x)))),
                      n_mut=("site", "size"),
                      l=("l", "sum"),
                      ))

    # build gtree structure
    gtree = tree_to_gtree_nx(tree)
    _set_gtree_names_from_lmatrix_rows(gtree, tree, tree_stats.row_labels)

    # annotate sites onto nodes and derive GT and edge lengths
    gtree = mut_to_tree(gtree, mut_nodes)

    return gtree


def mut_to_tree(gtree: nx.DiGraph, mut_nodes: pd.DataFrame) -> nx.DiGraph:
    """
      - site: may be comma-separated bundle (e.g. "17b,8b")
      - GT: paste of all non-empty site bundles along root->node path
      - last_mut: the *last non-empty site bundle along the path* (inherited)
      - edge length: n_mut(child); leaf edges min 0.2
    """
    if "name" not in mut_nodes.columns or "site" not in mut_nodes.columns:
        raise ValueError("mut_nodes must contain at least columns ['name', 'site'].")

    mut_nodes = mut_nodes.copy()

    if "n_mut" not in mut_nodes.columns:
        mut_nodes["n_mut"] = mut_nodes["site"].map(lambda s: len(_split_muts(s)))

    name_to_row = mut_nodes.set_index("name", drop=False)

    for nid in gtree.nodes:
        gtree.nodes[nid]["site"] = None
        gtree.nodes[nid]["n_mut"] = 0
        gtree.nodes[nid].pop("clone", None)
        gtree.nodes[nid]["GT"] = ""
        gtree.nodes[nid]["last_mut"] = ""

    # Join mutation placements onto nodes by 'name'
    for nid, attrs in gtree.nodes(data=True):
        nm = attrs.get("name", None)
        if nm is not None and nm in name_to_row.index:
            site = name_to_row.loc[nm, "site"]
            if site is None:
                site_str = ""
            else:
                site_str = str(site)
                if site_str.lower() == "nan":
                    site_str = ""
            if site_str == "":
                gtree.nodes[nid]["site"] = None
                gtree.nodes[nid]["n_mut"] = 0
            else:
                gtree.nodes[nid]["site"] = site_str
                gtree.nodes[nid]["n_mut"] = int(name_to_row.loc[nm, "n_mut"])
        else:
            gtree.nodes[nid]["site"] = None
            gtree.nodes[nid]["n_mut"] = 0

    # Edge lengths
    for u, v, eattrs in gtree.edges(data=True):
        child_nmut = int(gtree.nodes[v].get("n_mut", 0))
        length = float(child_nmut)
        if bool(eattrs.get("leaf", False)):
            length = float(max(length, 0.2))
        gtree.edges[u, v]["length"] = length

    # Root
    roots = [n for n, a in gtree.nodes(data=True) if a.get("root", False)]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly 1 root in gtree, found {len(roots)}.")
    root = roots[0]

    def site_bundle(n: int) -> str:
        s = gtree.nodes[n].get("site", None)
        if s is None:
            return ""
        s = str(s)
        if s == "" or s.lower() == "nan":
            return ""
        return s

    def append_bundle(gt_prefix: str, bundle: str) -> str:
        if bundle == "":
            return gt_prefix
        return bundle if gt_prefix == "" else f"{gt_prefix},{bundle}"

    # Initialize root
    rsite = site_bundle(root)
    gtree.nodes[root]["GT"] = rsite
    gtree.nodes[root]["last_mut"] = rsite  # last non-empty along path so far

    # BFS traversal: inherit last_mut, extend GT only when site present
    for u in nx.bfs_tree(gtree, root):
        parent_GT = gtree.nodes[u].get("GT", "") or ""
        parent_last = gtree.nodes[u].get("last_mut", "") or ""

        for v in gtree.successors(u):
            v_site = site_bundle(v)

            # GT accumulates bundles along path (skip empty)
            v_GT = append_bundle(parent_GT, v_site)

            # last_mut is the last *non-empty* bundle encountered so far (inherit)
            v_last = v_site if v_site != "" else parent_last

            gtree.nodes[v]["GT"] = v_GT
            gtree.nodes[v]["last_mut"] = v_last

            if (gtree.nodes[v]["GT"] == "") and (gtree.nodes[v].get("site", None) is not None):
                gtree.nodes[v]["GT"] = str(gtree.nodes[v]["site"])

    if "GT" in mut_nodes.columns and "clone" in mut_nodes.columns:
        gt_to_clone = (
            mut_nodes.dropna(subset=["GT"])
            .astype({"GT": str})
            .set_index("GT")["clone"]
            .to_dict()
        )
        for nid, attrs in gtree.nodes(data=True):
            gt = attrs.get("GT", "") or ""
            if gt in gt_to_clone:
                gtree.nodes[nid]["clone"] = int(gt_to_clone[gt])

    return gtree



def mark_tumor_lineage(gtree: nx.DiGraph) -> nx.DiGraph:
    """
    
    """
    candidates = [n for n, a in gtree.nodes(data=True) if a.get("site", None) not in (None, "", "nan")]

    if not candidates:
        for n in gtree.nodes:
            gtree.nodes[n]["compartment"] = "normal"
            gtree.nodes[n]["is_tumor_root"] = False
        for u, v in gtree.edges:
            gtree.edges[u, v]["compartment"] = "normal"
        return gtree

    # per-leaf mut burden
    mut_burden = {}
    for n, a in gtree.nodes(data=True):
        gt = a.get("GT", "")
        mut_burden[n] = 0 if gt == "" else (gt.count(",") + 1)

    leaves = [n for n, a in gtree.nodes(data=True) if a.get("leaf", False)]

    cand_score: Dict[int, int] = {}
    for c in candidates:
        desc = nx.descendants(gtree, c) | {c}
        leaf_in_subtree = [l for l in leaves if l in desc]
        cand_score[c] = int(sum(mut_burden[l] for l in leaf_in_subtree))

    # tie-break by depth (prefer deeper), then by node id for determinism
    def _key(c: int):
        depth = int(gtree.nodes[c].get("depth", 0))
        return (cand_score.get(c, 0), depth, -c)  # score high, depth high, id small

    tumor_root = max(candidates, key=_key)

    tumor_subtree = nx.descendants(gtree, tumor_root) | {tumor_root}

    for n in gtree.nodes:
        in_tumor = (n in tumor_subtree)
        gtree.nodes[n]["compartment"] = "tumor" if in_tumor else "normal"
        gtree.nodes[n]["is_tumor_root"] = (n == tumor_root)

    for u, v in gtree.edges:
        gtree.edges[u, v]["compartment"] = gtree.nodes[v]["compartment"]

    return gtree


def label_edges(Gm: nx.DiGraph) -> nx.DiGraph:
    for u, v in Gm.edges:
        from_label = Gm.nodes[u].get("label", "")
        to_label = Gm.nodes[v].get("label", "")
        Gm.edges[u, v]["from_label"] = from_label
        Gm.edges[u, v]["to_label"] = to_label
        Gm.edges[u, v]["label"] = f"{from_label}->{to_label}"
    return Gm


def transfer_links(Gm: nx.DiGraph) -> nx.DiGraph:
    for u, v in Gm.edges:
        Gm.edges[u, v]["from_node"] = Gm.nodes[u].get("node", None)
        Gm.edges[u, v]["to_node"] = Gm.nodes[v].get("node", None)
    return Gm


def get_mut_graph(gtree: nx.DiGraph) -> nx.DiGraph:
    """
    
    """
    # find root
    roots = [n for n, a in gtree.nodes(data=True) if a.get("root", False)]
    if len(roots) != 1:
        raise ValueError(f"Expected exactly 1 root in gtree, found {len(roots)}.")
    #root = roots[0]

    # map last_mut label -> mut-graph vertex id (stable insertion order)
    label_to_vid: Dict[str, int] = {"": 0}
    next_vid = 1

    def _vid(lbl: Optional[str]) -> int:
        nonlocal next_vid
        lbl = "" if lbl is None else str(lbl)
        if lbl not in label_to_vid:
            label_to_vid[lbl] = next_vid
            next_vid += 1
        return label_to_vid[lbl]

    Gm = nx.DiGraph()
    Gm.add_node(0, label="", GT="", clone=None, node=None)

    # add nodes and edges from gtree
    for u, v in gtree.edges:
        lu = "" if gtree.nodes[u].get("last_mut", None) is None else str(gtree.nodes[u]["last_mut"])
        lv = "" if gtree.nodes[v].get("last_mut", None) is None else str(gtree.nodes[v]["last_mut"])

        vu = _vid(lu)
        vv = _vid(lv)

        if vu not in Gm:
            Gm.add_node(vu, label=lu, GT="", clone=None, node=None)
        if vv not in Gm:
            Gm.add_node(vv, label=lv, GT="", clone=None, node=None)

        if vu != vv:
            Gm.add_edge(vu, vv)

    # deterministic relabel to 0..V-1 by DFS from root
    dfs_order = list(nx.dfs_preorder_nodes(Gm, source=0))
    old_to_new = {old: i for i, old in enumerate(dfs_order)}

    Gm2 = nx.DiGraph()
    for old in dfs_order:
        Gm2.add_node(old_to_new[old], **dict(Gm.nodes[old]))
    for a, b in Gm.edges:
        if a in old_to_new and b in old_to_new:
            Gm2.add_edge(old_to_new[a], old_to_new[b])

    # robust label(last_mut) -> phylo node name mapping
    lastmut_to_node: Dict[str, str] = {}
    for _, attrs in gtree.nodes(data=True):
        nm = attrs.get("name", None)
        lm = attrs.get("last_mut", None)
        if nm is None:
            continue
        lm = "" if lm is None else str(lm)
        if lm != "":
            # first wins, deterministic given gtree node iteration order
            lastmut_to_node.setdefault(lm, str(nm))

    for vid, attrs in Gm2.nodes(data=True):
        lbl = attrs.get("label", "")
        lbl = "" if lbl is None else str(lbl)
        Gm2.nodes[vid]["node"] = lastmut_to_node.get(lbl, None)

    # finalize edge labels + link attrs
    label_edges(Gm2)
    transfer_links(Gm2)

    return Gm2


def label_genotype(Gm: nx.DiGraph, root: int = 0) -> nx.DiGraph:
    """
    
    """
    if root not in Gm:
        raise ValueError(f"Mutation graph must contain root node id {root}.")

    # Deterministic BFS arborescence from root
    T = nx.bfs_tree(Gm, source=root)

    # Parent mapping in BFS tree
    parent: Dict[int, Optional[int]] = {root: None}
    for u, v in T.edges():
        parent[v] = u

    # Initialize root
    Gm.nodes[root]["GT"] = ""
    # Compute GT in a parent-before-child order
    for v in nx.topological_sort(T):
        if v == root:
            continue
        p = parent.get(v, None)
        if p is None:
            continue  # unreachable nodes won't be labeled

        p_gt = Gm.nodes[p].get("GT", "") or ""
        lbl = Gm.nodes[v].get("label", "") or ""
        lbl = str(lbl)

        if lbl == "":
            Gm.nodes[v]["GT"] = p_gt
        else:
            Gm.nodes[v]["GT"] = f"{p_gt},{lbl}" if p_gt != "" else lbl

    # clone = DFS preorder rank on the BFS tree (0-based)
    dfs = list(nx.dfs_preorder_nodes(T, source=root))
    for i, v in enumerate(dfs):
        Gm.nodes[v]["clone"] = i

    return Gm


def _merge_two_vertices(Gm: nx.DiGraph, keep: int, drop: int, node_tar: Optional[str] = None) -> nx.DiGraph:
    """
      - merges 'drop' into 'keep'
      - combines labels: sorted unique of comma-split labels
      - reattaches edges (in and out)
      - removes self loops, simplifies multi-edges by DiGraph nature
    """
    keep_label = Gm.nodes[keep].get("label", "")
    drop_label = Gm.nodes[drop].get("label", "")
    combined = sorted(set(_split_muts(keep_label) + _split_muts(drop_label)))
    Gm.nodes[keep]["label"] = _join_muts(combined)

    if node_tar is not None:
        Gm.nodes[keep]["node"] = node_tar

    # redirect incoming edges to drop -> keep
    for u in list(Gm.predecessors(drop)):
        if u != keep:
            Gm.add_edge(u, keep)
    # redirect outgoing edges from drop -> keep
    for v in list(Gm.successors(drop)):
        if v != keep:
            Gm.add_edge(keep, v)

    # remove drop and clean up
    if drop in Gm:
        Gm.remove_node(drop)

    # remove self loops
    for u, v in list(Gm.edges):
        if u == v:
            Gm.remove_edge(u, v)

    # relabel nodes to 0..V-1 in DFS order from root (keeps deterministic)
    dfs = list(nx.dfs_preorder_nodes(Gm, source=0))
    mapping = {old: i for i, old in enumerate(dfs)}
    Gm = nx.relabel_nodes(Gm, mapping, copy=True)

    label_edges(Gm)
    transfer_links(Gm)

    return Gm


def get_move_cost(muts: str, node_ori: str, node_tar: str, l_df: pd.DataFrame) -> float:
    """

    """
    if muts is None:
        return float("inf")
    muts = str(muts)
    if muts == "":
        return float("inf")

    ms = _split_muts(muts)
    if len(ms) == 0:
        return float("inf")

    # if node names missing, cost is inf (cannot evaluate)
    if node_ori is None or node_tar is None:
        return float("inf")
    node_ori = str(node_ori)
    node_tar = str(node_tar)
    if node_ori not in l_df.index or node_tar not in l_df.index:
        return float("inf")

    # if some muts missing in columns, ignore them
    ms = [m for m in ms if m in l_df.columns]
    if not ms:
        return float("inf")

    return float((l_df.loc[node_ori, ms] - l_df.loc[node_tar, ms]).sum())


def get_move_opt(Gm: nx.DiGraph, l_df: pd.DataFrame) -> Dict[str, Any]:
    """
      - for each edge (from->to), compute:
          up   = cost(move to_label muts from to_node -> from_node)
          down = cost(move from_label muts from from_node -> to_node)
        but down is disabled if branching at 'from' (out_degree > 1)
      - pick minimal cost among all up/down moves
    """
    best = {"cost": float("inf")}

    for u, v in Gm.edges:
        from_label = Gm.nodes[u].get("label", "")
        to_label = Gm.nodes[v].get("label", "")
        from_node = Gm.nodes[u].get("node", None)
        to_node = Gm.nodes[v].get("node", None)

        n_sibling = Gm.out_degree(u)

        up_cost = get_move_cost(to_label, to_node, from_node, l_df)
        # down move blocked if branching
        down_cost = float("inf") if n_sibling > 1 else get_move_cost(from_label, from_node, to_node, l_df)

        if up_cost < best["cost"]:
            best = dict(
                cost=up_cost,
                direction="up",
                from_id=u,
                to_id=v,
                from_label=from_label,
                to_label=to_label,
                from_node=from_node,
                to_node=to_node,
            )
        if down_cost < best["cost"]:
            best = dict(
                cost=down_cost,
                direction="down",
                from_id=u,
                to_id=v,
                from_label=from_label,
                to_label=to_label,
                from_node=from_node,
                to_node=to_node,
            )

    return best


def simplify_history(
    Gm: nx.DiGraph,
    l_df: pd.DataFrame,
    max_cost: float = 150.0,
    n_cut: int = 0,
    verbose: bool = True,
    ) -> nx.DiGraph:
    """
      - iteratively finds the least-cost move (up/down) and contracts the edge endpoints
      - stops when cost >= max_cost OR edge count <= n_cut
      - if n_cut > 0: max_cost treated as Inf in R
    """
    if n_cut > 0:
        max_cost = float("inf")

    # loop up to number of edges
    for _ in range(Gm.number_of_edges()):
        move = get_move_opt(Gm, l_df)

        if not np.isfinite(move["cost"]):
            break

        if (move["cost"] < max_cost) and (Gm.number_of_edges() > n_cut):
            u = move["from_id"]
            v = move["to_id"]

            if move["direction"] == "up":
                # contract (from_label, to_label) into 'from' node, node_tar=from_node
                Gm = _merge_two_vertices(Gm, keep=u, drop=v, node_tar=move.get("from_node", None))
                if verbose:
                    log.info(f"opt_move:{move['to_label']}->{move['from_label']}, cost={move['cost']:.3g}")
            else:
                # contract into 'to' node, node_tar=to_node
                Gm = _merge_two_vertices(Gm, keep=v, drop=u, node_tar=move.get("to_node", None))
                if verbose:
                    log.info(f"opt_move:{move['from_label']}->{move['to_label']}, cost={move['cost']:.3g}")
        else:
            break

    return Gm


@dataclass(frozen=True)
class TreePost:
    gtree: nx.DiGraph
    l_df: pd.DataFrame


def get_tree_post(tree: TreeNode, P_df: pd.DataFrame, clip_eps: float = 1e-10) -> TreePost:
    """

    """
    tree_stats = score_tree_treenode_fast(tree, P_df, get_l_matrix=True, clip_eps=clip_eps)
    if tree_stats.l_matrix is None:
        raise RuntimeError("Expected l_matrix from score_tree_treenode_fast(get_l_matrix=True).")
    l_df = pd.DataFrame(tree_stats.l_matrix, index=tree_stats.row_labels, columns=P_df.columns)

    gtree = annotate_tree(tree, P_df, clip_eps=clip_eps)
    return TreePost(gtree=gtree, l_df=l_df)


def get_gtree(
    tree: TreeNode,
    P_df: pd.DataFrame,
    n_cut: int = 0,
    max_cost: float = 0.0,
    clip_eps: float = 1e-10,
    verbose: bool = True,
    ) -> nx.DiGraph:
    """
      - computes l_matrix + initial gtree
      - builds mut graph, simplifies history, labels genotype
      - transfers back onto gtree with clone ids
      - marks tumor lineage
    """
    post = get_tree_post(tree, P_df, clip_eps=clip_eps)

    Gm = get_mut_graph(post.gtree)
    Gm = simplify_history(Gm, post.l_df, max_cost=max_cost, n_cut=n_cut, verbose=verbose)
    Gm = label_genotype(Gm)

    # build mut_nodes table:
    vertices = []
    for vid, a in Gm.nodes(data=True):
        vertices.append(
            dict(
                name=a.get("node", None),
                site=a.get("label", None),
                clone=a.get("clone", None),
                GT=a.get("GT", None),
            )
        )
    mut_nodes = pd.DataFrame(vertices) #.dropna(subset=["name"])  # only those that map to phylo nodes

    gtree = mut_to_tree(post.gtree, mut_nodes)
    gtree = mark_tumor_lineage(gtree)
    return gtree



def get_clone_post(
    gtree: nx.DiGraph,
    exp_post: pd.DataFrame,
    allele_post: pd.DataFrame,
    seg_col: str = "seg",
    cell_col: str = "cell",
    cnv_state_col: str = "cnv_state",
    Z_cnv_col: str = "Z_cnv",
    Z_n_col: str = "Z_n",
    ) -> pd.DataFrame:
    """
    
    """
    # clones table from gtree nodes
    nodes_df = pd.DataFrame([dict(GT=a.get("GT", "") if a.get("GT", "") is not None else "",
                                  clone=a.get("clone", np.nan),
                                  compartment=a.get("compartment", np.nan),
                                  leaf=bool(a.get("leaf", False)),)
                             for _, a in gtree.nodes(data=True)])

    clones = (nodes_df.groupby(["GT", "clone", "compartment"], dropna=False, as_index=False)
              .agg(clone_size=("leaf", "sum")))

    # ensure normal genotype exists (R: add normal if min(clone) > 1; Python 0-based => > 0)
    if clones["clone"].dropna().shape[0] == 0 or clones["clone"].dropna().min() > 0:
        clones = pd.concat([pd.DataFrame([dict(GT="", clone=0, compartment="normal", clone_size=0)]), clones],
                           ignore_index=True,)

    # prior_clone
    unique_gt = clones["GT"].astype(str).unique().tolist()
    tumor_gts = [g for g in unique_gt if g != ""]
    K = len(tumor_gts)

    def _prior(gt: str) -> float:
        return 0.5 if gt == "" else (0.5 / max(K, 1))

    clones["prior_clone"] = clones["GT"].astype(str).map(_prior)

    # clone_segs
    # Compute universe of seg values from GT strings
    seg_universe = sorted({s for gt in clones["GT"].astype(str).tolist() for s in _split_muts(gt) if s != ""})

    base = clones[["GT", "clone", "compartment", "prior_clone", "clone_size"]].drop_duplicates().copy()
    if len(seg_universe) > 0:
        base["_tmp"] = 1
        seg_df = pd.DataFrame({seg_col: seg_universe})
        seg_df["_tmp"] = 1

        clone_segs = seg_df.merge(base, on="_tmp").drop(columns=["_tmp"])

        gt_to_set = {gt: set(_split_muts(gt)) for gt in base["GT"].astype(str).unique().tolist()}
        clone_segs["I"] = [1 if seg in gt_to_set.get(gt, set()) else 0
                           for seg, gt in zip(clone_segs[seg_col].astype(str).tolist(),
                                              clone_segs["GT"].astype(str).tolist())]
        clone_segs["I"] = clone_segs["I"].astype(int)
    else:
        clone_segs = pd.DataFrame(columns=[seg_col, "GT", "clone", "compartment", "prior_clone", "clone_size", "I"])

    # likelihood aggregation blocks
    def _block(post: pd.DataFrame, suffix: str) -> pd.DataFrame:
        post = post.copy()
        post = post[post[cnv_state_col] != "neu"]
        post = post.merge(clone_segs, left_on=seg_col, right_on=seg_col, how="inner")
        post["l_clone"] = np.where(post["I"].to_numpy() == 1, post[Z_cnv_col].to_numpy(), post[Z_n_col].to_numpy())
        out = (post.groupby([cell_col, "clone", "GT", "prior_clone"], as_index=False)
               .agg(**{f"l_clone_{suffix}": ("l_clone", "sum")}))
        return out

    x = _block(exp_post, "x")
    y = _block(allele_post, "y")
    merged = x.merge(y, on=[cell_col, "clone", "GT", "prior_clone"], how="outer")

    if merged.shape[0] == 0:
        # R-faithful join logic yields empty when no overlapping seg evidence exists.
        # For pipeline compatibility, fall back to prior-only per-cell rows (no synthetic likelihood evidence).
        cells = pd.unique(pd.concat([
            exp_post[cell_col] if cell_col in exp_post.columns else pd.Series(dtype=object),
            allele_post[cell_col] if cell_col in allele_post.columns else pd.Series(dtype=object),
        ], ignore_index=True)).tolist()

        if len(cells) == 0:
            clone_post = pd.DataFrame(columns=[cell_col, "clone_opt", "GT_opt", "p_opt"])
            tumor_clones = clones.loc[clones["compartment"].astype(str) == "tumor", "clone"].dropna().astype(int).tolist()
            for c in tumor_clones:
                clone_post[f"p_{c}"] = pd.Series(dtype=float)
                clone_post[f"p_x_{c}"] = pd.Series(dtype=float)
                clone_post[f"p_y_{c}"] = pd.Series(dtype=float)
            clone_post["p_cnv"] = pd.Series(dtype=float)
            clone_post["p_cnv_x"] = pd.Series(dtype=float)
            clone_post["p_cnv_y"] = pd.Series(dtype=float)
            clone_post["compartment_opt"] = pd.Series(dtype=object)
            clone_post["sample"] = pd.Series(dtype=object)
            return clone_post

        merged = base[["GT", "clone", "prior_clone"]].copy()
        merged["_tmp"] = 1
        cells_df = pd.DataFrame({cell_col: cells, "_tmp": 1})
        merged = cells_df.merge(merged, on="_tmp", how="inner").drop(columns=["_tmp"])
        merged["l_clone_x"] = 0.0
        merged["l_clone_y"] = 0.0

    merged["l_clone_x"] = merged["l_clone_x"].fillna(0.0)
    merged["l_clone_y"] = merged["l_clone_y"].fillna(0.0)

    # compute posteriors per cell
    merged["Z_clone"] = np.log(merged["prior_clone"].to_numpy()) + merged["l_clone_x"].to_numpy() + merged["l_clone_y"].to_numpy()
    merged["Z_clone_x"] = np.log(merged["prior_clone"].to_numpy()) + merged["l_clone_x"].to_numpy()
    merged["Z_clone_y"] = np.log(merged["prior_clone"].to_numpy()) + merged["l_clone_y"].to_numpy()

    merged["p"] = np.nan
    merged["p_x"] = np.nan
    merged["p_y"] = np.nan

    for cell, idx in merged.groupby(cell_col, sort=False).groups.items():
        z = merged.loc[idx, "Z_clone"].to_numpy()
        zx = merged.loc[idx, "Z_clone_x"].to_numpy()
        zy = merged.loc[idx, "Z_clone_y"].to_numpy()
        merged.loc[idx, "p"] = np.exp(z - _log_sum_exp(z))
        merged.loc[idx, "p_x"] = np.exp(zx - _log_sum_exp(zx))
        merged.loc[idx, "p_y"] = np.exp(zy - _log_sum_exp(zy))

    # clone_opt / GT_opt / p_opt
    def _opt_block(df: pd.DataFrame) -> pd.Series:
        # Use the max-posterior row directly. This is robust in Python and preserves
        # downstream GT-based joins used by subtree construction.
        i = int(df["p"].to_numpy().argmax())
        clone_val = df["clone"].to_numpy()[i]
        return pd.Series({"clone_opt": int(clone_val) if pd.notna(clone_val) else np.nan,
                          "GT_opt": df["GT"].to_numpy()[i],
                          "p_opt": float(df["p"].to_numpy()[i]),
                          })

    opt = merged.groupby(cell_col, as_index=False).apply(_opt_block).reset_index(drop=True)
    merged2 = merged.merge(opt, on=cell_col, how="left")

    # wide output for p, p_x, p_y
    piv_p = merged2.pivot_table(index=[cell_col, "clone_opt", "GT_opt", "p_opt"], columns="clone", values="p", fill_value=0.0)
    piv_px = merged2.pivot_table(index=[cell_col, "clone_opt", "GT_opt", "p_opt"], columns="clone", values="p_x", fill_value=0.0)
    piv_py = merged2.pivot_table(index=[cell_col, "clone_opt", "GT_opt", "p_opt"], columns="clone", values="p_y", fill_value=0.0)

    piv_p.columns = [f"p_{int(c)}" for c in piv_p.columns]
    piv_px.columns = [f"p_x_{int(c)}" for c in piv_px.columns]
    piv_py.columns = [f"p_y_{int(c)}" for c in piv_py.columns]

    clone_post = pd.concat([piv_p, piv_px, piv_py], axis=1).reset_index()

    # p_cnv = sum_{tumor clones} p_* and compartment_opt
    tumor_clones = clones.loc[clones["compartment"].astype(str) == "tumor", "clone"].dropna().astype(int).tolist()

    def _row_sum_cols(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return np.zeros(len(df), dtype=float)
        return df[cols].to_numpy(dtype=float).sum(axis=1)

    clone_post["p_cnv"] = _row_sum_cols(clone_post, [f"p_{c}" for c in tumor_clones])
    clone_post["p_cnv_x"] = _row_sum_cols(clone_post, [f"p_x_{c}" for c in tumor_clones])
    clone_post["p_cnv_y"] = _row_sum_cols(clone_post, [f"p_y_{c}" for c in tumor_clones])

    clone_post["compartment_opt"] = np.where(clone_post["p_cnv"].to_numpy() > 0.5, "tumor", "normal")
    clone_post["sample"] = clone_post["clone_opt"].astype(str)

    return clone_post
