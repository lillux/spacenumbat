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
    """
    Split a comma-separated mutation string into a list of mutation labels.

    Parameters
    ----------
    s : str or None
        Comma-separated mutation string.

    Returns
    -------
    list of str
        Non-empty mutation labels with surrounding whitespace removed.
    """
    if s is None:
        return []
    s = str(s)
    if s == "" or s.lower() == "nan":
        return []
    return [x for x in (t.strip() for t in s.split(",")) if x != ""]


def _join_muts(muts: Iterable[str]) -> str:
    """
    Join mutation labels into a comma-separated string.

    Parameters
    ----------
    muts : iterable of str
        Mutation labels to join.

    Returns
    -------
    str
        Comma-separated mutation string with empty entries removed.
    """
    muts = [m for m in muts if m != ""]
    return ",".join(muts)


def _tree_root(tree: TreeNode) -> TreeNode:
    """
    Return the root node of a tree.

    Parameters
    ----------
    tree : TreeNode
        Any node in the tree.

    Returns
    -------
    TreeNode
        Root node of the tree.
    """
    # skbio TreeNode root has parent None
    r = tree
    while r.parent is not None:
        r = r.parent
    return r


def tree_to_gtree_nx(tree: TreeNode) -> nx.DiGraph:
    """
    Convert a scikit-bio tree into a directed NetworkX graph.

    Nodes are assigned deterministic integer identifiers by breadth-first
    traversal from the root, using the stored child order.

    Parameters
    ----------
    tree : TreeNode
        Input tree.

    Returns
    -------
    nx.DiGraph
        Directed graph with node and edge attributes describing the tree.
    """
    root = _tree_root(tree)

    # BFS from root, children order as stored in tree
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

        # tips must have barcodes; internals may be None
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
    Assign node names in a graph tree to match l-matrix row conventions.

    Tip nodes are named from tree tip names. Internal nodes are named in
    postorder as "Node0", "Node1", and so on.

    Parameters
    ----------
    gtree : nx.DiGraph
        Directed graph representation of the tree.
    tree : TreeNode
        Input phylogenetic tree.
    row_labels : sequence of str
        L-matrix row labels.

    Returns
    -------
    None
    """
    
    # This gives Node0..Node{k-1}.
    root = _tree_root(tree)

    # rebuild by BFS order used in tree_to_gtree_nx
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
        msg = "Some gtree nodes remained unnamed; check tree internal naming."
        log.error(msg)
    if len(names) != len(set(names)):
        msg = "gtree node names are not unique; check tree / naming pipeline."
        log.error(msg)

    return


def annotate_tree(
    tree: TreeNode,
    P_df: pd.DataFrame,
    clip_eps: float = 1e-10,
    ) -> nx.DiGraph:
    """
    Annotate a phylogenetic tree with mutation assignments and genotype labels.

    Parameters
    ----------
    tree : TreeNode
        Input phylogenetic tree.
    P_df : pd.DataFrame
        Probability matrix with tips as rows and mutation sites as columns.
    clip_eps : float, default=1e-10
        Lower bound used during tree scoring for numerical stability.

    Returns
    -------
    nx.DiGraph
        Annotated directed graph representation of the tree.
    """
    # score and l_matrix (rows: tips in P_df.index + internals Node0..)
    tree_stats = score_tree_treenode_fast(tree, P_df, get_l_matrix=True, clip_eps=clip_eps)
    l_matrix = tree_stats.l_matrix
    sites = list(P_df.columns)
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
    Annotate a tree graph with mutation bundles, genotypes, and edge lengths.

    Parameters
    ----------
    gtree : nx.DiGraph
        Directed tree graph with node names and a single root.
    mut_nodes : pd.DataFrame
        Mutation assignments by node name. Must contain a ``name`` column and a
        "site" column. An "n_mut" column is optional.

    Returns
    -------
    nx.DiGraph
        Input graph with updated node and edge attributes.
    """
    
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

    # inherit last_mut, extend GT only when site present
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

    # canonicalize clone ids on the full gtree to keep GT<->clone one-to-one.
    gtree_nodes = pd.DataFrame(
        [{"GT": attrs.get("GT", "") if attrs.get("GT", "") is not None else "",
          "clone": attrs.get("clone", np.nan)}
         for _, attrs in gtree.nodes(data=True)]
    )
    gt_to_clone = _build_canonical_gt_clone_map(gtree_nodes["GT"], gtree_nodes["clone"])
    for nid, attrs in gtree.nodes(data=True):
        gt = _normalize_gt(attrs.get("GT", ""))
        gtree.nodes[nid]["clone"] = int(gt_to_clone.get(gt, 0))

    return gtree


def mark_tumor_lineage(gtree: nx.DiGraph) -> nx.DiGraph:
    """
    Label tumor and normal compartments on a mutation-annotated tree.

    Parameters
    ----------
    gtree : nx.DiGraph
        Directed tree graph with node mutation and topology annotations.

    Returns
    -------
    nx.DiGraph
        Input graph with updated node and edge compartment labels.
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


def _norm_label(x: Optional[str]) -> str:
    """
    Normalize a node label to a non-null string.

    Parameters
    ----------
    x : str or None
        Input label.

    Returns
    -------
    str
        Normalized label, with "None" and "nan" mapped to "".
    """
    if x is None:
        return ""
    x = str(x)
    return "" if x.lower() == "nan" else x


def _graph_root(G: nx.DiGraph) -> int:
    """
    Return the root node of a directed tree graph.

    Parameters
    ----------
    G : nx.DiGraph
        Directed graph with a single root.

    Returns
    -------
    int
        Root node identifier.
    """
    roots = [n for n in G.nodes if G.in_degree(n) == 0]

    return roots[0]


def _reindex_graph_from_root(G: nx.DiGraph, root: int) -> nx.DiGraph:
    """
    Reindex graph nodes by depth-first preorder from a root.

    Parameters
    ----------
    G : nx.DiGraph
        Input directed graph.
    root : int
        Root node identifier.

    Returns
    -------
    nx.DiGraph
        Relabeled graph with nodes indexed from "0" to "n - 1".
    """
    dfs_order = list(nx.dfs_preorder_nodes(G, source=root))
    rest = [n for n in G.nodes if n not in dfs_order]
    order = dfs_order + rest
    mapping = {old: i for i, old in enumerate(order)}
    return nx.relabel_nodes(G, mapping, copy=True)


def label_edges(Gm: nx.DiGraph) -> nx.DiGraph:
    """
    Annotate graph edges with source, target, and combined labels.

    Parameters
    ----------
    Gm : nx.DiGraph
        Directed graph with node label annotations.

    Returns
    -------
    nx.DiGraph
        Input graph with updated edge label attributes.
    """
    for u, v in Gm.edges:
        from_label = Gm.nodes[u].get("label", "")
        to_label = Gm.nodes[v].get("label", "")
        Gm.edges[u, v]["from_label"] = from_label
        Gm.edges[u, v]["to_label"] = to_label
        Gm.edges[u, v]["label"] = f"{from_label}->{to_label}"
    return Gm


def transfer_links(Gm: nx.DiGraph) -> nx.DiGraph:
    """
    Transfer node linkage information onto graph edges.

    Parameters
    ----------
    Gm : nx.DiGraph
        Directed graph with node linkage annotations.

    Returns
    -------
    nx.DiGraph
        Input graph with updated edge linkage attributes.
    """
    for u, v in Gm.edges:
        Gm.edges[u, v]["from_node"] = Gm.nodes[u].get("node", None)
        Gm.edges[u, v]["to_node"] = Gm.nodes[v].get("node", None)
    return Gm


def get_mut_graph(gtree: nx.DiGraph) -> nx.DiGraph:
    """
    Construct a contracted mutation graph from an annotated tree.

    Parameters
    ----------
    gtree : nx.DiGraph
        Directed tree graph with "last_mut", "site", and node name attributes.

    Returns
    -------
    nx.DiGraph
        Directed mutation graph indexed from the root in deterministic order.
    """
    roots = [n for n, a in gtree.nodes(data=True) if a.get("root", False)]
    gtree_root = roots[0]
    root_label = _norm_label(gtree.nodes[gtree_root].get("last_mut", ""))


    mut_nodes_df = pd.DataFrame([
        {"name": a.get("name", None), "site": a.get("site", None)}
        for _, a in gtree.nodes(data=True)
        if a.get("site", None) is not None])
    
    if not mut_nodes_df.empty:
        mut_nodes_df = mut_nodes_df.drop_duplicates(subset=["name", "site"], keep="first")
    else:
        mut_nodes_df = pd.DataFrame(columns=["name", "site"])

    # Contract by last_mut label
    label_to_vid: Dict[str, int] = {}
    next_vid = 0

    def _vid(lbl: Optional[str]) -> int:
        nonlocal next_vid
        lbl = _norm_label(lbl)
        if lbl not in label_to_vid:
            label_to_vid[lbl] = next_vid
            next_vid += 1
        return label_to_vid[lbl]

    Gm = nx.DiGraph()

    # Ensure all contracted groups exist
    for n, a in gtree.nodes(data=True):
        lbl = _norm_label(a.get("last_mut", ""))
        vid = _vid(lbl)
        if vid not in Gm:
            Gm.add_node(vid, label=lbl, GT="", clone=None, node=None)

    # Add edges between contracted groups
    for u, v in gtree.edges:
        lu = _norm_label(gtree.nodes[u].get("last_mut", ""))
        lv = _norm_label(gtree.nodes[v].get("last_mut", ""))
        vu = _vid(lu)
        vv = _vid(lv)
        if vu != vv:
            Gm.add_edge(vu, vv)

    root_vid = _vid(root_label)
    Gm = _reindex_graph_from_root(Gm, root_vid)

    # Map label -> one original phylogeny node name
    label_to_node: Dict[str, str] = {}
    if not mut_nodes_df.empty:
        for _, row in mut_nodes_df.iterrows():
            site = _norm_label(row["site"])
            name = row["name"]
            if site != "" and site not in label_to_node:
                label_to_node[site] = name

    for vid, a in Gm.nodes(data=True):
        lbl = _norm_label(a.get("label", ""))
        Gm.nodes[vid]["node"] = label_to_node.get(lbl, None)

    Gm = label_edges(Gm)
    Gm = transfer_links(Gm)
    return Gm



def label_genotype(Gm: nx.DiGraph, root: Optional[int] = None) -> nx.DiGraph:
    """
    Annotate a mutation graph with genotype strings and clone identifiers.

    Parameters
    ----------
    Gm : nx.DiGraph
        Directed mutation graph with node label attributes.
    root : int or None, default=None
        Root node identifier. If None, the graph root is inferred.

    Returns
    -------
    nx.DiGraph
        Input graph with updated "GT" and "clone" node attributes.
    """
    if root is None:
        root = _graph_root(Gm)
    
    # unique root->v path in rooted mutation graph
    for v in Gm.nodes:
        path = nx.shortest_path(Gm, source=root, target=v)
        labels = [_norm_label(Gm.nodes[u].get("label", "")) for u in path]
        labels = [x for x in labels if x != ""]
        Gm.nodes[v]["GT"] = ",".join(labels)

    dfs_order = list(nx.dfs_preorder_nodes(Gm, source=root))
    for i, v in enumerate(dfs_order):
        Gm.nodes[v]["clone"] = i

    return Gm


def _normalize_gt(gt: Any) -> str:
    """
    Normalize a genotype value to a non-null string.

    Parameters
    ----------
    gt : Any
        Input genotype value.

    Returns
    -------
    str
        Normalized genotype string, with null-like values mapped to "".
    """
    if gt is None or (isinstance(gt, float) and np.isnan(gt)):
        return ""
    s = str(gt).strip()
    return "" if s.lower() == "nan" else s


def _build_canonical_gt_clone_map(gt_series: pd.Series, clone_series: pd.Series) -> Dict[str, int]:
    """
    Build a canonical one-to-one mapping from genotype strings to clone ids.

    Parameters
    ----------
    gt_series : pd.Series
        Genotype values.
    clone_series : pd.Series
        Clone identifiers.

    Returns
    -------
    dict of str to int
        Mapping from normalized genotype strings to clone ids.
    """
    tmp = pd.DataFrame({"GT": gt_series.map(_normalize_gt), "clone": clone_series})
    tmp = tmp.drop_duplicates()

    out: Dict[str, int] = {"": 0}
    used: set[int] = {0}

    for gt in sorted([x for x in tmp["GT"].unique().tolist() if x != ""]):
        cand = (
            tmp.loc[(tmp["GT"] == gt) & tmp["clone"].notna(), "clone"]
            .astype(int)
            .tolist()
        )
        cand = [c for c in cand if c > 0]
        chosen = min(cand) if cand else None
        if chosen in used:
            chosen = None
        if chosen is None:
            chosen = 1
            while chosen in used:
                chosen += 1
        out[gt] = int(chosen)
        used.add(int(chosen))

    return out


def _merge_two_vertices(
    Gm: nx.DiGraph,
    keep: int,
    drop: int,
    node_tar: Optional[str] = None,
    ) -> nx.DiGraph:
    """
    Merge two nodes in a mutation graph and reindex from the root.

    Parameters
    ----------
    Gm : nx.DiGraph
        Directed mutation graph.
    keep : int
        Node identifier to retain.
    drop : int
        Node identifier to remove.
    node_tar : str or None, default=None
        Replacement "node" attribute for the retained vertex.

    Returns
    -------
    nx.DiGraph
        Updated graph with merged nodes and refreshed edge annotations.
    """
    keep_label = "" if Gm.nodes[keep].get("label", None) is None else str(Gm.nodes[keep]["label"])
    drop_label = "" if Gm.nodes[drop].get("label", None) is None else str(Gm.nodes[drop]["label"])

    combined_label = ",".join(sorted([keep_label, drop_label]))
    Gm.nodes[keep]["label"] = combined_label

    if node_tar is not None:
        Gm.nodes[keep]["node"] = node_tar

    # redirect incoming
    for u in list(Gm.predecessors(drop)):
        if u != keep:
            Gm.add_edge(u, keep)

    # redirect outgoing
    for v in list(Gm.successors(drop)):
        if v != keep:
            Gm.add_edge(keep, v)

    if drop in Gm:
        Gm.remove_node(drop)

    # remove self-loops
    Gm.remove_edges_from(list(nx.selfloop_edges(Gm)))

    root = _graph_root(Gm)
    Gm = _reindex_graph_from_root(Gm, root)
    Gm = label_edges(Gm)
    Gm = transfer_links(Gm)
    return Gm


def get_move_cost(muts: str, node_ori: str, node_tar: str, l_df: pd.DataFrame) -> float:
    """
    Compute the score difference for moving mutations between two nodes.

    Parameters
    ----------
    muts : str
        Mutation label or comma-separated mutation labels.
    node_ori : str
        Source node label in "l_df".
    node_tar : str
        Target node label in "l_df".
    l_df : pd.DataFrame
        Score matrix with node labels as rows and mutation labels as columns.

    Returns
    -------
    float
        Sum of score differences between source and target nodes for the
        selected mutations. Returns "inf" when the move is not defined.
    """
    if muts is None:
        return float("inf")
    muts = str(muts)
    if muts == "":
        return float("inf")

    # splits only if comma exists
    ms = muts.split(",") if "," in muts else [muts]

    if node_ori is None or node_tar is None:
        return float("inf")
    node_ori = str(node_ori)
    node_tar = str(node_tar)

    if node_ori not in l_df.index or node_tar not in l_df.index:
        return float("inf")

    # keep current safe behavior for absent columns
    ms = [m for m in ms if m in l_df.columns]
    if len(ms) == 0:
        return float("inf")

    return float((l_df.loc[node_ori, ms] - l_df.loc[node_tar, ms]).sum())


def get_move_opt(Gm: nx.DiGraph, l_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Find the lowest-cost mutation move across graph edges.

    Parameters
    ----------
    Gm : nx.DiGraph
        Directed mutation graph with node label and linkage annotations.
    l_df : pd.DataFrame
        Score matrix with node labels as rows and mutation labels as columns.

    Returns
    -------
    dict[str, Any]
        Dictionary describing the best move and its cost.
    """
    best = {"cost": float("inf")}

    for u, v in Gm.edges:
        from_label = Gm.nodes[u].get("label", "")
        to_label = Gm.nodes[v].get("label", "")
        from_node = Gm.nodes[u].get("node", None)
        to_node = Gm.nodes[v].get("node", None)

        n_sibling = Gm.out_degree(u)

        up_cost = get_move_cost(to_label, to_node, from_node, l_df)
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
    Simplify a mutation history graph by iteratively merging low-cost edges.

    Parameters
    ----------
    Gm : nx.DiGraph
        Directed mutation graph.
    l_df : pd.DataFrame
        Score matrix used to evaluate mutation moves.
    max_cost : float, default=150.0
        Maximum allowed cost for a merge operation.
    n_cut : int, default=0
        Minimum number of edges to retain.
    verbose : bool, default=True
        Whether to log applied merge operations.

    Returns
    -------
    nx.DiGraph
        Simplified mutation graph.
    """
    if n_cut > 0:
        max_cost = float("inf")

    for _ in range(Gm.number_of_edges()):
        move = get_move_opt(Gm, l_df)

        if not np.isfinite(move["cost"]):
            break

        if (move["cost"] < max_cost) and (Gm.number_of_edges() > n_cut):
            u = move["from_id"]
            v = move["to_id"]

            if move["direction"] == "up":
                Gm = _merge_two_vertices(
                    Gm,
                    keep=u,
                    drop=v,
                    node_tar=move.get("from_node", None),
                )
                if verbose:
                    log.info(f"opt_move:{move['to_label']}->{move['from_label']}, cost={move['cost']:.3g}")
            else:
                Gm = _merge_two_vertices(
                    Gm,
                    keep=v,
                    drop=u,
                    node_tar=move.get("to_node", None),
                )
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
    Compute tree posterior outputs from a phylogenetic tree and probability matrix.

    Parameters
    ----------
    tree : TreeNode
        Input phylogenetic tree.
    P_df : pd.DataFrame
        Probability matrix with tips as rows and sites as columns.
    clip_eps : float, default=1e-10
        Lower bound used for numerical stability during scoring.

    Returns
    -------
    TreePost
        Tree posterior object containing the annotated tree and score matrix.
    """
    tree_stats = score_tree_treenode_fast(tree, P_df, get_l_matrix=True, clip_eps=clip_eps)
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
    Build and annotate a genotype tree from a phylogenetic tree.

    Parameters
    ----------
    tree : TreeNode
        Input phylogenetic tree.
    P_df : pd.DataFrame
        Probability matrix with tips as rows and sites as columns.
    n_cut : int, default=0
        Minimum number of edges to retain during history simplification.
    max_cost : float, default=0.0
        Maximum allowed merge cost during history simplification.
    clip_eps : float, default=1e-10
        Lower bound used for numerical stability during scoring.
    verbose : bool, default=True
        Whether to log simplification operations.

    Returns
    -------
    nx.DiGraph
        Annotated genotype tree.
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
    mut_nodes = pd.DataFrame(vertices, columns=["name", "site", "clone", "GT"])

    # Keep only rows that can be transferred back onto gtree by name.
    if not mut_nodes.empty:
        mut_nodes = mut_nodes.loc[mut_nodes["name"].notna()].copy()

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
    Compute per-cell posterior clone probabilities from expression and allele evidence.

    Parameters
    ----------
    gtree : nx.DiGraph
        Annotated genotype tree with node genotype, clone, and compartment attributes.
    exp_post : pd.DataFrame
        Expression-based segment posterior table.
    allele_post : pd.DataFrame
        Allele-based segment posterior table.
    seg_col : str, default="seg"
        Segment identifier column.
    cell_col : str, default="cell"
        Cell identifier column.
    cnv_state_col : str, default="cnv_state"
        CNV state column used to exclude neutral segments.
    Z_cnv_col : str, default="Z_cnv"
        Log-score column for the CNV model.
    Z_n_col : str, default="Z_n"
        Log-score column for the neutral model.

    Returns
    -------
    pd.DataFrame
        Per-cell clone posterior summary including the most likely clone and
        genotype, clone posterior probabilities, tumor posterior mass, and
        inferred compartment.
    """
    
    # clones table from gtree nodes; canonicalize GT<->clone mapping first.
    nodes_df = pd.DataFrame([dict(GT=_normalize_gt(a.get("GT", "")),
                                  clone=a.get("clone", np.nan),
                                  compartment=a.get("compartment", np.nan),
                                  leaf=a.get("leaf", False))
                             for _, a in gtree.nodes(data=True)])

    gt_to_clone = _build_canonical_gt_clone_map(nodes_df["GT"], nodes_df["clone"])
    nodes_df["clone"] = nodes_df["GT"].map(gt_to_clone).astype(int)

    clones = (nodes_df.groupby(["GT", "clone", "compartment"], dropna=False, as_index=False)
              .agg(clone_size=("leaf", "sum")))

    # ensure normal genotype exists exactly once
    if "" not in clones["GT"].tolist():
        clones = pd.concat(
            [pd.DataFrame([dict(GT="", clone=0, compartment="normal", clone_size=0)]), clones],
            ignore_index=True,
            )

    # prior_clone:
    unique_gt = clones["GT"].astype(str).unique().tolist()
    n_tumor_gt = sum(g != "" for g in unique_gt)

    def _prior(gt: str) -> float:
        gt = "" if pd.isna(gt) else str(gt)
        return 0.5 if gt == "" else 0.5 / n_tumor_gt

    clones["prior_clone"] = clones["GT"].map(_prior)

    # clone_segs
    seg_universe = sorted({
        s
        for gt in clones["GT"].astype(str).tolist()
        for s in _split_muts(gt)
        if s != ""
    })

    base = clones[["GT", "clone", "compartment", "prior_clone", "clone_size"]].drop_duplicates().copy()

    if len(seg_universe) > 0:
        base["_tmp"] = 1
        seg_df = pd.DataFrame({seg_col: seg_universe})
        seg_df["_tmp"] = 1

        clone_segs = seg_df.merge(base, on="_tmp", how="inner").drop(columns="_tmp")

        gt_to_set = {gt: set(_split_muts(gt))
                     for gt in base["GT"].astype(str).unique().tolist()}

        clone_segs["I"] = [1 if seg in gt_to_set.get(gt, set()) else 0
                           for seg, gt in zip(
                                   clone_segs[seg_col].astype(str).tolist(),
                                   clone_segs["GT"].astype(str).tolist())]
        clone_segs["I"] = clone_segs["I"].astype(int)
        
    else:
        clone_segs = pd.DataFrame(columns=[seg_col,
                                           "GT",
                                           "clone",
                                           "compartment",
                                           "prior_clone",
                                           "clone_size",
                                           "I"])

    def _block(post: pd.DataFrame, suffix: str) -> pd.DataFrame:
        post = post.copy()
        post = post.loc[post[cnv_state_col] != "neu"]
        post = post.merge(clone_segs, on=seg_col, how="inner")
        post["l_clone"] = np.where(
            post["I"].to_numpy() == 1,
            post[Z_cnv_col].to_numpy(),
            post[Z_n_col].to_numpy(),
        )
        out = (post.groupby([cell_col, "clone", "GT", "prior_clone"],
                            as_index=False, 
                            dropna=False)  # TODO: just added dropna=False 07/03/2026
               .agg(**{f"l_clone_{suffix}": ("l_clone", "sum")}))
        return out

    x = _block(exp_post, "x")
    y = _block(allele_post, "y")

    merged = x.merge(y, on=[cell_col, "clone", "GT", "prior_clone"], how="outer")
    merged["l_clone_x"] = merged["l_clone_x"].fillna(0.0)
    merged["l_clone_y"] = merged["l_clone_y"].fillna(0.0)

    if merged.shape[0] == 0:
        clone_post = pd.DataFrame(columns=[cell_col, "clone_opt", "GT_opt", "p_opt", "p_cnv", "p_cnv_x", "p_cnv_y", "compartment_opt"])
        return clone_post

    merged["Z_clone"] = (
        np.log(merged["prior_clone"].to_numpy())
        + merged["l_clone_x"].to_numpy()
        + merged["l_clone_y"].to_numpy()
    )
    merged["Z_clone_x"] = np.log(merged["prior_clone"].to_numpy()) + merged["l_clone_x"].to_numpy()
    merged["Z_clone_y"] = np.log(merged["prior_clone"].to_numpy()) + merged["l_clone_y"].to_numpy()

    merged["p"] = np.nan
    merged["p_x"] = np.nan
    merged["p_y"] = np.nan

    for _, idx in merged.groupby(cell_col, sort=False).groups.items():
        z = merged.loc[idx, "Z_clone"].to_numpy()
        zx = merged.loc[idx, "Z_clone_x"].to_numpy()
        zy = merged.loc[idx, "Z_clone_y"].to_numpy()

        merged.loc[idx, "p"] = np.exp(z - _log_sum_exp(z))
        merged.loc[idx, "p_x"] = np.exp(zx - _log_sum_exp(zx))
        merged.loc[idx, "p_y"] = np.exp(zy - _log_sum_exp(zy))

    def _opt_block(df: pd.DataFrame) -> pd.Series:
        i = int(df["p"].to_numpy().argmax())
        clone_val = df["clone"].to_numpy()[i]
        return pd.Series({
            "clone_opt": int(clone_val) if pd.notna(clone_val) else np.nan,
            "GT_opt": df["GT"].to_numpy()[i],
            "p_opt": float(df["p"].to_numpy()[i]),
        })

    opt = merged.groupby(cell_col, as_index=False).apply(_opt_block, include_groups=False).reset_index(drop=True)
    merged2 = merged.merge(opt, on=cell_col, how="left")

    piv_p = merged2.pivot(index=[cell_col, "clone_opt", "GT_opt", "p_opt"], columns="clone", values="p")
    piv_px = merged2.pivot(index=[cell_col, "clone_opt", "GT_opt", "p_opt"], columns="clone", values="p_x")
    piv_py = merged2.pivot(index=[cell_col, "clone_opt", "GT_opt", "p_opt"], columns="clone", values="p_y")

    piv_p.columns = [f"p_{int(c)}" for c in piv_p.columns]
    piv_px.columns = [f"p_x_{int(c)}" for c in piv_px.columns]
    piv_py.columns = [f"p_y_{int(c)}" for c in piv_py.columns]

    clone_post = pd.concat([piv_p, piv_px, piv_py], axis=1).reset_index()

    tumor_clones = (
        clones.loc[clones["compartment"].astype(str) == "tumor", "clone"]
        .dropna()
        .astype(int)
        .tolist()
    )

    def _row_sum_cols(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
        cols = [c for c in cols if c in df.columns]
        if not cols:
            return np.zeros(len(df), dtype=float)
        return df[cols].to_numpy(dtype=float).sum(axis=1)

    clone_post["p_cnv"] = _row_sum_cols(clone_post, [f"p_{c}" for c in tumor_clones])
    clone_post["p_cnv_x"] = _row_sum_cols(clone_post, [f"p_x_{c}" for c in tumor_clones])
    clone_post["p_cnv_y"] = _row_sum_cols(clone_post, [f"p_y_{c}" for c in tumor_clones])

    clone_post["compartment_opt"] = np.where(clone_post["p_cnv"].to_numpy() > 0.5, "tumor", "normal")

    return clone_post
