#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 16 12:06:28 2025

@author: carlino.calogero
"""

import math
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd

import skbio
from skbio import DistanceMatrix as SKDM
# from skbio.tree import nj
from skbio.tree import TreeNode

from Bio import Phylo
from Bio.Phylo.BaseTree import Clade, Tree
from io import StringIO
from scipy.spatial.distance import pdist, squareform          # faster than sklearn 

import tqdm


def biophylo_to_skbio(tree_bp: Phylo.BaseTree.Tree) -> TreeNode:
    buf = StringIO()
    Phylo.write(tree_bp, buf, "newick")
    return TreeNode.read([buf.getvalue()])

def skbio_to_biophylo(tnode: TreeNode) -> Phylo.BaseTree.Tree:
    buf = StringIO()
    tnode.write(buf, format="newick")         # to_newick()
    newick = buf.getvalue()    
    return Phylo.read(StringIO(newick), "newick")


# tree builders
def build_upgma_tree(dist_mat: SKDM, labels: list[str]) -> Phylo.BaseTree.Tree:
    """
    Build a UPGMA tree with scikit‑bio,
    then return it as a Biopython Tree and in post‑order.
    """
    # run UPGMA
    tnode = skbio.tree.upgma(dist_mat)
    # convert to Newick
    tree = skbio_to_biophylo(tnode)
    return tree #reorder_postorder(tree)

def build_nj_tree(dist_mat: SKDM, labels: list[str]) -> Phylo.BaseTree.Tree:
    """
    Build a Neighbor‑Joining tree with scikit‑bio,
    then return it as a Biopython Tree and in post‑order.
    """
    # run NJ
    tnode = skbio.tree.nj(dist_mat)
    # convert to Newick
    tree = skbio_to_biophylo(tnode)
    return tree #reorder_postorder(tree)

def root_and_prune(tree: Phylo.BaseTree.Tree, outgroup_name: str = "outgroup") -> Phylo.BaseTree.Tree:
    """
    Root the tree on `outgroup` and then delete that tip.
    """
    tree.root_with_outgroup(outgroup_name)
    tree.prune(target=outgroup_name)
    return tree #reorder_postorder(tree)

def all_children(tree: Phylo.BaseTree.Tree, label_order: list[str]) -> tuple[dict[int, list[int]], dict[Clade, int]]:
    """Return dict(parent_id -> [child_id1, child_id2]) with tip IDs respecting P_df row order."""
    # Map leaves strictly via label list
    leaves = [tree.find_any(name=lab) for lab in label_order]
    internals = tree.get_nonterminals(order='postorder')
    id_map = {cl: i for i, cl in enumerate(leaves + internals, start=0)} # was start=1
    children = {id_map[p]: [id_map[c] for c in p.clades] for p in internals}
    return children, id_map

def cgetQ(logQ: np.ndarray, children_dict: dict[int, list[int]], node_order: list[int]) -> np.ndarray:
    """Propagate log‑likelihoods upward."""
    for node in node_order:
        ch = children_dict[node]
        logQ[node] = logQ[ch[0]] + logQ[ch[1]]
    return logQ

def score_tree(tree: Phylo.BaseTree.Tree, P_df: pd.DataFrame, get_l_matrix: bool = False) -> dict:
    """Score tree"""
    #tree = reorder_postorder(tree)
    children_dict, id_map = all_children(tree, list(P_df.index))
    
    P     = P_df.values
    eps   = 1e-10
    logP1 = np.log(np.clip(P,  eps, 1-eps))
    logP0 = np.log(np.clip(1-P, eps, 1-eps))

    n, m  = P.shape
    n_int = len(children_dict)    # number of internal nodes
    logQ  = np.zeros((n + n_int, m))
    
    # log‑odds
    for clade, idx in id_map.items():
        if clade.is_terminal():
            logQ[idx] = logP1[idx] - logP0[idx]
    
    # internal nodes in post‑order (Biopython)
    node_order = [id_map[cl] for cl in tree.get_nonterminals(order='postorder')]
    logQ = cgetQ(logQ, children_dict, node_order)
    
    if get_l_matrix:
        l_matrix = logQ + logP0.sum(axis=0)
        l_tree   = np.sum(l_matrix.max(axis=0))
    else:
        l_matrix = None
        l_tree   = logQ.max(axis=0).sum() + logP0.sum()
    
    return {"l_tree": l_tree, "logQ": logQ, "l_matrix": l_matrix}

def tree_score_wrapper(tree_bp: Phylo.BaseTree.Tree, P_df: pd.DataFrame) -> tuple[float, TreeNode]:
    tnode = biophylo_to_skbio(tree_bp)
    llik = score_tree(tree_bp, P_df)
    return llik, tnode

def perform_nni_ml(tree_init, dm_skbio, P_df, ncores=1, eps=1e-6, max_iter=20):
    tree_cur = biophylo_to_skbio(tree_init).copy()
    best_ll  = score_tree(tree_init, P_df)['l_tree']
    history  = [tree_init]

    for _ in tqdm.tqdm(range(max_iter)):
        # one NNI sweep
        tree_next = skbio.tree.nni(tree_cur, dm_skbio) # raises if taxa mismatch

        # evaluate
        ll_new, _ = tree_score_wrapper(skbio_to_biophylo(tree_next), P_df)
        ll_new = ll_new['l_tree']

        if ll_new - best_ll > eps:
            best_ll = ll_new
            tree_cur = tree_next
            history.append(skbio_to_biophylo(tree_cur))
        else:
            break  # converged

    return history


# utilities
def _root_of_digraph(G: nx.DiGraph) -> int:
    """
    Return the unique root (node with in-degree 0)
    """
    roots = [n for n in G.nodes if G.in_degree(n) == 0]
    if not roots:
        raise ValueError("No root (in-degree 0) found.")
    if len(roots) > 1:
        # pick the first deterministically
        roots.sort()
    return roots[0]

def _bfs_rank(G: nx.DiGraph, root: int) -> Dict[int, int]:
    """
    BFS rank (distance in edges from root). Root has 0; unreachable nodes absent.
    """
    return dict(nx.single_source_shortest_path_length(G, root))

def _mut_burden(gt: str) -> int:
    """
    Count of mutations encoded in a comma-separated GT string.
    """
    if not gt:
        return 0
    # e.g., "a,b,c" -> 3 ; "x" -> 1
    return gt.count(",") + 1


# Phylogeny reconstruction
def mark_tumor_lineage(gtree: nx.DiGraph) -> nx.DiGraph:
    """
    pick the mutation-carrying node whose descendant leaves have 
    the largest total mutation burden (by GT), mark that as tumor root,
    set node/edge 'compartment' ('tumor' vs 'normal').
    """
    # nodes carrying any local mutation(s)
    mut_nodes = [n for n, d in gtree.nodes(data=True) if d.get("site") not in (None, "")]
    if not mut_nodes:
        # no mutations: set everything to normal
        root = _root_of_digraph(gtree)
        ranks = _bfs_rank(gtree, root)
        for n in gtree.nodes:
            seq = ranks.get(n, -1)
            gtree.nodes[n]["seq"] = seq
            gtree.nodes[n]["compartment"] = "normal"
            gtree.nodes[n]["is_tumor_root"] = (n == root)
        for u, v in gtree.edges:
            gtree.edges[u, v]["compartment"] = gtree.nodes[v]["compartment"]
        return gtree

    # compute total descendant leaf mutation burden for each candidate root
    mut_burdens = []
    for node in mut_nodes:
        ranks = _bfs_rank(gtree, node)
        total = 0
        for n, d in gtree.nodes(data=True):
            if d.get("leaf") and ranks.get(n, 0) > 0:
                total += _mut_burden(d.get("GT", ""))
        mut_burdens.append(total)

    # choose tumor root (tie -> earliest in mut_nodes order)
    tumor_root = mut_nodes[int(np.argmax(mut_burdens))]

    # annotate nodes
    ranks = _bfs_rank(gtree, tumor_root)
    for n in gtree.nodes:
        seq = ranks.get(n, -1)
        gtree.nodes[n]["seq"] = seq
        gtree.nodes[n]["compartment"] = "tumor" if seq > 0 else "normal"
        gtree.nodes[n]["is_tumor_root"] = (n == tumor_root)

    # annotate edges with downstream node compartment
    for u, v in gtree.edges:
        gtree.edges[u, v]["compartment"] = gtree.nodes[v]["compartment"]

    return gtree


def label_genotype(G: nx.DiGraph) -> nx.DiGraph:
    """
    - node 'label' is the mutation label carried at that node ('' allowed)
    - compute node 'GT' = concatenation of labels along root->node path
    - assign 'clone' as DFS preorder visit order (0-based)
    """
    # id_to_label
    id_to_label = {n: G.nodes[n].get("label", "") for n in G.nodes}
    root = _root_of_digraph(G)

    # compute GT by shortest path from root (tree/DAG assumed)
    GT = {}
    for v in G.nodes:
        path = nx.shortest_path(G, source=root, target=v)
        muts = [id_to_label[x] for x in path if id_to_label.get(x, "") != ""]
        GT[v] = ",".join(muts)

    nx.set_node_attributes(G, GT, "GT")

    # clone = DFS visit order (0-based for consistency)
    order = list(nx.dfs_preorder_nodes(G, source=root))
    clone_map = {n: i for i, n in enumerate(order)}
    nx.set_node_attributes(G, clone_map, "clone")

    return G


#  label_edges on mutation graph
def label_edges(G: nx.DiGraph) -> nx.DiGraph:
    """
    Add edge attributes: 'from_label', 'to_label', and 'label' = 'from_label->to_label'.
    """
    for u, v in G.edges:
        fl = G.nodes[u].get("label", "")
        tl = G.nodes[v].get("label", "")
        G.edges[u, v]["from_label"] = fl
        G.edges[u, v]["to_label"] = tl
        G.edges[u, v]["label"] = f"{fl}->{tl}"
    return G


# transfer_links (copy node 'node' id to edges)
def transfer_links(G: nx.DiGraph) -> nx.DiGraph:
    """
    For each edge, copy the upstream/downstream node's 'node' attribute to
    edge attrs 'from_node' and 'to_node'.
    """
    for u, v in G.edges:
        G.edges[u, v]["from_node"] = G.nodes[u].get("node")
        G.edges[u, v]["to_node"] = G.nodes[v].get("node")
    return G


# contract_nodes (merge adjacent vertices)
def contract_nodes(G: nx.DiGraph,
                   vset: List[str],
                   node_tar: Optional[str] = None,
                   debug: bool = False) -> nx.DiGraph:
    """
    Merge a set of adjacent vertices whose node attribute 'label' is in vset.
    - New node id is the smallest original id in the set.
    - New 'label' is comma-joined sorted labels.
    - Preserve 'node' from the first by default, or override with node_tar if given.
    - Rebuild simple DiGraph (no parallel edges, no self-loops).
    """
    # map labels -> node ids
    label_to_id = defaultdict(list)
    for n, d in G.nodes(data=True):
        lab = d.get("label", "")
        label_to_id[lab].append(n)

    # minimally enforce one id per label; 
    vset_ids = sorted([label_to_id[lab][0] for lab in vset if label_to_id.get(lab)])
    if len(vset_ids) <= 1:
        return G.copy()

    new_id = min(vset_ids)
    keep_ids = set(G.nodes) - set(vset_ids) | {new_id}

    # build mapping: contracted ids -> new_id, others stay
    map_id = {}
    for n in G.nodes:
        map_id[n] = new_id if n in vset_ids else n

    # build new graph
    H = nx.DiGraph()
    # nodes
    for n in keep_ids:
        H.add_node(n, **G.nodes[n])

    # merged node attributes
    merged_labels = sorted([G.nodes[i].get("label", "") for i in vset_ids if G.nodes[i].get("label", "") != ""])
    H.nodes[new_id]["label"] = ",".join(merged_labels)
    if node_tar is not None:
        H.nodes[new_id]["node"] = node_tar  # override

    # edges
    for u, v, ed in G.edges(data=True):
        uu, vv = map_id[u], map_id[v]
        if uu == vv:
            continue  # drop self-loop
        # combine; keep last attribute set is fine for our use
        H.add_edge(uu, vv)
    # refresh sequential 'id' attribute (0-based)
    for i, n in enumerate(sorted(H.nodes)):
        H.nodes[n]["id"] = i

    if debug:
        return H

    H = label_edges(H)
    return H


# move costs / optimal move
def get_move_cost(muts: str, node_ori: str, node_tar: str, l_matrix: pd.DataFrame) -> float:
    """
    Sum over mutations: l_matrix[node_ori, mut] - l_matrix[node_tar, mut].
    Empty muts, missing nodes, or missing columns => +inf (i.e., disallow move).
    """

    # No mutations to move
    if not muts:
        return math.inf

    # Nodes must be valid row labels
    if node_ori is None or node_tar is None:
        return math.inf
    if node_ori not in l_matrix.index or node_tar not in l_matrix.index:
        return math.inf

    # Keep only mutation columns that exist
    muts_list = muts.split(",") if "," in muts else [muts]
    muts_in = [m for m in muts_list if m in l_matrix.columns]
    if not muts_in:
        return math.inf

    return sum(l_matrix.loc[node_ori, muts_in] - l_matrix.loc[node_tar, muts_in])


def get_move_opt(G: nx.DiGraph, l_matrix: pd.DataFrame) -> Dict[str, object]:
    best = {"cost": math.inf}
    outdeg = dict(G.out_degree())

    for u, v in G.edges:
        ed = G.edges[u, v]
        from_label = ed.get("from_label", "")
        to_label   = ed.get("to_label", "")
        from_node  = ed.get("from_node", None)
        to_node    = ed.get("to_node", None)

        # If either node mapping is missing, this edge cannot be reassigned
        up_cost = get_move_cost(to_label, to_node, from_node, l_matrix)
        down_cost = get_move_cost(from_label, from_node, to_node, l_matrix)

        # Prevent a 'down' move if branching
        if outdeg.get(u, 0) > 1:
            down_cost = math.inf

        for direction, cost in (("up", up_cost), ("down", down_cost)):
            if cost < best.get("cost", math.inf):
                best = {
                    "from": u, "to": v,
                    "from_label": from_label, "to_label": to_label,
                    "from_node": from_node, "to_node": to_node,
                    "direction": direction, "cost": cost,
                }
    return best


# simplify_history
def simplify_history(G: nx.DiGraph,
                     l_matrix: pd.DataFrame,
                     max_cost: float = 150.0,
                     n_cut: int = 0,
                     verbose: bool = True) -> nx.DiGraph:
    """
    Iteratively contract edges by cheapest move (up/down) while cost < max_cost,
    or until ecount(G) <= n_cut.
    """
    if n_cut > 0:
        max_cost = math.inf

    # ensure required edge labels exist
    G = label_edges(G)
    G = transfer_links(G)

    # conservative bound on iterations
    for _ in range(len(G.edges)):
        move_opt = get_move_opt(G, l_matrix)
        if (move_opt["cost"] < max_cost) and (G.number_of_edges() > n_cut):
            if move_opt["direction"] == "up":
                # merge child into parent, keep parent's 'node'
                G = contract_nodes(G,
                                   vset=[move_opt["from_label"], move_opt["to_label"]],
                                   node_tar=move_opt["from_node"])
            else:
                # merge parent into child, keep child's 'node'
                G = contract_nodes(G,
                                   vset=[move_opt["from_label"], move_opt["to_label"]],
                                   node_tar=move_opt["to_node"])
            G = transfer_links(G)
            if verbose:
                if move_opt["direction"] == "up":
                    msg = f"opt_move:{move_opt['to_label']}->{move_opt['from_label']}, cost={move_opt['cost']:.3g}"
                else:
                    msg = f"opt_move:{move_opt['from_label']}->{move_opt['to_label']}, cost={move_opt['cost']:.3g}"
                print(msg)
        else:
            break
    return G


# get_ordered_tips
def get_ordered_tips(tree: "Phylo.BaseTree.Tree") -> List[str]:
    """
    Biopython: traverse edges in postorder and list terminal clades in that order.
    """
    tips = []
    for cl in tree.find_clades(order="postorder"):
        if cl.is_terminal():
            tips.append(cl.name)
    return tips


# get_mut_graph
def get_mut_graph(gtree: nx.DiGraph) -> nx.DiGraph:
    """
    Convert a single-cell phylogeny (with per-node last_mut, site) into a mutation graph:
    - contract nodes by identical 'last_mut'
    - node attrs: 'label' = last_mut, 'id' (0-based), 'node' linking back to tree node name (where site appears)
    - edges induced by parent-child relations between contracted groups
    """
    # distinct mutation-carrying nodes (name, site)
    mut_nodes_rows = []
    for n, d in gtree.nodes(data=True):
        site = d.get("site")
        if site not in (None, ""):
            mut_nodes_rows.append({"name": d.get("name"), "site": site})
    mut_nodes_df = pd.DataFrame(mut_nodes_rows).drop_duplicates()

    # group original nodes by last_mut
    last_groups: Dict[str, List[int]] = defaultdict(list)
    for n, d in gtree.nodes(data=True):
        last = d.get("last_mut", "")
        last_groups[last].append(n)

    # create mutation graph nodes in sorted order of label
    labels = sorted(last_groups.keys())
    label_to_id = {lab: i for i, lab in enumerate(labels)}
    G = nx.DiGraph()
    for lab, nid in label_to_id.items():
        G.add_node(nid, label=lab, id=nid)

    # edges between contracted groups
    for u, v in gtree.edges:
        lu = gtree.nodes[u].get("last_mut", "")
        lv = gtree.nodes[v].get("last_mut", "")
        if lu == lv:
            continue
        G.add_edge(label_to_id[lu], label_to_id[lv])

    # label edges
    G = label_edges(G)

    # attach back-pointer 'node' using site==label
    site_to_node = dict(mut_nodes_df[["site", "name"]].itertuples(index=False, name=None))
    for n in G.nodes:
        lab = G.nodes[n].get("label", "")
        if lab in site_to_node:
            G.nodes[n]["node"] = site_to_node[lab]
        else:
            G.nodes[n]["node"] = None

    G = transfer_links(G)
    return G


# get_tree_post and get_gtree (wire-up)
def _clade_name_map(tree: Phylo.BaseTree.Tree) -> Dict[Clade, str]:
    """Tips: original names; internals: Node0..Node{n_int-1} in postorder."""
    name_map: Dict[Clade, str] = {}
    for cl in tree.get_terminals():
        name_map[cl] = cl.name
    for i, cl in enumerate(tree.get_nonterminals(order='postorder')):
        name_map[cl] = f"Node{i}"
    return name_map


def phylo_to_graph(tree: Phylo.BaseTree.Tree,
                   name_map: Dict[Clade, str],
                   traversal_order: str = "postorder",
                   debug: bool = False) -> nx.DiGraph:
    """Convert tree -> DiGraph with node attrs and deterministic names."""
    G = nx.DiGraph()
    clades = list(tree.find_clades(order=traversal_order))
    id_map = {cl: i for i, cl in enumerate(clades)}
    root_id = id_map[tree.root]

    for clade in clades:
        idx = id_map[clade]
        G.add_node(idx, name=name_map[clade])
        for child in clade.clades:
            bl = child.branch_length if child.branch_length is not None else 0.0
            G.add_edge(idx, id_map[child], weight=bl)

    for node in G.nodes:
        G.nodes[node]["leaf"] = (G.out_degree(node) == 0)
        G.nodes[node]["root"] = (node == root_id)

    depth = nx.single_source_shortest_path_length(G, root_id)
    for node, d in depth.items():
        G.nodes[node]["depth"] = d
        G.nodes[node]["id"] = node

    for u, v in G.edges:
        G.edges[u, v]["leaf"] = G.nodes[v]["leaf"]

    if debug:
        nodes_df = pd.DataFrame.from_dict({n: d for n, d in G.nodes(data=True)}, orient="index")
        edges_df = nx.to_pandas_edgelist(G)
        return G, nodes_df, edges_df
    return G


def mut_to_tree(G: nx.DiGraph, mut_df: pd.DataFrame) -> nx.DiGraph:
    """

    Parameters
    ----------
    G      : nx.DiGraph
             Tree graph with node attrs 'name', 'leaf', 'root', 'depth', 'id'
    mut_df : pd.DataFrame
             Must have columns:
               - 'name'  : matches G.nodes[n]['name']
               - 'site'  : comma-separated mutations at that node
             Optionally:
               - 'clone' : clone label per genotype

    Returns
    -------
    G      : nx.DiGraph (modified in place)
             - Nodes get new attrs: 'site', 'n_mut', 'GT', 'last_mut', (optional) 'clone'
             - Edges get new attr: 'length' (and synced to 'weight')
    """
    
    # Ensure n_mut column
    if "n_mut" not in mut_df:
        mut_df["n_mut"] = mut_df["site"].str.split(",").apply(len)
    
    # Build quick lookups
    node_info = mut_df.set_index("name")[["site", "n_mut"]].to_dict("index")
    
    # Attach site and n_mut to each node
    for n, data in G.nodes(data=True):
        name = data.get("name", "")
        info = node_info.get(name, {"site": "", "n_mut": 0})
        data["site"]  = info["site"]
        data["n_mut"] = info["n_mut"]
    
    # Convert n_mut to edge length, enforce min tip‐length 0.2
    for u, v, ed in G.edges(data=True):
        ln = G.nodes[v]["n_mut"]
        if G.nodes[v]["leaf"]:
            ln = max(ln, 0.2)
        ed["length"] = ln
        # ed["weight"] = ln
    
    # Build a lookup: id -> local mutation string
    node_to_mut = {n: (d["site"] or "") for n, d in G.nodes(data=True)}
    
    # Locate the unique root (flagged earlier in phylo_to_graph)
    root = next(n for n, d in G.nodes(data=True) if d.get("root"))
    
    # Traverse the tree breadth-first and accumulate mutations
    # keep a running list of mutations for every node seen so far.
    G.nodes[root]["GT"] = node_to_mut[root]
    G.nodes[root]["last_mut"] = (node_to_mut[root].split(",")[-1] if node_to_mut[root] else "")
    
    queue = deque([root])
    while queue:
        parent = queue.popleft()
        # split the parent genotype into a Python list ('' -> [])
        parent_gt = (G.nodes[parent]["GT"].split(",") if G.nodes[parent]["GT"] else [])
    
        # visit all children (successors) in breadth-first order
        for child in G.successors(parent):
            queue.append(child)
    
            child_local = (node_to_mut[child].split(",") if node_to_mut[child] else [])
    
            # cumulative genotype along the path
            cum_gt = parent_gt + child_local
            G.nodes[child]["GT"] = ",".join(cum_gt)
    
            # last mutation on that path
            G.nodes[child]["last_mut"] = cum_gt[-1] if cum_gt else ""
    
    # if GT is empty but node carries local mutations, copy them
    for n, d in G.nodes(data=True):
        if d["GT"] == "" and d["site"]:
            d["GT"] = d["site"]

    # propagate clone IDs by GT
    if isinstance(mut_df, pd.DataFrame) and {"GT", "clone"}.issubset(mut_df.columns):
        # prefer one clone per GT (drop duplicates), cast to int where possible
        gt_clone = (
            mut_df[["GT", "clone"]]
            .dropna()
            .drop_duplicates(subset=["GT"])
        )
        gt_to_clone = dict(zip(gt_clone["GT"], gt_clone["clone"]))
        for n, d in G.nodes(data=True):
            gt = d.get("GT", "")
            if gt in gt_to_clone:
                try:
                    d["clone"] = int(gt_to_clone[gt])
                except Exception:
                    d["clone"] = gt_to_clone[gt]

    return G


def annotate_tree(tree: Phylo.BaseTree.Tree, P: pd.DataFrame) -> nx.DiGraph:
    sites = list(P.columns)

    # likelihood matrix (node × site)
    tree_stats = score_tree(tree, P, get_l_matrix=True)
    l_matrix = pd.DataFrame(tree_stats["l_matrix"], columns=sites)

    tips = [cl.name for cl in tree.get_terminals()]
    n_int = len(list(tree.get_nonterminals()))
    internals = [f"Node{i}" for i in range(n_int)]           # 0-based
    l_matrix.index = tips + internals

    # mutation assignment (site -> argmax row)
    mut_nodes = []
    for site in sites:
        row = l_matrix[site].idxmax()
        mut_nodes.append({"site": site, "name": row, "l": l_matrix.loc[row, site]})
    mut_df = (pd.DataFrame(mut_nodes)
                .groupby("name", sort=False)
                .agg(site=("site", lambda x: ",".join(sorted(x))),
                     n_mut=("site", "count"),
                     l=("l", "sum"))
                .reset_index())

    # build graph with consistent names
    name_map = _clade_name_map(tree)                      # tips + Node0..Node{n_int-1} in postorder
    gtree = phylo_to_graph(tree, name_map=name_map)

    # attach mutation metadata (node 'name' is the join key)
    gtree = mut_to_tree(gtree, mut_df)
    return gtree


def get_tree_post(tree, P: pd.DataFrame) -> Dict[str, object]:
    """
    compute l_matrix and a mutation-annotated tree (gtree).
    Uses 0-based internal labels (Node0..).
    """
    sites = list(P.columns)
    stats = score_tree(tree, P, get_l_matrix=True)
    l_matrix = pd.DataFrame(stats["l_matrix"], columns=sites)

    tips = [cl.name for cl in tree.get_terminals()]
    n_int = len(list(tree.get_nonterminals()))
    internals = [f"Node{i}" for i in range(n_int)]  # 0-based
    l_matrix.index = tips + internals

    gtree = annotate_tree(tree, P)
    return {"gtree": gtree, "l_matrix": l_matrix}


def get_gtree(tree, P: pd.DataFrame, n_cut: int = 0, max_cost: float = 0.0) -> nx.DiGraph:
    """
    High-level: build gtree, simplify mutation history, relabel genotypes, update original gtree,
    and mark tumor lineage.
    """
    # L-matrix + annotated phylogeny
    tree_post = get_tree_post(tree, P)

    # mutation graph -> simplify -> label genotypes
    G_m = get_mut_graph(tree_post["gtree"])
    G_m = simplify_history(G_m, tree_post["l_matrix"], max_cost=max_cost, n_cut=n_cut, verbose=True)
    G_m = label_genotype(G_m)

    # extract per-node mutation assignments from mutation graph
    mut_nodes = []
    Vdf = pd.DataFrame([{**{"id": n}, **d} for n, d in G_m.nodes(data=True)])
    mut_nodes = Vdf.rename(columns={"node": "name", "label": "site"})[["name", "site", "clone", "GT"]]

    # update original tree with he above assignments & tumor/normal labels
    gtree = mut_to_tree(tree_post["gtree"], mut_nodes)
    gtree = mark_tumor_lineage(gtree)
    return gtree





