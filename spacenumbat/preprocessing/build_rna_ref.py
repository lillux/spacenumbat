##!/usr/bin/env python3
"""Donor-balanced RNA references for SpaceNumbat (Python >=3.10).

Input: AnnData cells x genes containing raw counts, including backed .X.
Output: genes x profiles, matching the in-memory layout of data.ref_hca.

JSD ratios are empirical compression criteria, NOT equivalence-test p-values.
Defaults need calibration on held-out normal donors and CNA-positive controls.
"""

from __future__ import annotations

import argparse
import heapq
import hashlib
import itertools
import json
import logging
from pathlib import Path
import re
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import rel_entr

log = logging.getLogger(__name__)
__version__ = "0.3.0"
UPSTREAM_COMMIT = "987b112694f41fc0c2b19b1e0f492392b76bd73c"


def _label(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


# Permissions to compare, not instructions to merge. Unlisted labels remain
# separate families. In particular, neither broad_cell_class nor compartment
# alone is an adequate merge family. Compartment is a separate hard boundary.
_FAMILIES = {
    "CD4 T": [
        "cd4-positive, alpha-beta t cell", "cd4-positive, alpha-beta memory t cell",
        "cd4-positive memory t cell", "activated cd4-positive, alpha-beta t cell",
        "naive thymus-derived cd4-positive, alpha-beta t cell",
        "cd4-positive helper t cell", "effector cd4-positive, alpha-beta t cell",
        "cd4-positive, alpha-beta cytotoxic t cell", "t follicular helper cell",
    ],
    "CD8 T": [
        "cd8-positive, alpha-beta t cell", "cd8-positive, alpha-beta memory t cell",
        "activated cd8-positive, alpha-beta t cell", "naive cd8-positive t cell",
        "naive thymus-derived cd8-positive, alpha-beta t cell",
        "cd8-positive cytotoxic t cell", "effector cd8-positive, alpha-beta t cell",
        "cd8-positive, alpha-beta cytokine secreting effector t cell",
        "cd8+, alpha-beta cytokine secreting effector t cell",
    ],
    "Treg": ["regulatory t cell", "naive regulatory t cell",
             "cd4-positive, cd25-positive, alpha-beta regulatory t cell"],
    "NK": ["natural killer cell", "nk cell", "mature natural killer cell",
           "immature natural killer cell", "pre-natural killer cell",
           "proliferating nk cell", "uterine nk cell", "cox6a2+ nk cell",
           "cd160+ nk cell"],
    "NKT": ["mature nk t cell", "nk t cell", "type i nk t cell"],
    "B": ["b cell", "naive b cell", "memory b cell"],
    "Plasma": ["plasma cell", "antibody secreting cell"],
    "Monocyte": ["monocyte", "classical monocyte", "intermediate monocyte",
                 "non-classical monocyte"],
    "Macrophage": ["macrophage", "macrophage of small intestine",
                   "testicular macrophage (cd163+)", "tissue-resident macrophage",
                   "muscle macrophage", "colon macrophage"],
    "cDC1": ["cdc1", "cd141-positive myeloid dendritic cell"],
    "cDC2": ["cdc2", "cd1c-positive myeloid dendritic cell"],
    "Neutrophil": ["neutrophil", "nampt neutrophil", "cd24 neutrophil"],
    "Fibroblast": ["fibroblast", "fibroblast of breast", "cd34+ fibroblasts",
                   "alveolar fibroblast", "adventitial fibroblast",
                   "fibroblast of cardiac tissue", "thymic fibroblast type 1",
                   "thymic fibroblast type 2", "endometrial stromal fibroblast",
                   "uterine fibroblast", "myometrial fibroblast",
                   "stromal fibroblasts"],
    "Smooth muscle": ["smooth muscle cell", "vascular associated smooth muscle cell",
                      "airway smooth muscle cell", "myometrial smooth muscle"],
    "Vascular endothelium": [
        "endothelial cell", "vascular endothelial cell", "blood vessel endothelial cell",
        "endothelial cell of vascular tree", "cardiac endothelial cell",
        "endothelial cell of arteriole", "endothelial cell of venule",
        "capillary endothelial cell", "vein endothelial cell",
        "arterial endothelial cell", "endothelial cell of artery",
        "venous capillary endothelial cell", "capillary aerocyte",
        "retinal blood vessel endothelial cell", "colon endothelial cell",
    ],
    "Small intestine enterocyte": [
        "enterocyte of epithelium proper of small intestine",
        "enterocyte of epithelium proper of ileum",
        "enterocyte of epithelium proper of duodenum",
        "enterocyte of epithelium proper of jejunum",
    ],
    "Prostate luminal": ["ltf+ epithelial cell", "cpe high luminal epithelial cell",
                         "cpe low luminal epithelial cell"],
    "Club": ["club cell", "pigr high club epithelial cell",
             "scgb3a1 high club epithelial cell"],
    "Salivary acinar": ["acinar cell of salivary gland",
                        "sero-mucinous acinar cell of salivary gland",
                        "serous acinar cell of salivary gland",
                        "mucinous acinar cell of salivary gland"],
}
FAMILY_MAPPING = {_label(x): f for f, xs in _FAMILIES.items() for x in xs}

EXCLUDE_CELL_TYPES = {
    "t cells-b cell-nk cell", "epithelial and mesenchymal population",
    "schwann cell or melanocyte", "melanocyte or limbal stem cell",
    "smooth muscle_pericyte", "myofibroblast cell and pericyte",
    "monocyte_macrophage", "stellate_fibroblast", "theca_stroma",
    "follicle", "cycling cell",
}


def _assay(value):
    value = _label(value)
    if value in {"10x", "10x genomics", "10x chromium", "10xgenomics"}:
        return "10x"
    if value in {"smartseq", "smartseq2", "smart-seq", "smart-seq2", "ss2"}:
        return "smartseq2"
    return value


def _valid(series):
    s = series.astype("string").str.strip()
    return s.notna() & ~s.str.casefold().isin(["", "na", "nan", "none", "unknown"])


def _gene_mapping(adata, gtf, gene_col):
    required = {"gene", "CHROM", "gene_start", "gene_end"}
    if not required.issubset(gtf.columns):
        raise ValueError(f"gtf is missing {sorted(required - set(gtf.columns))}")
    gtf = gtf.copy()
    if gtf[list(required)].isna().any().any():
        raise ValueError("Missing gene identifiers or coordinates in gtf.")
    gtf["gene"] = gtf["gene"].astype(str)
    gtf["CHROM"] = gtf["CHROM"].astype(str).str.replace(r"^chr", "", regex=True)
    gtf = gtf[gtf.CHROM.isin([str(c) for c in range(1, 23)])].copy()
    gtf = gtf.drop_duplicates()
    if gtf.gene.duplicated().any():
        raise ValueError("gtf must have one unambiguous row per gene (as data.hg38 does).")
    for column in ["gene_start", "gene_end"]:
        values = pd.to_numeric(gtf[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or np.any(values != np.floor(values)):
            raise ValueError(f"Invalid gtf {column} coordinates.")
        gtf[column] = values.astype(np.int64)
    if (gtf.gene_end <= gtf.gene_start).any() or (gtf.gene_start < 0).any():
        raise ValueError("Invalid gtf interval bounds.")
    gtf = (gtf.assign(_chrom=gtf.CHROM.astype(int))
           .sort_values(["_chrom", "gene_start", "gene_end"], kind="stable")
           .drop(columns="_chrom").reset_index(drop=True))
    aliases = {g: g for g in gtf.gene}
    if "gene_ensembl" in gtf:
        ensembl = gtf.gene_ensembl.astype("string").str.replace(r"\.\d+$", "", regex=True)
        if ensembl.dropna().duplicated().any():
            raise ValueError("Ambiguous gene_ensembl entries in gtf.")
        for gene_id, symbol in zip(ensembl, gtf.gene):
            if pd.notna(gene_id):
                aliases[str(gene_id)] = symbol
    if gene_col is not None and gene_col not in adata.var:
        raise ValueError(f"var lacks gene_col={gene_col!r}")
    source = (pd.Series(adata.var_names.astype(str), dtype="string") if gene_col is None
              else adata.var[gene_col].astype("string").reset_index(drop=True))
    source = source.str.replace(r"^(ENSG\d+)\.\d+$", r"\1", regex=True)
    mapped = source.map(aliases)
    feature_map = pd.DataFrame({"input_feature": np.asarray(adata.var_names, str),
                                "input_identifier": source, "gene": mapped})
    feature_map["status"] = np.where(mapped.notna(), "mapped", "unmapped_or_non_autosomal")
    # Multiple input columns for one symbol are SUMMED, never averaged.
    feature_map["n_input_columns_for_gene"] = mapped.map(mapped.value_counts()).fillna(0).astype(int)
    gtf = gtf[gtf.gene.isin(mapped.dropna())].reset_index(drop=True)
    if len(gtf) < 2:
        raise ValueError("Fewer than two annotated autosomal genes matched. Set gene_col if needed.")
    rows = np.flatnonzero(mapped.notna())
    cols = pd.Index(gtf.gene).get_indexer(mapped.iloc[rows])
    projection = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                   shape=(adata.n_vars, len(gtf)))
    return gtf, projection, feature_map


def _prob(counts, prior_mass):
    """Posterior mean for a symmetric Dirichlet prior of TOTAL mass alpha."""
    x = np.asarray(counts, dtype=np.float64)
    total = x.sum(axis=-1, keepdims=True)
    if np.any(total <= 0):
        raise ValueError("Zero-count pseudobulk.")
    return (x + prior_mass / x.shape[-1]) / (total + prior_mass)


def _js(p, q):
    """Equal-prior Jensen-Shannon DIVERGENCE, natural logarithms, no sqrt."""
    p = np.asarray(p, float)
    q = np.asarray(q, float)
    p = p / p.sum()
    q = q / q.sum()
    m = (p + q) / 2
    return max(0.0, float((rel_entr(p, m).sum() + rel_entr(q, m).sum()) / 2))


def _block_difference(p, q, block_codes, min_expression, min_block_genes):
    """Largest absolute block mean gene log2 ratio (not a CNA call)."""
    keep = np.maximum(p, q) >= min_expression
    n = np.bincount(block_codes, weights=keep, minlength=block_codes.max() + 1)
    lfc = np.log2(np.maximum(p, 1e-300)) - np.log2(np.maximum(q, 1e-300))
    sums = np.bincount(block_codes, weights=np.where(keep, lfc, 0), minlength=len(n))
    eligible = n >= min_block_genes
    if not eligible.any():
        return np.nan, 0
    return float(np.max(np.abs(sums[eligible] / n[eligible]))), int(eligible.sum())


def _pair_score(left, right, block_codes, *, min_compare_donors, ratio_threshold,
                max_js, noise_floor, max_block_log2, min_expression,
                min_block_genes):
    """left/right map donor -> normalized gene profile; match donors explicitly."""
    shared = sorted(set(left) & set(right))
    result = {"n_shared_donors": len(shared), "shared_donors": json.dumps(shared),
              "between_js": np.nan, "donor_noise": np.nan, "ratio": np.nan,
              "max_block_log2": np.nan, "n_blocks": 0, "mergeable": False,
              "reason": "insufficient_shared_donors"}
    if len(shared) < min_compare_donors:
        return result
    a = np.stack([left[d] for d in shared])
    b = np.stack([right[d] for d in shared])
    noise = [_js(p[i], p[j]) for p in (a, b)
             for i, j in itertools.combinations(range(len(shared)), 2)]
    noise = max(float(np.median(noise)), noise_floor)
    pa, pb = a.mean(axis=0), b.mean(axis=0)
    between = _js(pa, pb)
    block_diff, n_blocks = _block_difference(pa, pb, block_codes, min_expression, min_block_genes)
    ratio = between / noise
    failures = []
    if ratio > ratio_threshold:
        failures.append("donor_scaled_js")
    if between > max_js:
        failures.append("absolute_js")
    if max_block_log2 is not None:
        if not n_blocks:
            failures.append("insufficient_genomic_blocks")
        elif block_diff > max_block_log2:
            failures.append("genomic_block_difference")
    result.update(between_js=between, donor_noise=noise, ratio=ratio,
                  max_block_log2=block_diff, n_blocks=n_blocks,
                  mergeable=not failures, reason=";".join(failures) or "similar")
    return result


def _complete_link(items, scores):
    """Only merge clusters when ALL original cross-pairs satisfy the criteria.

    This prevents a chain of individually similar neighbours from erasing a
    clearly different endpoint. Missing evidence never counts as similarity.
    """
    clusters = [(i,) for i in sorted(items)]
    history = []
    while len(clusters) > 1:
        candidates = []
        for i, j in itertools.combinations(range(len(clusters)), 2):
            pair_scores = [scores[tuple(sorted((a, b)))]
                           for a in clusters[i] for b in clusters[j]]
            if all(s["mergeable"] for s in pair_scores):
                candidates.append((max(s["ratio"] for s in pair_scores),
                                   clusters[i], clusters[j], i, j))
        if not candidates:
            break
        ratio, left, right, i, j = min(candidates)
        merged = tuple(sorted(left + right))
        history.append({"left": json.dumps(left), "right": json.dumps(right),
                        "merged": json.dumps(merged), "worst_ratio": ratio})
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
        clusters.sort()
    return clusters, history


def _aggregate_counts(adata, metadata, projection, *, layer, chunk_size,
                      min_cell_counts, min_cell_genes, max_pct_counts_mt,
                      memmap_path):
    """Read raw counts in contiguous cell chunks; only pseudobulks are dense."""
    by = ["assay", "compartment", "compression_group", "family", "label",
          "context", "donor"]
    selected = metadata["_keep"].to_numpy(bool)
    tuples = pd.MultiIndex.from_frame(metadata.loc[selected, by])
    selected_codes, groups = pd.factorize(tuples, sort=True)
    pb_obs = groups.to_frame(index=False)
    pb_obs.columns = by
    codes = np.full(adata.n_obs, -1, dtype=np.int64)
    codes[selected] = selected_codes
    shape = (len(groups), projection.shape[1])
    log.info("Pseudobulk matrix: %s (~%.2f GiB float64).", shape, np.prod(shape) * 8 / 2**30)
    if memmap_path is None:
        counts = np.zeros(shape, dtype=np.float64)
    else:
        memmap_path = Path(memmap_path)
        if memmap_path.exists():
            raise FileExistsError(f"Refusing to overwrite pseudobulk scratch file: {memmap_path}")
        memmap_path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float64, shape=shape)
        counts[:] = 0
    n_cells = np.zeros(len(groups), dtype=np.int64)
    cell_qc = {"metadata_selected": int(selected.sum()), "passed_count_qc": 0,
               "failed_count_qc": 0}
    matrix = adata.X if layer is None else adata.layers[layer]
    for start in range(0, adata.n_obs, chunk_size):
        end = min(start + chunk_size, adata.n_obs)
        use = codes[start:end] >= 0
        if not use.any():
            continue
        x = matrix[start:end, :]
        x = sparse.csr_matrix(x)[use].astype(np.float64)
        if x.data.size and (not np.isfinite(x.data).all() or np.any(x.data < 0)
                            or np.any(np.abs(x.data - np.rint(x.data)) > 1e-6)):
            raise ValueError("Selected .X/layer entries must be finite, nonnegative raw integer counts.")
        x = (x @ projection).tocsr()
        x.eliminate_zeros()
        total = np.asarray(x.sum(axis=1)).ravel()
        good = (total >= min_cell_counts) & (x.getnnz(axis=1) >= min_cell_genes) & (total > 0)
        if max_pct_counts_mt is not None:
            mt = pd.to_numeric(adata.obs.iloc[start:end].loc[use, "pct_counts_mt"], errors="coerce").to_numpy()
            good &= np.isfinite(mt) & (mt <= max_pct_counts_mt)
        cell_qc["passed_count_qc"] += int(good.sum())
        cell_qc["failed_count_qc"] += int((~good).sum())
        batch_codes = codes[start:end][use][good]
        if not len(batch_codes):
            continue
        active, inv = np.unique(batch_codes, return_inverse=True)
        group_matrix = sparse.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))),
                                         shape=(len(active), len(inv)))
        counts[active] += (group_matrix @ x[good]).toarray()
        n_cells[active] += np.bincount(inv, minlength=len(active))
        if start // chunk_size % 50 == 0:
            log.info("Read %s/%s cells.", end, adata.n_obs)
    if isinstance(counts, np.memmap):
        counts.flush()
    pb_obs["n_cells"] = n_cells
    pb_obs["n_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    return counts, pb_obs, cell_qc


def _fingerprint(frame):
    return hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes()).hexdigest()


def _json_union(values):
    """Union JSON lists while tolerating scalar legacy manifest fields."""
    result = set()
    for value in values:
        try:
            decoded = json.loads(value) if isinstance(value, str) else value
        except json.JSONDecodeError:
            decoded = value
        if isinstance(decoded, list):
            for item in decoded:
                # Contexts are themselves lists and therefore unhashable.
                result.add(json.dumps(item, sort_keys=True, ensure_ascii=False))
        elif pd.notna(decoded):
            result.add(json.dumps(decoded, sort_keys=True, ensure_ascii=False))
    return json.dumps([json.loads(item) for item in sorted(result)], ensure_ascii=False)


def annotate_manifest_compression_groups(
    manifest,
    obs,
    *,
    cell_type_col="free_annotation",
    compression_group_col="broad_cell_class",
    family_mapping=None,
    protect_curated_families=True,
):
    """Add biological compression groups to a pre-v0.2 reference manifest.

    This lets an already-built reference be compacted without rereading the
    count matrix. A leaf label must map unambiguously to one broad group in
    `obs`; ambiguous or absent mappings receive a reference-specific group and
    therefore cannot merge silently. Curated family boundaries take priority.

    Returns annotated_manifest, audit.
    """
    required = {cell_type_col, compression_group_col}
    missing = required - set(obs.columns)
    if missing:
        raise ValueError(f"obs is missing compression fields: {sorted(missing)}")
    if "reference" not in manifest or "original_types" not in manifest:
        raise ValueError("manifest needs reference and original_types columns.")
    families = FAMILY_MAPPING if family_mapping is None else {
        _label(k): str(v) for k, v in family_mapping.items()}
    valid_rows = _valid(obs[cell_type_col]) & _valid(obs[compression_group_col])
    pairs = pd.DataFrame({
        "label": obs.loc[valid_rows, cell_type_col].astype("string").map(_label),
        "group": obs.loc[valid_rows, compression_group_col].astype("string").str.strip(),
    }).drop_duplicates()
    label_groups = pairs.groupby("label", observed=True).group.apply(
        lambda x: tuple(sorted(set(map(str, x)))))
    result = manifest.copy()
    groups = []
    audit = []
    for _, row in result.iterrows():
        labels = [_label(x) for x in json.loads(row.original_types)]
        curated = {families[x] for x in labels if x in families}
        if protect_curated_families and len(curated) == 1 and all(x in families for x in labels):
            group = next(iter(curated))
            reason = "curated_family"
        else:
            candidates = {value for label in labels for value in label_groups.get(label, ())}
            unambiguous = all(len(label_groups.get(label, ())) == 1 for label in labels)
            if unambiguous and len(candidates) == 1:
                group = next(iter(candidates))
                reason = "broad_group"
            else:
                group = f"unresolved::{row.reference}"
                reason = "missing_or_ambiguous_group"
        groups.append(group)
        audit.append({"reference": row.reference, "compression_group": group,
                      "reason": reason, "original_types": row.original_types})
    result["compression_group"] = groups
    return result, pd.DataFrame(audit)


def _compression_score(
    left,
    right,
    profiles,
    block_codes,
    *,
    max_information_loss,
    max_information_ratio,
    max_js,
    max_pairwise_js,
    max_block_log2,
    min_expression,
    min_block_genes,
    source_noise=None,
    global_noise=np.nan,
    noise_floor=1e-8,
):
    """Score loss from replacing two clusters by one equal-leaf centroid.

    `left` and `right` are tuples of original reference column indices. Each
    original reference receives equal weight, so a tissue represented by many
    atlas cells cannot dominate merely because it was sampled more deeply.
    """
    left_idx = np.asarray(left, dtype=int)
    right_idx = np.asarray(right, dtype=int)
    merged_idx = np.r_[left_idx, right_idx]
    center_left = profiles[:, left_idx].mean(axis=1)
    center_right = profiles[:, right_idx].mean(axis=1)
    center = profiles[:, merged_idx].mean(axis=1)
    center_left /= center_left.sum()
    center_right /= center_right.sum()
    center /= center.sum()
    between_js = _js(center_left, center_right)
    source_js = np.asarray([_js(profiles[:, i], center) for i in merged_idx])
    pairwise_js = [_js(profiles[:, i], profiles[:, j])
                   for i, j in itertools.combinations(merged_idx, 2)]
    # Generalized JSD = I(source reference; gene identity) under a uniform
    # source prior. This is the exact information discarded by representing
    # all source distributions with their equal-weight centroid.
    information_loss = float(np.mean([
        rel_entr(profiles[:, i], center).sum() for i in merged_idx]))
    max_source_js = float(source_js.max())
    max_complete_link_js = float(max(pairwise_js, default=0.0))
    noise_values = np.asarray([], dtype=float)
    if source_noise is not None:
        noise_values = np.asarray(source_noise, dtype=float)[merged_idx]
        noise_values = noise_values[np.isfinite(noise_values)]
    local_noise = float(np.median(noise_values)) if len(noise_values) else np.nan
    noise_candidates = [x for x in (local_noise, global_noise) if np.isfinite(x)]
    # Match the ATAC logic: a locally quiet family must not make a tiny
    # cross-context difference look disproportionately large.
    donor_noise = (max(*noise_candidates, noise_floor)
                   if noise_candidates else np.nan)
    information_ratio = (information_loss / donor_noise
                         if np.isfinite(donor_noise) else np.nan)
    block_values = []
    n_blocks = []
    for i in merged_idx:
        value, n = _block_difference(
            profiles[:, i], center, block_codes, min_expression, min_block_genes)
        block_values.append(value)
        n_blocks.append(n)
    finite_blocks = [x for x in block_values if np.isfinite(x)]
    max_block = max(finite_blocks) if finite_blocks else np.nan
    failures = []
    if information_loss > max_information_loss:
        failures.append("information_loss")
    if (max_information_ratio is not None and np.isfinite(information_ratio)
            and information_ratio > max_information_ratio):
        failures.append("donor_noise_ratio")
    if max_source_js > max_js:
        failures.append("source_js")
    if max_pairwise_js is not None and max_complete_link_js > max_pairwise_js:
        failures.append("complete_link_js")
    if max_block_log2 is not None:
        if not finite_blocks or min(n_blocks, default=0) == 0:
            failures.append("insufficient_genomic_blocks")
        elif max_block > max_block_log2:
            failures.append("genomic_block_difference")
    return {
        "information_loss": information_loss,
        "donor_noise_js": donor_noise,
        "information_ratio": information_ratio,
        "between_js": between_js,
        "max_source_js": max_source_js,
        "max_pairwise_js": max_complete_link_js,
        "max_block_log2": max_block,
        "n_sources": int(len(merged_idx)),
        "mergeable": not failures,
        "reason": ";".join(failures) or "similar",
    }


def compact_rna_reference(
    reference,
    manifest,
    gene_gtf,
    *,
    target_references=64,
    group_col="compression_group",
    family_col="family",
    max_information_loss=0.002,
    max_information_ratio=2.0,
    max_js=0.01,
    max_pairwise_js=0.02,
    max_block_log2=0.20,
    donor_qc=None,
    allow_cross_compartment=True,
    cross_compartment_threshold_scale=1,
    noise_floor=1e-8,
    block_size=5_000_000,
    min_block_genes=10,
    min_block_expression=2e-6,
):
    """Greedily compress a SpaceNumbat RNA reference under biological gates.

    References may merge only when assay and curated biological family agree.
    Tissue and anatomy are evidence variables rather than hard boundaries.
    Compartments may also differ by default, but cross-compartment candidates
    use thresholds multiplied by `cross_compartment_threshold_scale`.

    The target is soft: compression stops above it if no admissible merge
    remains. The objective is generalized JSD—the mutual information between
    source-reference identity and gene identity discarded by replacing the
    sources with one centroid. Worst-source JSD, complete-link pairwise JSD,
    donor-noise ratio, and genomic-block residuals protect against chaining,
    tissue-specific programs, and CNA-like baselines.

    This function can be used on an existing reference if its manifest has an
    explicit biologically curated grouping column. It never infers biological
    compatibility from expression similarity alone.

    Returns compact_reference, compact_manifest, mapping, history, rejected.
    """
    if not isinstance(reference, pd.DataFrame) or reference.empty:
        raise ValueError("reference must be a nonempty gene-by-profile DataFrame.")
    if reference.columns.has_duplicates or reference.index.has_duplicates:
        raise ValueError("reference gene and profile names must be unique.")
    if not np.isfinite(reference.to_numpy(float)).all() or (reference < 0).any().any():
        raise ValueError("reference must contain finite nonnegative values.")
    column_sums = reference.sum(axis=0)
    if (column_sums <= 0).any() or not np.isfinite(column_sums.to_numpy()).all():
        bad = column_sums.index[(column_sums <= 0) | ~np.isfinite(column_sums)].tolist()
        raise ValueError(f"Reference columns must have finite positive sums; invalid: {bad[:10]}")
    reference = reference.div(column_sums, axis=1)
    required = {"reference", "assay", "compartment", family_col}
    if group_col is not None:
        required.add(group_col)
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing compression fields: {sorted(missing)}")
    if set(reference.columns) != set(manifest["reference"]):
        raise ValueError("manifest.reference must match reference columns exactly.")
    if manifest.reference.duplicated().any():
        raise ValueError("manifest must contain one row per reference.")
    if target_references is None:
        target_references = 1
    if int(target_references) != target_references or target_references < 1:
        raise ValueError("target_references must be a positive integer or None.")
    for name, value in {"max_information_loss": max_information_loss,
                        "max_js": max_js,
                        "min_block_expression": min_block_expression}.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if max_block_log2 is not None and (not np.isfinite(max_block_log2) or max_block_log2 < 0):
        raise ValueError("max_block_log2 must be None or finite and nonnegative.")
    for name, value in {"max_information_ratio": max_information_ratio,
                        "max_pairwise_js": max_pairwise_js}.items():
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be None or finite and nonnegative.")
    if (not np.isfinite(cross_compartment_threshold_scale)
            or not 0 < cross_compartment_threshold_scale <= 1):
        raise ValueError("cross_compartment_threshold_scale must be in (0, 1].")
    if not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError("noise_floor must be finite and positive.")
    gene_gtf = gene_gtf.set_index("gene").loc[reference.index].reset_index()
    block_keys = pd.MultiIndex.from_arrays([
        gene_gtf.CHROM.astype(str),
        pd.to_numeric(gene_gtf.gene_start, errors="raise").astype(np.int64) // block_size,
    ])
    block_codes, _ = pd.factorize(block_keys, sort=False)
    profiles = reference.to_numpy(float)
    meta = manifest.set_index("reference").loc[reference.columns]
    if not _valid(meta[family_col]).all():
        raise ValueError("Every reference needs a non-missing biological family.")
    keys = [(str(row.assay), str(row[family_col])) +
            (() if allow_cross_compartment else (str(row.compartment),))
            for _, row in meta.iterrows()]
    source_noise = np.full(reference.shape[1], np.nan, dtype=float)
    global_noise = np.nan
    if donor_qc is not None and len(donor_qc):
        donor_qc = pd.DataFrame(donor_qc).copy()
        needed = {"reference", "js_to_consensus"}
        if not needed.issubset(donor_qc.columns):
            raise ValueError("donor_qc needs reference and js_to_consensus columns.")
        values = pd.to_numeric(donor_qc.js_to_consensus, errors="coerce")
        valid_noise = np.isfinite(values) & (values >= 0)
        donor_qc = donor_qc.loc[valid_noise].assign(js_to_consensus=values[valid_noise])
        noise_by_reference = donor_qc.groupby("reference").js_to_consensus.median()
        source_noise = np.asarray([
            noise_by_reference.get(name, np.nan) for name in reference.columns], dtype=float)
        if len(donor_qc):
            global_noise = max(float(donor_qc.js_to_consensus.median()), noise_floor)
    clusters = {i: (i,) for i in range(reference.shape[1])}
    cluster_keys = {i: keys[i] for i in clusters}
    active = set(clusters)
    heap = []
    serial = itertools.count()
    rejected = []

    def add_candidate(i, j):
        if cluster_keys[i] != cluster_keys[j]:
            return
        merged_sources = tuple(sorted(clusters[i] + clusters[j]))
        rows = meta.iloc[list(merged_sources)]
        compartments = sorted(set(rows.compartment.astype(str)))
        cross_compartment = len(compartments) > 1
        scale = cross_compartment_threshold_scale if cross_compartment else 1.0
        tissues = json.loads(_json_union(rows.tissues)) if "tissues" in rows else []
        contexts = json.loads(_json_union(rows.contexts)) if "contexts" in rows else []
        score = _compression_score(
            clusters[i], clusters[j], profiles, block_codes,
            max_information_loss=max_information_loss * scale,
            max_information_ratio=(None if max_information_ratio is None
                                   else max_information_ratio * scale),
            max_js=max_js * scale,
            max_pairwise_js=(None if max_pairwise_js is None
                             else max_pairwise_js * scale),
            max_block_log2=(None if max_block_log2 is None
                            else max_block_log2 * scale),
            min_expression=min_block_expression,
            min_block_genes=min_block_genes,
            source_noise=source_noise,
            global_noise=global_noise,
            noise_floor=noise_floor)
        score.update({
            "cross_compartment": cross_compartment,
            "threshold_scale": scale,
            "n_tissues": len(tissues),
            "n_contexts": len(contexts),
            "compartments": json.dumps(compartments),
        })
        # Inadmissible candidates are retained in diagnostics but do not enter
        # the priority queue, avoiding repeated evaluation.
        if score["mergeable"]:
            heapq.heappush(heap, (score["information_loss"], score["max_pairwise_js"],
                                  next(serial), i, j, score))
        else:
            rejected.append({"left_cluster": json.dumps(clusters[i]),
                             "right_cluster": json.dumps(clusters[j]), **score})

    for i, j in itertools.combinations(sorted(active), 2):
        add_candidate(i, j)
    history = []
    next_id = reference.shape[1]
    while len(active) > target_references and heap:
        _, _, _, i, j, score = heapq.heappop(heap)
        if i not in active or j not in active:
            continue
        left, right = clusters[i], clusters[j]
        merged = tuple(sorted(left + right))
        k = next_id
        next_id += 1
        clusters[k] = merged
        cluster_keys[k] = cluster_keys[i]
        active.remove(i)
        active.remove(j)
        active.add(k)
        history.append({
            "step": len(history) + 1,
            "left_sources": json.dumps([reference.columns[x] for x in left]),
            "right_sources": json.dumps([reference.columns[x] for x in right]),
            "n_references_after": len(active),
            **score,
        })
        for other in sorted(active - {k}):
            add_candidate(min(k, other), max(k, other))

    compact_profiles = {}
    compact_rows = []
    mapping_rows = []
    output_group_col = group_col or "compression_group"
    for cluster_id in sorted(active, key=lambda x: clusters[x]):
        sources = clusters[cluster_id]
        source_names = [reference.columns[i] for i in sources]
        rows = meta.loc[source_names]
        center = profiles[:, np.asarray(sources)].mean(axis=1)
        center /= center.sum()
        if len(sources) == 1:
            name = source_names[0]
        else:
            group = str(rows[family_col].iloc[0])
            digest = hashlib.sha256(json.dumps(source_names).encode()).hexdigest()[:10]
            name = f"compact::{group}::{digest}"
        compact_profiles[name] = center
        family = str(rows[family_col].iloc[0])
        compartments = sorted(set(rows.compartment.astype(str)))
        compression_groups = (sorted(set(rows[group_col].astype(str)))
                              if group_col is not None else [family])
        compact_group = (compression_groups[0] if len(compression_groups) == 1
                         else f"mixed::{family}")
        for source in source_names:
            mapping_rows.append({"source_reference": source, "compact_reference": name,
                                 "family": family,
                                 "source_compartment": str(meta.loc[source, "compartment"]),
                                 "compression_group": compact_group})
        original_types = _json_union(rows.original_types) if "original_types" in rows else json.dumps([])
        tissues = _json_union(rows.tissues) if "tissues" in rows else json.dumps([])
        contexts = _json_union(rows.contexts) if "contexts" in rows else json.dumps([])
        donors = _json_union(rows.donors) if "donors" in rows else json.dumps([])
        donor_list = json.loads(donors)
        context_list = json.loads(contexts)
        compact_rows.append({
            "reference": name,
            "merged_type": (str(rows.merged_type.iloc[0])
                            if len(sources) == 1 and "merged_type" in rows else family),
            "assay": rows.assay.iloc[0],
            "compartment": (compartments[0] if len(compartments) == 1
                            else "pan_compartment"),
            "compartments": json.dumps(compartments),
            "family": family,
            output_group_col: compact_group,
            "compression_groups": json.dumps(compression_groups),
            "original_types": original_types,
            "tissues": tissues,
            "contexts": contexts,
            "donors": donors,
            "n_donors": len(donor_list),
            "n_contexts": len(context_list),
            "n_cells": int(rows.n_cells.sum()) if "n_cells" in rows else np.nan,
            "n_counts": float(rows.n_counts.sum()) if "n_counts" in rows else np.nan,
            "entropy_nats": float(-np.sum(center * np.log(center))),
            "consensus": "equal_source_reference_mean",
            "source_references": json.dumps(source_names),
            "n_source_references": len(source_names),
        })
    compact = pd.DataFrame(compact_profiles, index=reference.index)
    compact.index.name = reference.index.name
    compact_manifest = pd.DataFrame(compact_rows)
    history_frame = pd.DataFrame(history)
    rejected_frame = pd.DataFrame(rejected)
    if len(compact.columns) > target_references:
        log.warning(
            "Compression stopped at %s profiles (soft target %s): no further "
            "biologically allowed merge passed the information/CNA safeguards.",
            len(compact.columns), target_references)
    return compact, compact_manifest, pd.DataFrame(mapping_rows), history_frame, rejected_frame


def build_rna_reference(
    adata,
    gtf=None,
    *,
    donor_col="donor",
    tissue_col="tissue",
    cell_type_col="free_annotation",
    compartment_col="compartment",
    assay_col="method",
    assays=("10x",),
    anatomy_col=None,
    gene_col=None,
    layer=None,
    family_mapping=None,
    compression_group_col="broad_cell_class",
    protect_curated_families=True,
    target_references=64,
    compression_max_information_loss=0.002,
    compression_max_information_ratio=2.0,
    compression_max_js=0.01,
    compression_max_pairwise_js=0.02,
    compression_max_block_log2=0.20,
    compression_allow_cross_compartment=True,
    compression_cross_compartment_scale=1,
    exclude_cell_types=None,
    exclude_germline=True,
    cell_mask=None,
    min_cell_counts=100,
    min_cell_genes=100,
    max_pct_counts_mt=None,
    min_cells=50,
    min_counts=50_000,
    min_donors=2,
    min_compare_donors=3,
    subtype_ratio_threshold=1.5,
    tissue_ratio_threshold=2.0,
    max_js=0.01,
    noise_floor=1e-8,
    prior_mass=1.0,
    consensus="mean",
    block_size=5_000_000,
    max_block_log2=0.15,
    min_block_genes=10,
    min_block_expression=2e-6,
    chunk_size=8192,
    memmap_path=None,
    return_diagnostics=False,
):
    """Construct a normal RNA atlas compatible with run_spacenumbat(mode='rna').

    Cells are aggregated within donor x tissue [x anatomy] x label x assay.
    min_cells/min_counts are per such pseudobulk, after per-cell count QC.
    Profiles require >=min_donors; merges require >=min_compare_donors present
    in BOTH candidates. Missing evidence keeps profiles separate.

    1. Subtype merging: compare only curated-family labels within the SAME
       tissue/anatomy context, on matched donors. All shared contexts must
       pass; one inadequately replicated context blocks the global merge.
    2. Tissue/region merging: within a resulting type, merge only contexts
       whose matched-donor profiles pass the same safeguards.
    Both steps use complete linkage, equal-prior JSD (nats), donor-noise
    ratios, an absolute JSD cap, and a genomic-block log2-ratio guard.

    Each assay is processed separately even when assays=None (all assays).
    For 10x/Visium applications keep the default assays=('10x',). Smart-seq2
    references remain protocol-specific; library-size normalization does not
    remove gene-length/capture bias. No TPM, HVGs, log1p or scVI is used here.

    gtf defaults to spacenumbat.data.hg38. gene_col=None maps var_names by
    exact symbols or version-stripped Ensembl IDs. Otherwise use adata.var's
    indicated column. Duplicate mapped symbols are summed. All matched,
    expressed autosomal genes are retained; no sex-chromosome inference.

    family_mapping=None uses FAMILY_MAPPING; a supplied dict REPLACES it.
    Unmapped labels remain their own families. exclude_cell_types, when
    supplied, replaces EXCLUDE_CELL_TYPES. Germline exclusion is independent.
    cell_mask must be a positional bool array or an exactly aligned Series.
    No universal mitochondrial cutoff is imposed; apply tissue-aware QC.

    Default second-stage compression
    --------------------------------
    target_references=64 enables biologically gated compression by default.
    Set it to None to retain every conservative first-stage reference, or to
    another positive integer for a different soft maximum (for example 48).
    Candidate profiles must share assay and the same curated biological
    family. Tissue and anatomy are not hard boundaries: references may become
    pan-tissue when their residual information is no larger than the allowed
    donor-calibrated and absolute thresholds. Compartments may also differ by
    default, because the same functional population can receive inconsistent
    compartment annotations across tissues; such candidates use thresholds
    multiplied by compression_cross_compartment_scale (0.5 by default).

    A merge is accepted only if generalized information loss, its ratio to
    donor variability, worst-source JSD, complete-link pairwise JSD, and the
    genomic-block residual all pass. The target is never forced across a
    failed family, information, or CNA safeguard. compression_group_col is
    retained for annotation and audit, not used as a substitute for family.

    Final estimator: sum labels within donor/context, normalize independently,
    average contexts within each donor, then average donors. consensus=
    'log_median' provides the ATAC-like alternative. prior_mass is the TOTAL
    uniform Dirichlet pseudocount per pseudobulk (1 count by default), not
    one count per gene. Defaults are heuristics to calibrate, not significance
    levels or proof that an atlas population is diploid.

    Memory: O(n_pseudobulks*n_genes), never a dense atlas-wide cell matrix.
    memmap_path places raw pseudobulk counts on disk; use a fresh .npy path.
    Chunk size also bounds each temporary group-by-gene aggregation.

    Returns (reference, manifest, gene_gtf), plus diagnostics if requested.
    reference has unique gene index, unique profile columns, finite positive
    fractions and column sums of 1. No inputs are modified.
    """
    params = dict(locals())
    for key in ["adata", "gtf", "cell_mask"]:
        params.pop(key)
    params["cell_mask_provided"] = cell_mask is not None
    params["family_mapping"] = family_mapping
    if consensus not in {"mean", "log_median"}:
        raise ValueError("consensus must be 'mean' or 'log_median'.")
    for key in ["min_cells", "min_counts", "min_donors", "min_compare_donors",
                "chunk_size", "block_size", "min_block_genes"]:
        if params[key] < 1 or int(params[key]) != params[key]:
            raise ValueError(f"{key} must be a positive integer.")
    if min_compare_donors < 2:
        raise ValueError("At least two shared donors are needed to estimate donor variability.")
    for key in ["min_cell_counts", "min_cell_genes", "subtype_ratio_threshold",
                "tissue_ratio_threshold", "max_js", "min_block_expression"]:
        if not np.isfinite(params[key]) or params[key] < 0:
            raise ValueError(f"{key} must be finite and nonnegative.")
    if not np.isfinite(prior_mass) or prior_mass <= 0 or not np.isfinite(noise_floor) or noise_floor <= 0:
        raise ValueError("prior_mass and noise_floor must be finite and positive.")
    if max_block_log2 is not None and (not np.isfinite(max_block_log2) or max_block_log2 < 0):
        raise ValueError("max_block_log2 must be None or finite and nonnegative.")
    if max_pct_counts_mt is not None and not 0 <= max_pct_counts_mt <= 100:
        raise ValueError("max_pct_counts_mt must be between 0 and 100.")
    if target_references is not None and (target_references < 1 or int(target_references) != target_references):
        raise ValueError("target_references must be None or a positive integer.")
    for name, value in {
        "compression_max_information_loss": compression_max_information_loss,
        "compression_max_js": compression_max_js,
    }.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    if compression_max_block_log2 is not None and (
            not np.isfinite(compression_max_block_log2) or compression_max_block_log2 < 0):
        raise ValueError("compression_max_block_log2 must be None or finite and nonnegative.")
    for name, value in {
        "compression_max_information_ratio": compression_max_information_ratio,
        "compression_max_pairwise_js": compression_max_pairwise_js,
    }.items():
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be None or finite and nonnegative.")
    if (not np.isfinite(compression_cross_compartment_scale)
            or not 0 < compression_cross_compartment_scale <= 1):
        raise ValueError("compression_cross_compartment_scale must be in (0, 1].")
    if adata.obs_names.has_duplicates:
        raise ValueError("Duplicate cell IDs: verify concatenation and avoid duplicated atlas cells.")
    if adata.var_names.has_duplicates:
        raise ValueError("Duplicate var_names: give features unique IDs; mapped duplicate symbols are supported.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"No layer {layer!r}.")
    if gtf is None:
        from spacenumbat.data import hg38
        gtf = hg38
    gtf, projection, feature_map = _gene_mapping(adata, gtf, gene_col)
    compression_enabled = target_references is not None
    required = [donor_col, tissue_col, cell_type_col, compartment_col, assay_col]
    if compression_enabled and compression_group_col is not None:
        required.append(compression_group_col)
    if anatomy_col is not None:
        required.append(anatomy_col)
    if max_pct_counts_mt is not None:
        required.append("pct_counts_mt")
    missing = set(required) - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing obs columns: {sorted(missing)}")
    obs = adata.obs
    meta = pd.DataFrame(index=obs.index)
    meta["donor"] = obs[donor_col].astype("string").str.strip()
    meta["tissue"] = obs[tissue_col].astype("string").str.strip()
    meta["label"] = obs[cell_type_col].astype("string").map(_label)
    meta["compartment"] = obs[compartment_col].astype("string").str.strip()
    meta["assay"] = obs[assay_col].astype("string").map(_assay)
    if anatomy_col is None:
        meta["anatomy"] = ""
    else:
        meta["anatomy"] = obs[anatomy_col].astype("string").str.strip().where(_valid(obs[anatomy_col]), "unknown")
    meta["context"] = [json.dumps([str(t), str(a)], ensure_ascii=False)
                       for t, a in zip(meta.tissue, meta.anatomy)]
    families = FAMILY_MAPPING if family_mapping is None else {_label(k): str(v) for k, v in family_mapping.items()}
    meta["family"] = meta.label.map(families).fillna(meta.label)
    if not compression_enabled or compression_group_col is None:
        meta["compression_group"] = meta["family"].astype("string")
    else:
        broad_group = obs[compression_group_col].astype("string").str.strip()
        meta["compression_group"] = broad_group
        if protect_curated_families:
            curated = meta.label.isin(families)
            meta.loc[curated, "compression_group"] = meta.loc[curated, "family"]
    if not _valid(meta.family).all():
        # Missing original labels are removed below; invalid explicit family values are errors.
        bad = meta.loc[_valid(obs[cell_type_col]) & ~_valid(meta.family), "label"].unique()
        if len(bad):
            raise ValueError(f"Invalid family values for: {bad.tolist()}")
    keep = np.ones(adata.n_obs, dtype=bool)
    for key in [donor_col, tissue_col, cell_type_col, compartment_col, assay_col]:
        keep &= _valid(obs[key]).to_numpy(bool)
    if compression_enabled and compression_group_col is not None:
        keep &= _valid(obs[compression_group_col]).to_numpy(bool)
    if cell_mask is not None:
        if isinstance(cell_mask, pd.Series) and not cell_mask.index.equals(obs.index):
            raise ValueError("cell_mask Series index must exactly match adata.obs_names.")
        arr = np.asarray(cell_mask)
        if arr.shape != (adata.n_obs,) or arr.dtype.kind != "b":
            raise ValueError("cell_mask must be a positional Boolean array of length n_obs.")
        keep &= arr
    if assays is not None:
        if isinstance(assays, str):
            assays = (assays,)
        keep &= meta.assay.isin([_assay(a) for a in assays]).to_numpy()
    excluded = EXCLUDE_CELL_TYPES if exclude_cell_types is None else {_label(x) for x in exclude_cell_types}
    keep &= ~meta.label.isin(excluded).to_numpy()
    if exclude_germline:
        keep &= ~meta.label.str.contains(r"spermat|oocyte|male germ cell", regex=True).to_numpy()
    meta["_keep"] = keep
    if not keep.any():
        raise ValueError(f"No cells after metadata/assay filters. Available assays: {sorted(meta.assay.unique())}")
    metadata_audit = (meta.groupby(["assay", "tissue", "label", "_keep"], dropna=False, observed=True)
                      .size().rename("n_cells").reset_index())
    raw, pb, cell_qc = _aggregate_counts(
        adata, meta, projection, layer=layer, chunk_size=chunk_size,
        min_cell_counts=min_cell_counts, min_cell_genes=min_cell_genes,
        max_pct_counts_mt=max_pct_counts_mt, memmap_path=memmap_path)
    pb["status"] = "used"
    pb.loc[(pb.n_cells < min_cells) | (pb.n_counts < min_counts), "status"] = "low_pseudobulk_depth_or_cells"
    by = ["assay", "compartment", "compression_group", "family", "label", "context"]
    supported = pb[pb.status.eq("used")].groupby(by, observed=True).donor.transform("nunique")
    pb.loc[supported.index[supported < min_donors], "status"] = "insufficient_donors"
    good_rows = np.flatnonzero(pb.status.eq("used").to_numpy())
    if not len(good_rows):
        raise ValueError("No reference populations passed pseudobulk QC and min_donors.")
    # Avoid copying the full pseudobulk matrix merely to subset genes.
    gene_totals = np.zeros(raw.shape[1])
    for i in good_rows:
        gene_totals += raw[i]
    gene_idx = np.flatnonzero(gene_totals > 0)
    if len(gene_idx) < 2:
        raise ValueError("Fewer than two expressed autosomal genes remain.")
    gtf = gtf.iloc[gene_idx].reset_index(drop=True)
    feature_map.loc[feature_map.gene.notna() & ~feature_map.gene.isin(gtf.gene), "status"] = "not_expressed_in_eligible_pseudobulks"
    block_keys = pd.MultiIndex.from_arrays([gtf.CHROM, gtf.gene_start // block_size])
    block_codes, _ = pd.factorize(block_keys, sort=False)
    score_params = dict(min_compare_donors=min_compare_donors, max_js=max_js,
                        noise_floor=noise_floor, max_block_log2=max_block_log2,
                        min_expression=min_block_expression, min_block_genes=min_block_genes)
    pair_records, history_records, type_records = [], [], []
    pb["merged_type"] = None
    eligible = pb[pb.status.eq("used")]
    for (assay, compartment, family), family_frame in eligible.groupby(
            ["assay", "compartment", "family"], sort=True, observed=True):
        profiles = {}
        for i, row in family_frame.iterrows():
            profiles.setdefault(row.label, {}).setdefault(row.context, {})[row.donor] = _prob(raw[i, gene_idx], prior_mass)
        scores = {}
        for a, b in itertools.combinations(sorted(profiles), 2):
            shared_contexts = sorted(set(profiles[a]) & set(profiles[b]))
            details = []
            for context in shared_contexts:
                score = _pair_score(profiles[a][context], profiles[b][context], block_codes,
                                    ratio_threshold=subtype_ratio_threshold, **score_params)
                details.append(score)
                pair_records.append(dict(stage="subtype", assay=assay, compartment=compartment,
                                         family=family, left=a, right=b, context=context, **score))
            if not details:
                pair_records.append(dict(stage="subtype", assay=assay, compartment=compartment,
                                         family=family, left=a, right=b, context=None,
                                         mergeable=False, reason="no_shared_context"))
            scores[(a, b)] = {"mergeable": bool(details) and all(s["mergeable"] for s in details),
                              "ratio": max((s["ratio"] if np.isfinite(s["ratio"]) else np.inf
                                            for s in details), default=np.inf)}
        clusters, history = _complete_link(list(profiles), scores)
        history_records.extend(dict(stage="subtype", assay=assay, compartment=compartment,
                                    family=family, **h) for h in history)
        for cluster in clusters:
            short = hashlib.sha256(json.dumps(cluster).encode()).hexdigest()[:10]
            name = cluster[0] if len(cluster) == 1 else f"{family} [{short}]"
            name = f"{compartment}::{name}"
            rows = family_frame.index[family_frame.label.isin(cluster)]
            pb.loc[rows, "merged_type"] = name
            for label in cluster:
                type_records.append(dict(assay=assay, compartment=compartment, family=family,
                                         original_type=label, merged_type=name))
        log.info("%s / %s: %s labels -> %s subtype profiles.", assay, family, len(profiles), len(clusters))
        del profiles

    references, manifest, donor_qc = {}, [], []
    pb["reference"] = None
    for (assay, merged_type), type_frame in pb[pb.status.eq("used")].groupby(
            ["assay", "merged_type"], sort=True, observed=True):
        contexts = {}
        for (context, donor), frame in type_frame.groupby(["context", "donor"], sort=True, observed=True):
            counts = np.zeros(len(gene_idx))
            for i in frame.index:
                counts += raw[i, gene_idx]
            contexts.setdefault(context, {})[donor] = _prob(counts, prior_mass)
        scores = {}
        for a, b in itertools.combinations(sorted(contexts), 2):
            score = _pair_score(contexts[a], contexts[b], block_codes,
                                ratio_threshold=tissue_ratio_threshold, **score_params)
            scores[(a, b)] = score
            pair_records.append(dict(stage="tissue", assay=assay, compartment=type_frame.compartment.iloc[0],
                                     family=type_frame.family.iloc[0], merged_type=merged_type,
                                     left=a, right=b, context=None, **score))
        clusters, history = _complete_link(list(contexts), scores)
        history_records.extend(dict(stage="tissue", assay=assay, merged_type=merged_type, **h)
                               for h in history)
        for cluster in clusters:
            members = type_frame[type_frame.context.isin(cluster)]
            donors = sorted(members.donor.unique())
            donor_profiles = np.stack([
                np.mean([contexts[c][d] for c in cluster if d in contexts[c]], axis=0)
                for d in donors])
            if consensus == "mean":
                reference = donor_profiles.mean(axis=0)
            else:
                reference = np.exp(np.median(np.log(donor_profiles), axis=0))
            reference /= reference.sum()
            parsed = [json.loads(c) for c in cluster]
            if len(cluster) == 1:
                tissue_name, anatomy_name = parsed[0]
                suffix = tissue_name + ("/" + anatomy_name if anatomy_name else "")
            else:
                digest = hashlib.sha256(json.dumps(cluster).encode()).hexdigest()[:10]
                suffix = f"shared-{len(cluster)}contexts-{digest}"
            ref_name = f"{merged_type} | {suffix} | {assay}"
            if ref_name in references:
                raise RuntimeError(f"Reference name collision: {ref_name}")
            references[ref_name] = reference
            pb.loc[members.index, "reference"] = ref_name
            manifest.append(dict(reference=ref_name, merged_type=merged_type,
                                 assay=assay, compartment=members.compartment.iloc[0],
                                 compression_group=members.compression_group.iloc[0],
                                 family=members.family.iloc[0],
                                 original_types=json.dumps(sorted(members.label.unique())),
                                 tissues=json.dumps(sorted({x[0] for x in parsed})),
                                 contexts=json.dumps(parsed), donors=json.dumps(donors),
                                 n_donors=len(donors), n_contexts=len(cluster),
                                 n_cells=int(members.n_cells.sum()), n_counts=float(members.n_counts.sum()),
                                 entropy_nats=float(-np.sum(reference * np.log(reference))),
                                 consensus=consensus))
            for d, profile in zip(donors, donor_profiles):
                rows = members[members.donor.eq(d)]
                donor_qc.append(dict(reference=ref_name, donor=d, n_contexts=rows.context.nunique(),
                                     n_cells=int(rows.n_cells.sum()), n_counts=float(rows.n_counts.sum()),
                                     js_to_consensus=_js(profile, reference)))
    reference = pd.DataFrame(references, index=pd.Index(gtf.gene, name="gene"), dtype=np.float64)
    manifest = pd.DataFrame(manifest)
    n_references_before_compression = reference.shape[1]
    precompression_manifest = manifest.copy()
    compression_map = pd.DataFrame()
    compression_history = pd.DataFrame()
    compression_rejected = pd.DataFrame()
    if target_references is not None and reference.shape[1] > target_references:
        reference, manifest, compression_map, compression_history, compression_rejected = compact_rna_reference(
            reference=reference,
            manifest=manifest,
            gene_gtf=gtf,
            target_references=target_references,
            group_col="compression_group",
            family_col="family",
            max_information_loss=compression_max_information_loss,
            max_information_ratio=compression_max_information_ratio,
            max_js=compression_max_js,
            max_pairwise_js=compression_max_pairwise_js,
            max_block_log2=compression_max_block_log2,
            donor_qc=pd.DataFrame(donor_qc),
            allow_cross_compartment=compression_allow_cross_compartment,
            cross_compartment_threshold_scale=compression_cross_compartment_scale,
            noise_floor=noise_floor,
            block_size=block_size,
            min_block_genes=min_block_genes,
            min_block_expression=min_block_expression,
        )
        ref_map = compression_map.set_index("source_reference").compact_reference
        pb["reference_precompression"] = pb["reference"]
        pb["reference"] = pb["reference_precompression"].map(ref_map)
        for row in donor_qc:
            row["compact_reference"] = ref_map.get(row["reference"], row["reference"])
    if reference.empty or reference.columns.has_duplicates:
        raise RuntimeError("No valid unique reference profiles were produced.")
    if not np.isfinite(reference.to_numpy()).all() or (reference <= 0).any().any():
        raise RuntimeError("Nonfinite/nonpositive reference values.")
    if not np.allclose(reference.sum(axis=0), 1.0, atol=1e-12):
        raise RuntimeError("Reference normalization failed.")
    if isinstance(raw, np.memmap):
        raw.flush()
    versions = {}
    for package in ["numpy", "pandas", "scipy", "anndata", "spacenumbat"]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not installed"
    info = dict(builder_version=__version__, inspected_spacenumbat_commit=UPSTREAM_COMMIT,
                parameters=params, resolved_family_mapping=families,
                excluded_cell_types=sorted(excluded), versions=versions,
                n_input_cells=int(adata.n_obs), n_input_genes=int(adata.n_vars),
                n_genes=int(reference.shape[0]), n_references=int(reference.shape[1]),
                n_references_before_compression=int(n_references_before_compression),
                compression_target_reached=bool(target_references is None or reference.shape[1] <= target_references),
                cell_qc=cell_qc, gene_annotation_sha256=_fingerprint(gtf),
                selected_metadata_sha256=_fingerprint(meta),
                input_counts_fingerprint="not computed; record source h5ad checksums separately",
                units="natural-log JSD (nats); relative expression fractions",
                warning="Normal atlas origin is not proof of genomic diploidy; thresholds are heuristic.")
    diagnostics = dict(pseudobulk_qc=pb, metadata_qc=metadata_audit,
                       feature_mapping=feature_map, subtype_mapping=pd.DataFrame(type_records),
                       pair_scores=pd.DataFrame(pair_records), merge_history=pd.DataFrame(history_records),
                       profile_donor_qc=pd.DataFrame(donor_qc),
                       precompression_manifest=precompression_manifest,
                       reference_compression_map=compression_map,
                       compression_history=compression_history,
                       compression_rejected=compression_rejected,
                       run_info=info)
    log.info("Finished: %s genes x %s profiles; %s reliable pseudobulks.", *reference.shape, len(good_rows))
    result = (reference, manifest, gtf)
    return (*result, diagnostics) if return_diagnostics else result


def save_rna_reference(reference, manifest, gene_gtf, diagnostics, output_dir):
    """Write ref_hca-compatible TSV and audit tables to a NEW directory.

    ref_rna.tsv intentionally has the same implicit index header as ref_hca.tsv:
    pd.read_csv(path, sep='\\t') and pd.read_csv(path, sep='\\t', index_col=0)
    both produce a numeric gene-indexed DataFrame. No annotation columns are
    mixed into the reference matrix. Other tables have explicit headers.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    reference.rename_axis(None).to_csv(output_dir / "ref_rna.tsv", sep="\t", index_label=False)
    manifest.to_csv(output_dir / "reference_manifest.tsv", sep="\t", index=False)
    gene_gtf.to_csv(output_dir / "gene_gtf.tsv", sep="\t", index=False)
    for name, value in diagnostics.items():
        if isinstance(value, pd.DataFrame):
            # Empty audit tables still have a readable header.
            if value.shape[1] == 0:
                value = pd.DataFrame(columns=["stage", "reason"])
            value.to_csv(output_dir / f"{name}.tsv", sep="\t", index=False)
    (output_dir / "run_info.json").write_text(
        json.dumps(diagnostics["run_info"], indent=2, default=lambda x: sorted(x) if isinstance(x, set) else str(x)) + "\n")
    return output_dir / "ref_rna.tsv"


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad", help="Concatenated raw-count atlas .h5ad")
    parser.add_argument("output_dir", help="New directory for reference and diagnostics")
    parser.add_argument("--gtf", help="SpaceNumbat annotation TSV; defaults to data.hg38")
    parser.add_argument("--assays", nargs="+", default=["10x"], help="10x, smartseq2, or ALL (kept separate)")
    parser.add_argument("--assay-col", default="method")
    parser.add_argument("--cell-type-col", default="free_annotation")
    parser.add_argument("--gene-col")
    parser.add_argument("--anatomy-col")
    parser.add_argument("--layer")
    parser.add_argument("--family-json", help="Explicit label-to-family mapping replacing the defaults")
    parser.add_argument("--compression-group-col", default="broad_cell_class",
                        help="Biological gate for compacting; use NONE for conservative family only")
    parser.add_argument("--target-references", type=int, default=64,
                        help="Soft compact-reference target (default: 64)")
    parser.add_argument("--no-compress-reference", action="store_true",
                        help="Disable the default compacting stage and retain all conservative profiles")
    parser.add_argument("--compression-max-information-loss", type=float, default=0.002)
    parser.add_argument("--compression-max-information-ratio", type=float, default=2.0,
                        help="Maximum information loss relative to donor variability")
    parser.add_argument("--compression-max-js", type=float, default=0.01)
    parser.add_argument("--compression-max-pairwise-js", type=float, default=0.02,
                        help="Complete-link JSD cap across every source pair")
    parser.add_argument("--compression-max-block-log2", type=float, default=0.20)
    parser.add_argument("--same-compartment-only", action="store_true",
                        help="Disallow otherwise eligible cross-compartment family merges")
    parser.add_argument("--cross-compartment-threshold-scale", type=float, default=0.5,
                        help="Multiplier applied to all cross-compartment merge limits")
    parser.add_argument("--no-protect-curated-families", action="store_true",
                        help="Use raw broad groups in audit metadata; the family gate still applies")
    parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--min-counts", type=int, default=50_000)
    parser.add_argument("--min-donors", type=int, default=2)
    parser.add_argument("--min-compare-donors", type=int, default=3)
    parser.add_argument("--max-js", type=float, default=0.01)
    parser.add_argument("--max-block-log2", type=float, default=0.15)
    parser.add_argument("--consensus", choices=["mean", "log_median"], default="mean")
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--memmap-path", help="Optional fresh .npy file for dense pseudobulk scratch matrix")
    args = parser.parse_args()
    import anndata as ad
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if Path(args.output_dir).exists():
        parser.error("output_dir already exists; choose a new directory")
    atlas = ad.read_h5ad(args.h5ad, backed="r")
    try:
        result = build_rna_reference(
            atlas, gtf=pd.read_csv(args.gtf, sep="\t") if args.gtf else None,
            assays=None if args.assays == ["ALL"] else args.assays,
            assay_col=args.assay_col, cell_type_col=args.cell_type_col,
            gene_col=args.gene_col, anatomy_col=args.anatomy_col, layer=args.layer,
            family_mapping=json.loads(Path(args.family_json).read_text()) if args.family_json else None,
            compression_group_col=None if args.compression_group_col == "NONE" else args.compression_group_col,
            protect_curated_families=not args.no_protect_curated_families,
            target_references=None if args.no_compress_reference else args.target_references,
            compression_max_information_loss=args.compression_max_information_loss,
            compression_max_information_ratio=args.compression_max_information_ratio,
            compression_max_js=args.compression_max_js,
            compression_max_pairwise_js=args.compression_max_pairwise_js,
            compression_max_block_log2=args.compression_max_block_log2,
            compression_allow_cross_compartment=not args.same_compartment_only,
            compression_cross_compartment_scale=args.cross_compartment_threshold_scale,
            min_cells=args.min_cells, min_counts=args.min_counts, min_donors=args.min_donors,
            min_compare_donors=args.min_compare_donors, max_js=args.max_js,
            max_block_log2=args.max_block_log2, consensus=args.consensus,
            chunk_size=args.chunk_size, memmap_path=args.memmap_path, return_diagnostics=True)
        path = save_rna_reference(*result, args.output_dir)
        log.info("Saved %s", path)
    finally:
        atlas.file.close()


if __name__ == "__main__":
    _main()


