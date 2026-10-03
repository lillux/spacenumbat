#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 13:33:16 2026

@author: carlino.calogero

Donor-balanced RNA references for SpaceNumbat (Python >=3.10).

Input: AnnData cells x genes containing raw counts, including backed .X.
Output: genes x profiles, matching the in-memory layout of data.ref_hca.
Only numpy, pandas and scipy are imported; AnnData/SpaceNumbat are optional
until reading an h5ad or obtaining the default hg38 annotation, respectively.

JSD ratios are empirical compression criteria, NOT equivalence-test p-values.
Defaults need calibration on held-out normal donors and CNA-positive controls.
"""

#from __future__ import annotations

import argparse
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
__version__ = "0.1.0"
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
    by = ["assay", "compartment", "family", "label", "context", "donor"]
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
    required = [donor_col, tissue_col, cell_type_col, compartment_col, assay_col]
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
    if not _valid(meta.family).all():
        # Missing original labels are removed below; invalid explicit family values are errors.
        bad = meta.loc[_valid(obs[cell_type_col]) & ~_valid(meta.family), "label"].unique()
        if len(bad):
            raise ValueError(f"Invalid family values for: {bad.tolist()}")
    keep = np.ones(adata.n_obs, dtype=bool)
    for key in [donor_col, tissue_col, cell_type_col, compartment_col, assay_col]:
        keep &= _valid(obs[key]).to_numpy(bool)
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
    by = ["assay", "compartment", "family", "label", "context"]
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
                cell_qc=cell_qc, gene_annotation_sha256=_fingerprint(gtf),
                selected_metadata_sha256=_fingerprint(meta),
                input_counts_fingerprint="not computed; record source h5ad checksums separately",
                units="natural-log JSD (nats); relative expression fractions",
                warning="Normal atlas origin is not proof of genomic diploidy; thresholds are heuristic.")
    diagnostics = dict(pseudobulk_qc=pb, metadata_qc=metadata_audit,
                       feature_mapping=feature_map, subtype_mapping=pd.DataFrame(type_records),
                       pair_scores=pd.DataFrame(pair_records), merge_history=pd.DataFrame(history_records),
                       profile_donor_qc=pd.DataFrame(donor_qc), run_info=info)
    log.info("Finished: %s genes x %s profiles; %s reliable pseudobulks.", *reference.shape, len(good_rows))
    result = (reference, manifest, gtf)
    return (*result, diagnostics) if return_diagnostics else result


def save_rna_reference(reference, manifest, gene_gtf, diagnostics, output_dir):
    """Write ref_hca-compatible TSV and audit tables to a NEW directory.

    ref_rna.tsv has the same implicit index header as ref_hca.tsv:
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
