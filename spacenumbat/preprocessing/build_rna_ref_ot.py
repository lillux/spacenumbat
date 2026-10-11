#!/usr/bin/env python3
"""Build donor-balanced scRNA references with bounded information distortion.

Within-family subtype compression and context sharing use complete-linkage
information checks, explicit support requirements and optional cell-population
Sinkhorn divergence. Coarse parent-family fallbacks remain identifiable.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp, rel_entr
from scipy.spatial.distance import cdist


log = logging.getLogger(__name__)
__version__ = "1.1.0"


def _label(value):
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def _assay(value):
    value = _label(value)
    if value in {"10x", "10x genomics", "10x chromium", "10xgenomics"}:
        return "10x"
    if value in {"smartseq", "smartseq2", "smart-seq", "smart-seq2", "ss2"}:
        return "smartseq2"
    return value


def _valid(series):
    value = series.astype("string").str.strip()
    return value.notna() & ~value.str.casefold().isin({"", "na", "nan", "none"})


# Curated permissions to compare/merge, based on the supplied free_annotation
# vocabulary. Membership permits a statistical comparison; it never forces a
# merge. Unlisted labels remain singleton families.
_FAMILIES = {
    "CD4 T": [
        "cd4-positive, alpha-beta t cell", "activated cd4-positive, alpha-beta t cell",
        "naive thymus-derived cd4-positive, alpha-beta t cell",
        "cd4-positive, alpha-beta memory t cell", "cd4-positive memory t cell",
        "cd4-positive helper t cell", "effector cd4-positive, alpha-beta t cell",
        "cd4-positive, alpha-beta cytotoxic t cell",
    ],
    "CD8 T": [
        "cd8-positive, alpha-beta t cell", "activated cd8-positive, alpha-beta t cell",
        "naive thymus-derived cd8-positive, alpha-beta t cell", "naive cd8-positive t cell",
        "cd8-positive, alpha-beta memory t cell", "cd8-positive cytotoxic t cell",
        "effector cd8-positive, alpha-beta t cell",
        "cd8-positive, alpha-beta cytokine secreting effector t cell",
        "cd8+, alpha-beta cytokine secreting effector t cell",
    ],
    "Treg": ["regulatory t cell", "naive regulatory t cell",
             "cd4-positive, cd25-positive, alpha-beta regulatory t cell"],
    "T follicular helper": ["t follicular helper cell"],
    "Gamma-delta T": ["gamma-delta t cell"],
    "NKT": ["mature nk t cell", "nk t cell", "type i nk t cell"],
    "NK": [
        "natural killer cell", "nk cell", "mature natural killer cell",
        "immature natural killer cell", "pre-natural killer cell",
        "proliferating nk cell", "uterine nk cell", "cox6a2+ nk cell", "cd160+ nk cell",
    ],
    "B": ["b cell", "naive b cell", "memory b cell"],
    "Plasma": ["plasma cell", "antibody secreting cell"],
    "Monocyte": ["monocyte", "classical monocyte", "intermediate monocyte",
                 "non-classical monocyte"],
    "Macrophage": [
        "macrophage", "macrophage of small intestine", "colon macrophage",
        "testicular macrophage (cd163+)", "tissue-resident macrophage", "muscle macrophage",
    ],
    "Dendritic": [
        "dendritic cell", "myeloid dendritic cell", "conventional dendritic cell",
        "cdc1", "cdc2", "cd1c-positive myeloid dendritic cell",
        "cd141-positive myeloid dendritic cell", "plasmacytoid dendritic cell",
    ],
    "Neutrophil": ["neutrophil", "nampt neutrophil", "cd24 neutrophil"],
    "Mast": ["mast cell"],
    "Basophil": ["basophil"],
    "ILC": ["innate lymphoid cell", "kit+ lymphocyte"],
    "Thymocyte": ["thymocyte", "cd4-positive, alpha-beta thymocyte",
                  "cd8-positive, alpha-beta thymocyte"],
    "Hematopoietic progenitor": [
        "hematopoietic stem cell", "hematopoietic precursor cell",
        "myeloid progenitor", "common myeloid progenitor",
    ],
    "Erythroid": ["erythrocyte", "erythroid progenitor cell", "erythroid lineage cell"],
    "Platelet": ["platelet"],
    "Microglia": ["microglial cell", "retina - microglia"],
    "Langerhans": ["langerhans cell"],
    "Fibroblast": [
        "fibroblast", "fibroblast of breast", "cd34+ fibroblasts", "alveolar fibroblast",
        "adventitial fibroblast", "fibroblast of cardiac tissue",
        "thymic fibroblast type 1", "thymic fibroblast type 2",
        "endometrial stromal fibroblast", "uterine fibroblast",
        "myometrial fibroblast", "stromal fibroblasts", "hepatic stellate cell",
        "stellate_fibroblast",
    ],
    "Ovarian stroma": ["stromal cell of ovary", "theca cell"],
    "Myofibroblast": ["myofibroblast cell"],
    "Pericyte": ["pericyte", "mural cell", "adventitial cell"],
    "Smooth muscle": [
        "smooth muscle cell", "vascular associated smooth muscle cell",
        "airway smooth muscle cell", "myometrial smooth muscle", "peritubular myoid cell",
    ],
    "Vascular endothelium": [
        "endothelial cell", "vascular endothelial cell", "blood vessel endothelial cell",
        "endothelial cell of vascular tree", "cardiac endothelial cell",
        "endothelial cell of arteriole", "endothelial cell of venule",
        "capillary endothelial cell", "vein endothelial cell", "arterial endothelial cell",
        "endothelial cell of artery", "venous capillary endothelial cell",
        "capillary aerocyte", "retinal blood vessel endothelial cell", "colon endothelial cell",
    ],
    "Lymphatic endothelium": ["endothelial cell of lymphatic vessel"],
    "Mesothelial": ["mesothelial cell"],
    "Mesenchymal progenitor": ["mesenchymal stem cell",
                               "mesenchymal stem cell of adipose tissue",
                               "stromal cell", "connective tissue cell"],
    "Adipocyte": ["fat cell"],
    "Skeletal muscle": ["muscle cell", "fast muscle cell", "slow muscle cell",
                        "tongue muscle cell", "orbicularis oculi muscle"],
    "Cardiac muscle": ["atrial cardiac muscle cell", "ventricular cardiac muscle cell"],
    "Muscle satellite": ["skeletal muscle satellite stem cell"],
    "Tendon": ["tendon cell"],
    "Interstitial Cajal": ["interstitial cell of cajal"],
    "Schwann": ["schwann cell"],
    "Glial": ["glial cell", "enteroglial cell"],
    "Neuron": ["neuron"],
    "Prostate luminal": ["luminal epithelial cell", "ltf+ epithelial cell",
                         "cpe high luminal epithelial cell", "cpe low luminal epithelial cell"],
    "Prostate basal": ["basal epithelial cell", "intermediate basal-hillock epithelial cell",
                       "hillock epithelial cell", "intermediate hillock-club epithelial cell"],
    "Club": ["club cell", "pigr high club epithelial cell", "scgb3a1 high club epithelial cell"],
    "Intestinal enterocyte": [
        "enterocyte of epithelium proper of small intestine",
        "enterocyte of epithelium proper of ileum",
        "enterocyte of epithelium proper of duodenum",
        "enterocyte of epithelium proper of jejunum",
        "enterocyte of epithelium of large intestine", "mature enterocyte",
        "crypt top colonocyte", "best4+ intestinal epithelial cell",
    ],
    "Intestinal goblet": ["small intestine goblet cell", "large intestine goblet cell",
                          "goblet cell"],
    "Paneth": ["paneth cell of epithelium of small intestine", "paneth cell of colon"],
    "Intestinal stem/TA": [
        "intestinal crypt stem cell of small intestine",
        "intestinal crypt stem cell of large intestine",
        "transit amplifying cell of small intestine", "transit amplifying cell of colon",
    ],
    "Intestinal tuft": ["intestinal tuft cell", "tuft cell of colon"],
    "Enteroendocrine": ["enteroendocrine cell", "enteroendocrine cell of small intestine",
                        "enterochromaffin-like cell", "type l enteroendocrine cell"],
    "Mammary basal": ["basal cell"],
    "Mammary luminal": [
        "secretory luminal epithelial cell of mammary gland",
        "hr positive luminal epithelial cell of mammary gland",
        "progenitor cell of mammary luminal epithelium",
    ],
    "Pancreatic duct": ["pancreatic ductal cell"],
    "Pancreatic acinar": ["pancreatic acinar cell"],
    "Pancreatic alpha": ["pancreatic a cell"],
    "Pancreatic beta": ["type b pancreatic cell"],
    "Pancreatic delta": ["pancreatic d cell"],
    "Pancreatic PP": ["pancreatic pp cell"],
    "Pancreatic stellate": ["pancreatic stellate cell"],
    "Hepatocyte": ["hepatocyte"],
    "Cholangiocyte": ["biliary epithelial cell", "intrahepatic cholangiocyte"],
    "Urothelial": ["bladder urothelial cell", "basal bladder urothelial cell",
                   "intermediate bladder urothelial cell", "luminal bladder urothelial cell"],
    "Airway ciliated": ["lung ciliated cell", "ciliated columnar cell of tracheobronchial tree",
                        "ciliated epithelial cell"],
    "Airway goblet": ["respiratory goblet cell", "tracheal goblet cell", "mucus secreting cell"],
    "Airway serous": ["serous cell of epithelium of bronchus",
                      "serous cell of epithelium of trachea"],
    "Ionocyte": ["ionocyte", "pulmonary ionocyte"],
    "Alveolar type I": ["type i pneumocyte"],
    "Alveolar type II": ["type ii pneumocyte"],
    "Keratinocyte": ["krt epithelial", "stratified squamous epithelial cell",
                     "corneal and limbal epithelium"],
    "Sebocyte": ["sebum secreting cell", "sebocytes"],
    "Melanocyte": ["melanocyte"],
    "Salivary acinar": ["acinar cell of salivary gland",
                        "sero-mucinous acinar cell of salivary gland",
                        "serous acinar cell of salivary gland",
                        "mucinous acinar cell of salivary gland"],
    "Salivary duct": ["duct epithelial cell", "striated duct epithelial cell",
                      "intercalated duct cell of salivary gland",
                      "interlobular duct cell of salivary gland"],
    "Myoepithelial": ["myoepithelial cell"],
    "Uterine epithelium": ["epithelial cell of uterus", "unciliated epithelium"],
    "Thymic epithelium": ["medullary thymic epithelial cell",
                          "neuro-medullary thymic epithelial cell",
                          "myo-medullary thymic epithelial cell"],
    "Retinal photoreceptor": ["retina - photoreceptor cell", "eye photoreceptor cell"],
    "Muller glia": ["retina - muller glia", "mueller cell"],
    "Retinal neuron": ["retinal bipolar neuron", "retina - horizontal cell",
                       "retina - ganglion cell"],
    "Retinal pigment epithelium": ["retinal pigment epithelial cell"],
    "Radial glia": ["radial glia progenitor cell", "radial glial cell"],
    "Corneal epithelium": ["corneal epithelial cell", "conjunctival epithelial cell"],
    "Corneal stroma": ["keratocyte", "limbal stromal cell",
                       "cornea - mesenchymal cell - stromal keratinocytes"],
    "Lacrimal": ["lacrimal gland functional unit cell"],
    "Vestibular support": ["supporting cell of vestibular epithelium", "vestibular dark cell"],
    "Ovarian granulosa": ["granulosa cell"],
    "Ovarian epithelium": ["ovarian surface epithelial cell", "glandular epithelial cell"],
    "Cycling epithelial": ["cycling epithelial cell"],
}

FAMILY_MAPPING = {_label(cell_type): family for family, cell_types in _FAMILIES.items()
                  for cell_type in cell_types}


EXCLUDE_CELL_TYPES = {_label(x) for x in {
    "t cells-b cell-nk cell", "epithelial and mesenchymal population",
    "schwann cell or melanocyte", "melanocyte or limbal stem cell",
    "smooth muscle_pericyte", "myofibroblast cell and pericyte",
    "Monocyte_Macrophage", "theca_stroma", "follicle", "cycling cell",
    "cycling epithelial cell", "proliferating nk cell", "erythrocyte", "platelet",
    "immune cell", "myeloid cell", "hematopoietic cell", "leukocyte", "lymphocyte",
    "myeloid leukocyte", "granulocyte", "epithelial cell", "glandular cell",
    "kidney epithelial cell", "salivary gland cell", "male germ cell",
    "round spermatid", "elongating spermatid", "spermatid", "spermatocyte",
    "Mixed population of spermatocytes", "pachytene spermatocyte",
    "preleptotene spermatocytes", "leptotene_zygotene spermatocyte",
    "diplotene spermatocyte", "spermatogonium", "spermatogonial stem cell", "oocyte",
}}


def _gene_mapping(adata, gtf, gene_col):
    required = {"gene", "CHROM", "gene_start", "gene_end"}
    missing = required - set(gtf.columns)
    if missing:
        raise ValueError(f"gtf is missing {sorted(missing)}")
    gtf = gtf.copy()
    gtf["gene"] = gtf.gene.astype(str)
    gtf["CHROM"] = gtf.CHROM.astype(str).str.replace(r"^chr", "", regex=True)
    gtf = gtf[gtf.CHROM.isin({str(i) for i in range(1, 23)})].copy()
    for column in ["gene_start", "gene_end"]:
        gtf[column] = pd.to_numeric(gtf[column], errors="raise").astype(np.int64)
    gtf = gtf.drop_duplicates()
    if gtf.gene.duplicated().any():
        raise ValueError("gtf must contain one unambiguous row per gene.")
    gtf = (gtf.assign(_chrom=gtf.CHROM.astype(int))
           .sort_values(["_chrom", "gene_start", "gene_end"], kind="stable")
           .drop(columns="_chrom").reset_index(drop=True))
    aliases = {gene: gene for gene in gtf.gene}
    if "gene_ensembl" in gtf:
        ensembl = gtf.gene_ensembl.astype("string").str.replace(r"\.\d+$", "", regex=True)
        for identifier, symbol in zip(ensembl, gtf.gene):
            if pd.notna(identifier):
                aliases[str(identifier)] = symbol
    if gene_col is not None and gene_col not in adata.var:
        raise ValueError(f"adata.var lacks gene_col={gene_col!r}")
    source = (pd.Series(adata.var_names.astype(str), dtype="string") if gene_col is None
              else adata.var[gene_col].astype("string").reset_index(drop=True))
    source = source.str.replace(r"^(ENSG\d+)\.\d+$", r"\1", regex=True)
    mapped = source.map(aliases)
    feature_map = pd.DataFrame({"input_feature": np.asarray(adata.var_names, str),
                                "input_identifier": source, "gene": mapped})
    feature_map["status"] = np.where(mapped.notna(), "mapped", "unmapped_or_non_autosomal")
    feature_map["n_input_columns_for_gene"] = mapped.map(mapped.value_counts()).fillna(0).astype(int)
    gtf = gtf[gtf.gene.isin(mapped.dropna())].reset_index(drop=True)
    if len(gtf) < 2:
        raise ValueError("Fewer than two autosomal genes matched the reference annotation.")
    rows = np.flatnonzero(mapped.notna())
    cols = pd.Index(gtf.gene).get_indexer(mapped.iloc[rows])
    projection = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)),
                                   shape=(adata.n_vars, len(gtf)))
    return gtf, projection, feature_map


def _prob(counts, prior_mass):
    counts = np.asarray(counts, dtype=np.float64)
    depth = counts.sum(axis=1, keepdims=True)
    if np.any(depth <= 0):
        raise ValueError("A pseudobulk contains zero mapped counts.")
    return (counts + prior_mass / counts.shape[1]) / (depth + prior_mass)


def _kl(p, q):
    return float(rel_entr(p, q).sum())


def _js(p, q, weight=0.5):
    p = np.asarray(p, float); p /= p.sum()
    q = np.asarray(q, float); q /= q.sum()
    center = weight * p + (1.0 - weight) * q
    return max(0.0, weight * _kl(p, center) + (1.0 - weight) * _kl(q, center))


def _generalized_js(profiles, weights=None):
    profiles = np.asarray(profiles, float)
    profiles /= profiles.sum(axis=1, keepdims=True)
    n_profiles = profiles.shape[0]
    weights = (np.full(n_profiles, 1 / n_profiles) if weights is None
               else np.asarray(weights, float))
    weights /= weights.sum()
    center = np.sum(weights[:, None] * profiles, axis=0)
    return float(sum(w * _kl(p, center) for w, p in zip(weights, profiles)))


def _pairwise_js(profiles):
    return [_js(profiles[i], profiles[j])
            for i, j in itertools.combinations(range(len(profiles)), 2)]


def _aggregate_rows(counts, obs, by):
    keys = pd.MultiIndex.from_frame(obs[by].astype("string"))
    codes, groups = pd.factorize(keys, sort=True)
    group_matrix = sparse.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))),
                                     shape=(len(groups), len(codes)))
    aggregated = np.asarray(group_matrix @ np.asarray(counts, float))
    result = groups.to_frame(index=False); result.columns = by
    weights = obs.n_cells.to_numpy(float) if "n_cells" in obs else np.ones(len(obs))
    result["n_cells"] = np.bincount(codes, weights=weights).astype(int)
    result["n_counts"] = aggregated.sum(axis=1)
    return aggregated, result


def _aggregate_counts(adata, metadata, projection, *, layer, chunk_size,
                      min_cell_counts, min_cell_genes, max_pct_counts_mt, memmap_path,
                      cell_reservoir=None, sinkhorn_max_cells=64, random_state=0,
                      cell_embedding=None):
    by = ["assay", "context", "family", "label", "donor"]
    selected = metadata._keep.to_numpy(bool)
    keys = pd.MultiIndex.from_frame(metadata.loc[selected, by])
    selected_codes, groups = pd.factorize(keys, sort=True)
    pb_obs = groups.to_frame(index=False); pb_obs.columns = by
    codes = np.full(adata.n_obs, -1, dtype=np.int64); codes[selected] = selected_codes
    shape = (len(groups), projection.shape[1])
    if memmap_path is None:
        counts = np.zeros(shape, dtype=np.float64)
    else:
        memmap_path = Path(memmap_path)
        if memmap_path.exists():
            raise FileExistsError(f"Refusing to overwrite {memmap_path}")
        memmap_path.parent.mkdir(parents=True, exist_ok=True)
        counts = np.lib.format.open_memmap(memmap_path, mode="w+", dtype=np.float64, shape=shape)
        counts[:] = 0
    n_cells = np.zeros(len(groups), dtype=np.int64)
    qc = {"metadata_selected": int(selected.sum()), "passed_count_qc": 0,
          "failed_count_qc": 0}
    matrix = adata.X if layer is None else adata.layers[layer]
    rng = np.random.default_rng(random_state)
    for start in range(0, adata.n_obs, chunk_size):
        end = min(start + chunk_size, adata.n_obs)
        use = codes[start:end] >= 0
        if not use.any():
            continue
        x = sparse.csr_matrix(matrix[start:end, :])[use].astype(np.float64)
        if x.data.size and (not np.isfinite(x.data).all() or np.any(x.data < 0)
                            or np.any(np.abs(x.data - np.rint(x.data)) > 1e-6)):
            raise ValueError("Selected expression values must be finite nonnegative raw integers.")
        x = (x @ projection).tocsr()
        total = np.asarray(x.sum(axis=1)).ravel()
        good = (total >= min_cell_counts) & (x.getnnz(axis=1) >= min_cell_genes) & (total > 0)
        if max_pct_counts_mt is not None:
            mt = pd.to_numeric(adata.obs.iloc[start:end].loc[use, "pct_counts_mt"],
                               errors="coerce").to_numpy()
            good &= np.isfinite(mt) & (mt <= max_pct_counts_mt)
        qc["passed_count_qc"] += int(good.sum()); qc["failed_count_qc"] += int((~good).sum())
        batch_codes = codes[start:end][use][good]
        if not len(batch_codes):
            continue
        active, inverse = np.unique(batch_codes, return_inverse=True)
        grouping = sparse.csr_matrix((np.ones(len(inverse)), (inverse, np.arange(len(inverse)))),
                                     shape=(len(active), len(inverse)))
        counts[active] += (grouping @ x[good]).toarray()
        n_cells[active] += np.bincount(inverse, minlength=len(active))
        if cell_reservoir is not None:
            # Lowest random priorities give a uniform bounded sample of all
            # QC-passing cells, independent of chunk_size and expression depth.
            indices = start + np.flatnonzero(use)[good]
            if not np.isfinite(cell_embedding[indices]).all():
                raise ValueError("QC-passing cells have nonfinite embedding coordinates.")
            priorities = rng.random(len(indices))
            for code in active:
                take = batch_codes == code
                old_indices, old_priorities = cell_reservoir.get(
                    int(code), (np.empty(0, dtype=int), np.empty(0)))
                ix = np.concatenate([old_indices, indices[take]])
                pr = np.concatenate([old_priorities, priorities[take]])
                order = np.argsort(pr, kind="stable")[:sinkhorn_max_cells]
                cell_reservoir[int(code)] = (ix[order], pr[order])
    pb_obs["n_cells"] = n_cells; pb_obs["n_counts"] = np.asarray(counts.sum(axis=1)).ravel()
    return counts, pb_obs, qc


def _donor_profiles(counts, obs, prior_mass):
    donor_counts, donor_obs = _aggregate_rows(counts, obs.reset_index(drop=True), ["donor"])
    return _prob(donor_counts, prior_mass), donor_obs


def _global_donor_noise(counts, obs, prior_mass, noise_floor):
    values = []
    for _, idx in obs.groupby(["label", "context"], sort=False).indices.items():
        profiles, _ = _donor_profiles(counts[np.asarray(idx)], obs.iloc[idx], prior_mass)
        values.extend(_pairwise_js(profiles))
    return max(float(np.median(values)), noise_floor) if values else noise_floor


def _within_context_donor_noise(counts, obs, prior_mass):
    values = []
    for _, idx in obs.groupby("context", sort=False).indices.items():
        profiles, _ = _donor_profiles(counts[np.asarray(idx)], obs.iloc[idx], prior_mass)
        values.extend(_pairwise_js(profiles))
    return float(np.median(values)) if values else np.nan


def _sinkhorn_divergence(x, y, a=None, b=None, *, epsilon, cost_scale,
                         max_iter=1000, tolerance=1e-7):
    """Debiased balanced OT on point clouds; failures never imply equivalence.

    Cost is mean squared coordinate distance / cost_scale**2. Regularization
    uses KL(plan || a*b); this convention is used for cross and self costs.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    a = np.full(len(x), 1 / len(x)) if a is None else np.asarray(a, float)
    b = np.full(len(y), 1 / len(y)) if b is None else np.asarray(b, float)
    a, b = a / a.sum(), b / b.sum()

    def objective(u, v, wa, wb, symmetric=False):
        cost = cdist(u, v, metric="sqeuclidean") / (u.shape[1] * cost_scale**2)
        loga, logb = np.log(wa), np.log(wb)
        kernel = -cost / epsilon
        logv = np.zeros(len(v))
        for iteration in range(max_iter):
            if symmetric:
                # Symmetric scaling avoids slow alternating convergence for
                # nearly disconnected self-cost kernels at small epsilon.
                logv = 0.5 * (logv + loga - logsumexp(kernel + logv[None, :], axis=1))
                logu = logv
            else:
                logu = loga - logsumexp(kernel + logv[None, :], axis=1)
                logv = logb - logsumexp(kernel + logu[:, None], axis=0)
            if iteration % 10 == 0 or iteration == max_iter - 1:
                logplan = kernel + logu[:, None] + logv[None, :]
                row = np.exp(logsumexp(logplan, axis=1))
                col = np.exp(logsumexp(logplan, axis=0))
                if max(np.max(np.abs(row - wa)), np.max(np.abs(col - wb))) <= tolerance:
                    plan = np.exp(logplan)
                    entropy = np.sum(plan * (logplan - loga[:, None] - logb[None, :]))
                    return float(np.sum(plan * cost) + epsilon * entropy)
        return np.nan

    cross = objective(x, y, a, b)
    xx = objective(x, x, a, a, symmetric=True)
    yy = objective(y, y, b, b, symmetric=True)
    value = cross - 0.5 * (xx + yy)
    if not np.isfinite(value) or value < -10 * tolerance:
        return np.nan
    return max(0.0, float(value))


class _CellTransport:
    """Cached comparisons of original label/context populations by donor."""

    def __init__(self, embedding, samples, obs, *, epsilon, cost_scale,
                 max_iter, tolerance, min_donors):
        self.embedding, self.samples = embedding, samples
        self.obs = obs.reset_index(drop=True)
        self.kwargs = dict(epsilon=epsilon, cost_scale=cost_scale,
                           max_iter=max_iter, tolerance=tolerance)
        self.min_donors, self.cache = min_donors, {}

    def compare(self, left, right):
        # Each selector is (frozenset(labels), frozenset(contexts)).
        key = tuple(sorted((tuple(map(tuple, map(sorted, left))),
                            tuple(map(tuple, map(sorted, right))))))
        if key in self.cache:
            return self.cache[key]

        def population(selector):
            labels, contexts = selector
            mask = self.obs.label.isin(labels) & self.obs.context.isin(contexts)
            result = {}
            for donor, rows in self.obs.loc[mask].groupby("donor", sort=True):
                blocks = [self.samples[tuple(row)] for row in
                          rows[["assay", "context", "family", "label", "donor"]]
                          .itertuples(index=False, name=None)
                          if tuple(row) in self.samples]
                if blocks:
                    # Equal label/context blocks within donor, equal donors
                    # in an unmatched-donor population comparison.
                    points = np.concatenate(blocks)
                    weights = np.concatenate([np.full(len(x), 1 / (len(blocks) * len(x)))
                                              for x in blocks])
                    result[str(donor)] = (np.asarray(self.embedding[points], float), weights)
            return result

        pa, pb = population(left), population(right)
        if min(len(pa), len(pb)) < self.min_donors:
            result = (np.nan, "insufficient_sinkhorn_donors")
        else:
            shared = sorted(set(pa) & set(pb))
            if len(shared) >= self.min_donors:
                values = [_sinkhorn_divergence(*[pa[d][0], pb[d][0], pa[d][1], pb[d][1]],
                                               **self.kwargs) for d in shared]
                # Worst matched donor protects donor-specific discrepancies.
                result = (float(np.max(values)), "matched_donors")
            else:
                def pooled(pop):
                    points = np.concatenate([x for x, _ in pop.values()])
                    weights = np.concatenate([w / len(pop) for _, w in pop.values()])
                    return points, weights
                xa, wa = pooled(pa); xb, wb = pooled(pb)
                result = (_sinkhorn_divergence(xa, xb, wa, wb, **self.kwargs),
                          "unmatched_donor_balanced")
        self.cache[key] = result
        return result


def _profile_distortion(counts, obs, *, units, prior_mass, eps, reference=None):
    """Equal-unit information loss and worst-unit loss to the actual profile."""
    positive = np.asarray(counts.sum(axis=1)).ravel() > 0
    counts = counts[positive]; obs = obs.loc[positive].reset_index(drop=True)
    profiles, baseline_losses = [], []
    for _, idx in obs.groupby(units, sort=True).indices.items():
        p, _ = _donor_profiles(counts[np.asarray(idx)], obs.iloc[idx], prior_mass)
        center = p.mean(axis=0)
        own = np.exp(np.median(np.log(p + eps), axis=0)); own /= own.sum()
        profiles.append(center); baseline_losses.append(_kl(center, own))
    if not profiles:
        return dict(information_loss=np.inf, max_pair_js=np.inf,
                    max_profile_kl=np.inf, max_excess_kl=np.inf)
    profiles = np.vstack(profiles)
    if reference is None:
        reference, _ = _robust_donor_consensus(counts, obs, prior_mass=prior_mass, eps=eps)
    return {
        "information_loss": _generalized_js(profiles),
        "max_pair_js": max(_pairwise_js(profiles), default=0.0),
        "max_profile_kl": max(_kl(p, reference) for p in profiles),
        "max_excess_kl": max(max(0.0, _kl(p, reference) - base)
                             for p, base in zip(profiles, baseline_losses)),
    }


def _distortion_allowed(score, max_merge_js, max_merge_kl):
    return (score["information_loss"] <= max_merge_js and
            score["max_pair_js"] <= max_merge_js and
            score["max_excess_kl"] <= max_merge_kl)


def _supported_sources(counts, obs, *, min_cells, min_counts):
    """Keep source rows whose pooled donor/context group passes count QC."""
    pooled, grouped = _aggregate_rows(counts, obs, ["context", "donor"])
    good = (grouped.n_cells >= min_cells) & (grouped.n_counts >= min_counts)
    keys = pd.MultiIndex.from_frame(grouped.loc[good, ["context", "donor"]])
    mask = pd.MultiIndex.from_frame(obs[["context", "donor"]]).isin(keys)
    return counts[mask], obs.loc[mask].reset_index(drop=True)


def _subtype_pair_score(counts, obs, cluster_a, cluster_b, *, prior_mass,
                        fallback_noise, noise_floor, min_donors=2,
                        unshared_context_policy="defer", transport=None):
    mask_a = obs.label.isin(cluster_a); mask_b = obs.label.isin(cluster_b)
    contexts_a = set(obs.loc[mask_a, "context"])
    contexts_b = set(obs.loc[mask_b, "context"])
    shared = sorted(contexts_a & contexts_b)
    rejected = {"between_js": np.nan, "donor_noise": fallback_noise,
                "ratio": np.inf, "shared_contexts": len(shared),
                "sinkhorn": np.nan, "sinkhorn_mode": "disabled"}
    if not shared and unshared_context_policy == "defer":
        return {**rejected, "reason": "no_shared_context_deferred"}
    comparisons = [(mask_a & obs.context.eq(t), mask_b & obs.context.eq(t), {t}, {t})
                   for t in shared]
    if not shared:
        comparisons = [(mask_a, mask_b, contexts_a, contexts_b)]
    between, noises, ratios, ot_values, ot_modes = [], [], [], [], []
    for a, b, ca, cb in comparisons:
        pa, _ = _donor_profiles(counts[a.to_numpy()], obs.loc[a], prior_mass)
        pb, _ = _donor_profiles(counts[b.to_numpy()], obs.loc[b], prior_mass)
        if min(len(pa), len(pb)) < min_donors:
            return {**rejected, "reason": "insufficient_comparison_donors"}
        # Equal subtype priors prevent poor sampling from suppressing a contrast.
        js = _js(pa.mean(axis=0), pb.mean(axis=0))
        variation = _pairwise_js(pa) + _pairwise_js(pb)
        noise = max(float(np.median(variation)) if variation else fallback_noise,
                    noise_floor)
        between.append(js); noises.append(noise); ratios.append(js / noise)
        if transport is not None:
            value, mode = transport.compare((cluster_a, ca), (cluster_b, cb))
            if not np.isfinite(value):
                return {**rejected, "reason": "sinkhorn_unresolved", "sinkhorn_mode": mode}
            ot_values.append(value); ot_modes.append(mode)
    return {"between_js": max(between), "donor_noise": float(np.median(noises)),
            "ratio": max(ratios), "shared_contexts": len(shared),
            "reason": "shared_context" if shared else "unshared_context_compared",
            "sinkhorn": max(ot_values) if ot_values else np.nan,
            "sinkhorn_mode": json.dumps(sorted(set(ot_modes))) if ot_modes else "disabled"}


def _agglomerate_family(counts, obs, labels, family, *, prior_mass,
                        global_noise, ratio_threshold, noise_floor,
                        source_counts, source_obs, min_cells, min_counts,
                        min_donors, max_merge_js, max_merge_kl, eps,
                        unshared_context_policy, transport, sinkhorn_threshold):
    clusters = [frozenset([label]) for label in sorted(labels)]
    history, pair_cache, assessed = [], {}, set()
    while len(clusters) > 1:
        candidates = []
        for i, j in itertools.combinations(range(len(clusters)), 2):
            members = clusters[i] | clusters[j]
            atomic = []
            for a, b in itertools.combinations(sorted(members), 2):
                if (a, b) not in pair_cache:
                    pair_cache[a, b] = _subtype_pair_score(
                        counts, obs, {a}, {b}, prior_mass=prior_mass,
                        fallback_noise=global_noise, noise_floor=noise_floor,
                        min_donors=min_donors, unshared_context_policy=unshared_context_policy,
                        transport=transport)
                atomic.append(pair_cache[a, b])
            worst = max(atomic, key=lambda x: x["ratio"])
            score = dict(worst)
            score["between_js"] = max((x["between_js"] for x in atomic
                                       if np.isfinite(x["between_js"])), default=np.nan)
            score["sinkhorn"] = max((x["sinkhorn"] for x in atomic
                                     if np.isfinite(x["sinkhorn"])), default=np.nan)
            reason = None
            if not np.isfinite(score["ratio"]):
                reason = worst["reason"]
            elif score["ratio"] > ratio_threshold:
                reason = "subtype_ratio_exceeded"
            elif score["between_js"] > max_merge_js:
                reason = "pairwise_js_exceeded"
            elif transport is not None and score["sinkhorn"] > sinkhorn_threshold:
                reason = "sinkhorn_exceeded"
            if reason is None:
                take = source_obs.label.isin(members).to_numpy()
                sc, so = _supported_sources(source_counts[take], source_obs.loc[take],
                                             min_cells=min_cells, min_counts=min_counts)
                distortions = []
                for _, idx in so.groupby("context", sort=True).indices.items():
                    distortions.append(_profile_distortion(
                        sc[np.asarray(idx)], so.iloc[idx].reset_index(drop=True),
                        units=["label"], prior_mass=prior_mass, eps=eps))
                if any(x["reason"] == "unshared_context_compared" for x in atomic):
                    distortions.append(_profile_distortion(
                        sc, so, units=["label", "context"], prior_mass=prior_mass, eps=eps))
                score.update({name: max((x[name] for x in distortions), default=np.inf)
                              for name in ("information_loss", "max_pair_js",
                                           "max_profile_kl", "max_excess_kl")})
                if not _distortion_allowed(score, max_merge_js, max_merge_kl):
                    reason = "profile_distortion_exceeded"
            decision = {"family": family, "left": json.dumps(sorted(clusters[i])),
                        "right": json.dumps(sorted(clusters[j])),
                        "merged": json.dumps(sorted(members)), **score,
                        "admissible": reason is None, "accepted": False, "selected": False,
                        "decision_reason": reason or "within_bounds"}
            proposal = tuple(sorted((tuple(sorted(clusters[i])), tuple(sorted(clusters[j])))))
            if proposal not in assessed:
                history.append(decision); assessed.add(proposal)
            if reason is None:
                candidates.append((score["ratio"], i, j))
        if not candidates:
            break
        _, i, j = min(candidates)
        members = clusters[i] | clusters[j]
        # 'accepted' records an actual merge, not merely an admissible proposal.
        selected = tuple(sorted((tuple(sorted(clusters[i])), tuple(sorted(clusters[j])))))
        for row in history:
            proposal = tuple(sorted((tuple(json.loads(row["left"])), tuple(json.loads(row["right"])))))
            if proposal == selected:
                row["selected"] = True; row["accepted"] = True
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [members]
    return sorted(clusters, key=lambda cluster: sorted(cluster)), history


def _test_context_information(counts, obs, *, min_context_donors, prior_mass,
                              global_noise, ratio_threshold, noise_floor,
                              source_counts, source_obs, min_donors, max_merge_js,
                              max_merge_kl, eps, labels, transport, sinkhorn_threshold):
    profiles, eligible, noise_values, donor_profiles = [], [], [], {}
    for context, idx in obs.groupby("context", sort=True).indices.items():
        p, donors = _donor_profiles(counts[np.asarray(idx)], obs.iloc[idx], prior_mass)
        if len(donors) < max(min_context_donors, min_donors):
            continue
        context = str(context)
        eligible.append(context); profiles.append(p.mean(axis=0))
        donor_profiles[context] = p
        noise_values.extend(_pairwise_js(p))
    js = _generalized_js(np.vstack(profiles)) if profiles else np.nan
    noise = max(float(np.median(noise_values)) if noise_values else global_noise,
                global_noise, noise_floor)
    groups = [frozenset([x]) for x in eligible]
    pair_cache, history, assessed = {}, [], set()
    while len(groups) > 1:
        candidates = []
        for i, j in itertools.combinations(range(len(groups)), 2):
            members = groups[i] | groups[j]
            pairs = []
            for a, b in itertools.combinations(sorted(members), 2):
                if (a, b) not in pair_cache:
                    pa, pb = donor_profiles[a], donor_profiles[b]
                    variation = _pairwise_js(pa) + _pairwise_js(pb)
                    local_noise = max(float(np.median(variation)) if variation else global_noise,
                                      global_noise, noise_floor)
                    distance = _js(pa.mean(axis=0), pb.mean(axis=0))
                    ot_value, ot_mode = (transport.compare((labels, {a}), (labels, {b}))
                                         if transport is not None else (np.nan, "disabled"))
                    pair_cache[a, b] = (distance, distance / local_noise, ot_value, ot_mode)
                pairs.append(pair_cache[a, b])
            ratio = max(x[1] for x in pairs)
            sc_mask = source_obs.context.isin(members).to_numpy()
            distortion = _profile_distortion(source_counts[sc_mask], source_obs.loc[sc_mask],
                                               units=["label", "context"],
                                               prior_mass=prior_mass, eps=eps)
            reason = None
            if ratio > ratio_threshold:
                reason = "context_ratio_exceeded"
            elif not _distortion_allowed(distortion, max_merge_js, max_merge_kl):
                reason = "profile_distortion_exceeded"
            elif transport is not None and any(not np.isfinite(x[2]) for x in pairs):
                reason = "sinkhorn_unresolved"
            elif transport is not None and max(x[2] for x in pairs) > sinkhorn_threshold:
                reason = "sinkhorn_exceeded"
            proposal = tuple(sorted((tuple(sorted(groups[i])), tuple(sorted(groups[j])))))
            if proposal not in assessed:
                history.append({"left": json.dumps(sorted(groups[i])),
                                "right": json.dumps(sorted(groups[j])),
                                "merged": json.dumps(sorted(members)), "ratio": ratio,
                                "sinkhorn": max(x[2] for x in pairs),
                                "sinkhorn_mode": json.dumps(sorted({x[3] for x in pairs})),
                                **distortion, "admissible": reason is None,
                                "decision_reason": reason or "within_bounds", "selected": False})
                assessed.add(proposal)
            if reason is None:
                candidates.append((ratio, i, j))
        if not candidates:
            break
        _, i, j = min(candidates)
        selected = tuple(sorted((tuple(sorted(groups[i])), tuple(sorted(groups[j])))))
        for row in history:
            if tuple(sorted((tuple(json.loads(row["left"])), tuple(json.loads(row["right"]))))) == selected:
                row["selected"] = True
        groups = [g for k, g in enumerate(groups) if k not in (i, j)] + [groups[i] | groups[j]]
    return {"split": len(groups) > 1, "context_js": js, "donor_noise": noise,
            "ratio": js / noise, "eligible_contexts": eligible,
            "groups": sorted([sorted(x) for x in groups]),
            "deferred_contexts": sorted(set(obs.context.astype(str)) - set(eligible)),
            "history": history}


def _robust_donor_consensus(counts, obs, *, prior_mass, eps):
    profiles, donor_obs = _donor_profiles(counts, obs, prior_mass)
    reference = np.exp(np.median(np.log(profiles + eps), axis=0)); reference /= reference.sum()
    return reference, donor_obs.donor.astype(str).tolist()


def build_rna_reference(
    adata, gtf=None, *, donor_col="donor", tissue_col="tissue",
    cell_type_col="free_annotation", assay_col="method", assays=("10x",),
    anatomy_col=None, gene_col=None, layer=None, family_mapping=None,
    exclude_cell_types=None, min_cells=50, min_counts=50_000, min_donors=2,
    min_context_donors=2, subtype_ratio_threshold=1.5, tissue_ratio_threshold=2.0,
    noise_floor=1e-8, prior_mass=1.0, eps=1e-12, min_cell_counts=100,
    min_cell_genes=100, max_pct_counts_mt=None, chunk_size=8192,
    memmap_path=None, return_diagnostics=False,
    max_merge_js=0.02, max_merge_kl=0.05, unshared_context_policy="defer",
    fallback_mode="missing", sinkhorn_embedding=None, sinkhorn_threshold=0.05,
    sinkhorn_epsilon=0.05, sinkhorn_cost_scale=1.0, sinkhorn_max_cells=64,
    sinkhorn_max_iter=1000, sinkhorn_tolerance=1e-7, random_state=0,
):
    """Build diploid RNA reference profiles with information tests.

    The function converts raw single-cell RNA counts into a gene-by-reference
    probability matrix suitable for the RNA mode of SpaceNumbat.  Donors are
    treated as the independent biological replicates.  Within each assay, the
    procedure first merges statistically redundant annotations *only within a
    curated biological family*, then decides whether each merged cell type
    requires tissue/anatomy-specific reference profiles.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-by-gene atlas. ``adata.X`` (or ``adata.layers[layer]``) must contain
        finite, non-negative, integer-valued raw counts; normalized or
        log-transformed expression is not accepted. Cell and feature identifiers
        must be unique. The object is not modified.
    gtf : pandas.DataFrame, optional
        Gene annotation with columns ``gene``, ``CHROM``, ``gene_start``, and
        ``gene_end``. A ``gene_ensembl`` column is used when present. If omitted,
        ``spacenumbat.data.hg38`` is used. Only autosomal genes (chromosomes
        1--22) represented in both ``adata`` and the annotation are retained.
    donor_col : str, default: "donor"
        Column in ``adata.obs`` identifying biological donors. Donors, rather
        than cells, define replicate profiles and the information-noise scale.
    tissue_col : str, default: "tissue"
        Column in ``adata.obs`` containing tissue labels.
    cell_type_col : str, default: "free_annotation"
        Column in ``adata.obs`` containing the input cell-type annotation to be
        compressed. Labels are normalized to lower-case, whitespace-collapsed
        strings before mapping and comparison.
    assay_col : str, default: "method"
        Column in ``adata.obs`` containing the library protocol. Assays are
        analyzed independently so that protocol effects cannot drive merges.
        Common 10x and Smart-seq spellings are normalized to ``"10x"`` and
        ``"smartseq2"``, respectively.
    assays : sequence of str or None, default: ("10x",)
        Assays to retain after protocol-name normalization. Pass ``None`` to use
        every assay present. Profiles from different assays are never pooled.
    anatomy_col : str or None, default: None
        Optional ``adata.obs`` column defining anatomy within tissue. If set,
        contexts are ``"<tissue>::<anatomy>"``; missing anatomy is encoded as
        ``"NA"``. If ``None``, tissue alone defines context.
    gene_col : str or None, default: None
        Column in ``adata.var`` containing gene symbols or Ensembl identifiers.
        By default, ``adata.var_names`` is used. Version suffixes are removed
        from Ensembl identifiers, and multiple input features mapping to the
        same gene are summed.
    layer : str or None, default: None
        AnnData layer containing raw counts. ``None`` uses ``adata.X``.
    family_mapping : mapping of str to str or None, default: None
        Additional or replacement label-to-family assignments, keyed by the
        normalized values of ``cell_type_col``. These entries update the bundled
        biological-family map. Labels without a mapping form singleton families;
        therefore they cannot merge with biologically unrelated labels.
    exclude_cell_types : iterable of str or None, default: None
        Extra normalized labels to discard in addition to the bundled exclusions
        for composite, overly broad, cycling, germline/meiotic, and anucleate
        annotations. This argument extends rather than replaces the defaults.
    min_cells : int, default: 50
        Minimum cells required in each donor-by-context-by-cell-type pseudobulk.
        Applied before subtype merging and again after re-aggregation.
    min_counts : int, default: 50000
        Minimum mapped autosomal counts required per pseudobulk, before and after
        subtype merging.
    min_donors : int, default: 2
        Minimum distinct donors required for an emitted reference profile.
    min_context_donors : int, default: 2
        Minimum donors required for a context to participate in the context
        grouping step. A context must have at least max(min_context_donors,
        min_donors) donors. Insufficiently supported contexts remain unresolved
        and can use a separately identified family fallback.
    subtype_ratio_threshold : float, default: 1.5
        Maximum subtype information-to-noise ratio accepted for agglomerative
        merging within a biological family. Equal-prior pairwise JS is divided
        by within-label donor JS. Every original label pair and shared context
        must pass; scores are never diluted by averaging merged annotations.
        This is an empirical decision statistic, not an equivalence test.
    tissue_ratio_threshold : float, default: 2.0
        Maximum pairwise context JS / donor-noise ratio for pooling contexts.
        Complete-linkage grouping allows similar contexts to share a column
        while keeping dissimilar contexts separate. Absolute distortion bounds
        and the optional Sinkhorn check must also pass.
    noise_floor : float, default: 1e-8
        Positive lower bound on the donor-noise denominator. This prevents
        unstable ratios when donor profiles are nearly identical.
    prior_mass : float, default: 1.0
        Total mass of the symmetric Dirichlet prior used when converting donor
        pseudobulks to gene probabilities. It prevents zero probabilities while
        remaining independent of the number of retained genes.
    eps : float, default: 1e-12
        Positive numerical offset used by the log-space robust donor consensus.
    min_cell_counts : int, default: 100
        Minimum mapped autosomal raw counts required for an individual cell.
    min_cell_genes : int, default: 100
        Minimum number of detected mapped autosomal genes required per cell.
    max_pct_counts_mt : float or None, default: None
        Optional upper bound for ``adata.obs["pct_counts_mt"]``. No mitochondrial
        percentage filter is applied when ``None``.
    chunk_size : int, default: 8192
        Number of cells processed per block while mapping features and building
        pseudobulks. Lower this value to reduce peak working memory.
    memmap_path : path-like or None, default: None
        Optional path for the dense float64 pseudobulk-by-gene work matrix. The
        path must not already exist and is not a resumable cache. If ``None``,
        the matrix is held in memory.
    return_diagnostics : bool, default: False
        If ``True``, append a diagnostics dictionary to the returned tuple.

    max_merge_js : float, default: 0.02
        Absolute bound in nats on equal-unit generalized JS information loss
        and on every original-unit pairwise JS. Subtype guards are evaluated
        within contexts; context guards use original label-by-context units.
        Equal-unit weights protect rare annotations from abundance dilution.
    max_merge_kl : float, default: 0.05
        Maximum additional KL loss for any original unit when using the actual
        proposed robust reference instead of its own robust donor reference:
        max(0, KL(unit_mean || pooled_reference) - KL(unit_mean || own_reference)).
        Together with max_merge_js this prevents high donor noise from allowing
        large signal distortion. Both bounds are effect-size tolerances, not
        significance levels or validated universal biological defaults.
    unshared_context_policy : {"defer", "compare"}, default: "defer"
        Defer pairs without shared contexts. "compare" permits a measured
        comparison of donor-balanced pooled profiles and, when enabled, cell
        populations, with the same ratio and absolute distortion checks.
        These contrasts remain confounded with context and are identified as
        "unshared_context_compared" in diagnostics; no zero-score shortcut exists.
    fallback_mode : {"missing", "separate", "off"}, default: "missing"
        Preserve QC-supported parent-family profiles in diagnostics. "missing"
        also inserts a parent fallback into the returned reference when any
        family label/context lacks a supported primary profile. "separate"
        keeps fallbacks outside the primary matrix; "off" disables them.
        Fallbacks are coarse estimates, may exceed distortion tolerances, and
        are explicitly marked. Active parents may overlap supported children;
        use "separate" for a nonredundant primary mixture basis.
    sinkhorn_embedding : str, array-like, or None, default: None
        Common, dense cell-state representation with shape (adata.n_obs, d),
        or an adata.obsm key. None disables the check. Provide a count-aware
        PCA/latent representation fitted in a common coordinate system;
        coordinates must be finite for all QC-passing cells. The builder does
        not learn, standardize, integrate, or modify this embedding. Distinct
        assays are compared independently. The full gene set still defines the
        reference profiles. A representation hash is recorded for replication.
    sinkhorn_threshold : float, default: 0.05
        Maximum debiased balanced Sinkhorn divergence for every constituent
        population pair. Cost units are mean squared coordinate distance divided
        by sinkhorn_cost_scale**2. This is an additional veto, never a substitute
        for gene information checks. It is active only with an embedding.
    sinkhorn_epsilon : float, default: 0.05
        Positive entropy regularization in the same units as the ground cost.
        Cross and self costs use KL(plan || product_of_marginals).
    sinkhorn_cost_scale : float, default: 1.0
        Fixed global coordinate scale; do not normalize each comparison
        separately. Thresholds and epsilon must be calibrated to the embedding.
    sinkhorn_max_cells : int, default: 64
        Uniform reservoir sample cap per original donor/context/label/assay
        pseudobulk. Only cells passing the exact count-QC pass are sampled.
        Matched donors are used when at least min_donors overlap; otherwise
        cell populations are compared with equal donor mass. Pooled blocks
        within a donor have equal mass. The worst matched-donor distance is used.
    sinkhorn_max_iter : int, default: 1000
        Maximum log-domain Sinkhorn iterations per cross or self problem.
        Nonconvergence or missing cell support defers a merge.
    sinkhorn_tolerance : float, default: 1e-7
        Maximum absolute marginal residual accepted by the transport solver.
    random_state : int, default: 0
        Nonnegative seed for reproducible cell reservoir sampling.

    Returns
    -------
    reference : pandas.DataFrame
        Gene-by-reference matrix. Columns are strictly positive transcriptional
        probability profiles and each column sums to one. This is the reference
        matrix passed to the RNA SpaceNNumBAT pipeline.
    manifest : pandas.DataFrame
        One row per reference column, including the emitted reference name,
        merged cell type, biological family, assay, scope (``"pan_context"`` or
        ``"context"``), contributing contexts/tissues/donors, support counts,
        source annotations, and context information statistics.
    gene_gtf : pandas.DataFrame
        Annotation for the expressed autosomal genes in ``reference``, ordered
        genomically and suitable for downstream RNA CNA inference.
    diagnostics : dict, optional
        Returned only when ``return_diagnostics=True``. Contains
        ``cell_type_map``, ``merge_history``, ``context_tests``,
        ``initial_pseudobulk_qc``, ``merged_pseudobulk_qc``,
        ``global_donor_noise``, ``feature_mapping``, and ``run_info``. Additional
        tables are ``merge_decisions`` (including rejected/deferred proposals),
        ``context_merge_decisions``, ``unresolved_units``,
        ``fallback_reference`` (gene-by-parent matrix), and ``fallback_manifest``.
        ``merge_history`` contains only actual selected subtype merges.
        Manifest roles distinguish primary and family_fallback columns.

    Raises
    ------
    ValueError
        If counts are not raw non-negative integers, required metadata or gene
        annotation is missing, parameters are invalid, or filtering leaves
        insufficient genes, pseudobulks, or reference profiles.
    FileExistsError
        If ``memmap_path`` already exists.
    RuntimeError
        If reference names collide or final profiles fail positivity, finiteness,
        or normalization checks.

    Notes
    -----
    All candidate merges require complete-linkage constituent pair checks,
    absolute information-loss bounds, and a worst-unit incremental KL bound
    against the actual proposed robust profile. The final emitted profile is
    rechecked after count pooling/QC. Context groups use the same protections.
    Insufficient evidence produces an explicit unresolved decision, rather than
    a zero divergence. These are deterministic effect-size heuristics; they
    do not provide posterior equivalence probabilities or optimality guarantees.
    Family fallbacks intentionally relax subtype resolution, carry measured
    distortion diagnostics, and are not counted as accepted merges.

    The atlas should contain non-malignant samples appropriate for use as a
    diploid baseline. Normal-tissue annotation alone does not prove copy-number
    neutrality; donor- and tissue-level QC remains the caller's responsibility.

    Examples
    --------
    >>> import numpy as np
    >>> from spacenumbat.preprocessing.build_rna_ref import build_rna_reference
    >>> ref_rna, manifest, gene_gtf, diagnostics = build_rna_reference(
    ...     adata_full,
    ...     gtf=spacenumbat.data.hg38,
    ...     donor_col="donor",
    ...     tissue_col="tissue",
    ...     cell_type_col="free_annotation",
    ...     assay_col="method",
    ...     assays=("10x",),
    ...     anatomy_col=None,
    ...     min_cells=50,
    ...     min_counts=50_000,
    ...     min_donors=2,
    ...     subtype_ratio_threshold=1.5,
    ...     tissue_ratio_threshold=2.0,
    ...     return_diagnostics=True,
    ... )
    >>> ref_rna.shape[0] == gene_gtf.shape[0]
    True
    >>> ref_rna.columns.tolist() == manifest["reference"].tolist()
    True
    >>> bool(np.allclose(ref_rna.sum(axis=0), 1.0))
    True

    Enable the additional population check by supplying a common embedding:

    >>> ref_rna, manifest, gene_gtf, diagnostics = build_rna_reference(
    ...     adata_full, gtf=spacenumbat.data.hg38,
    ...     sinkhorn_embedding="X_pca", fallback_mode="separate",
    ...     return_diagnostics=True,
    ... )
    """
    if gtf is None:
        from spacenumbat.data import hg38
        gtf = hg38
    if adata.obs_names.has_duplicates or adata.var_names.has_duplicates:
        raise ValueError("Cell and feature identifiers must be unique.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"AnnData has no layer {layer!r}.")
    for name, value in {"min_cells": min_cells, "min_counts": min_counts,
                        "min_donors": min_donors, "min_context_donors": min_context_donors,
                        "min_cell_counts": min_cell_counts, "min_cell_genes": min_cell_genes,
                        "chunk_size": chunk_size, "sinkhorn_max_cells": sinkhorn_max_cells,
                        "sinkhorn_max_iter": sinkhorn_max_iter}.items():
        if value < 1 or int(value) != value:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in {"subtype_ratio_threshold": subtype_ratio_threshold,
                        "tissue_ratio_threshold": tissue_ratio_threshold,
                        "noise_floor": noise_floor, "prior_mass": prior_mass,
                        "eps": eps}.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    for name, value in {"max_merge_js": max_merge_js, "max_merge_kl": max_merge_kl,
                        "sinkhorn_threshold": sinkhorn_threshold}.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative.")
    for name, value in {"sinkhorn_epsilon": sinkhorn_epsilon,
                        "sinkhorn_cost_scale": sinkhorn_cost_scale,
                        "sinkhorn_tolerance": sinkhorn_tolerance}.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if unshared_context_policy not in {"defer", "compare"}:
        raise ValueError("unshared_context_policy must be 'defer' or 'compare'.")
    if fallback_mode not in {"missing", "separate", "off"}:
        raise ValueError("fallback_mode must be 'missing', 'separate', or 'off'.")
    if isinstance(random_state, bool) or int(random_state) != random_state or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer.")
    embedding_key = sinkhorn_embedding if isinstance(sinkhorn_embedding, str) else None
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise ValueError(f"AnnData has no obsm embedding {embedding_key!r}.")
        sinkhorn_embedding = adata.obsm[embedding_key]
    embedding = None
    if sinkhorn_embedding is not None:
        if sparse.issparse(sinkhorn_embedding):
            raise ValueError("sinkhorn_embedding must be a dense low-dimensional array.")
        embedding = np.asarray(sinkhorn_embedding)
        if embedding.ndim != 2 or embedding.shape[0] != adata.n_obs or embedding.shape[1] < 1:
            raise ValueError("sinkhorn_embedding must have shape (adata.n_obs, n_dimensions).")
    required = [donor_col, tissue_col, cell_type_col, assay_col]
    if anatomy_col is not None:
        required.append(anatomy_col)
    if max_pct_counts_mt is not None:
        required.append("pct_counts_mt")
    missing = set(required) - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing obs columns: {sorted(missing)}")

    gene_gtf, projection, feature_map = _gene_mapping(adata, gtf, gene_col)
    obs = adata.obs; meta = pd.DataFrame(index=obs.index)
    meta["donor"] = obs[donor_col].astype("string").str.strip()
    meta["tissue"] = obs[tissue_col].astype("string").str.strip()
    meta["label"] = obs[cell_type_col].astype("string").map(_label)
    meta["assay"] = obs[assay_col].astype("string").map(_assay)
    if anatomy_col is None:
        meta["context"] = meta.tissue
    else:
        anatomy = obs[anatomy_col].astype("string").fillna("NA").str.strip()
        meta["context"] = meta.tissue.astype(str) + "::" + anatomy.astype(str)
    families = dict(FAMILY_MAPPING)
    if family_mapping is not None:
        families.update({_label(k): str(v) for k, v in family_mapping.items()})
    meta["family"] = meta.label.map(families).fillna(meta.label)
    excluded = set(EXCLUDE_CELL_TYPES)
    if exclude_cell_types is not None:
        excluded |= {_label(x) for x in exclude_cell_types}
    keep = (_valid(obs[donor_col]).to_numpy() & _valid(obs[tissue_col]).to_numpy() &
            _valid(obs[cell_type_col]).to_numpy() & _valid(obs[assay_col]).to_numpy() &
            ~meta.label.isin(excluded).to_numpy())
    if assays is not None:
        assays = {_assay(x) for x in assays}; keep &= meta.assay.isin(assays).to_numpy()
    meta["_keep"] = keep

    embedding_hash = None
    if embedding is not None:
        digest = hashlib.sha256(str(embedding.shape).encode())
        for start in range(0, adata.n_obs, chunk_size):
            digest.update(np.ascontiguousarray(embedding[start:start + chunk_size],
                                              dtype=np.float64).tobytes())
        embedding_hash = digest.hexdigest()
    cell_reservoir = {} if embedding is not None else None
    counts, pb, cell_qc = _aggregate_counts(
        adata, meta, projection, layer=layer, chunk_size=chunk_size,
        min_cell_counts=min_cell_counts, min_cell_genes=min_cell_genes,
        max_pct_counts_mt=max_pct_counts_mt, memmap_path=memmap_path,
        cell_reservoir=cell_reservoir, sinkhorn_max_cells=sinkhorn_max_cells,
        random_state=random_state, cell_embedding=embedding)
    expressed = counts.sum(axis=0) > 0; counts = counts[:, expressed]
    gene_gtf = gene_gtf.loc[expressed].reset_index(drop=True)
    if counts.shape[1] < 2:
        raise ValueError("Fewer than two expressed autosomal genes remain.")
    initial_good = ((pb.n_cells >= min_cells) & (pb.n_counts >= min_counts)).to_numpy()
    pb["initial_qc_pass"] = initial_good
    test_counts = counts[initial_good]; test_obs = pb.loc[initial_good].reset_index(drop=True)
    samples = {}
    if embedding is not None:
        for code, (indices, _) in cell_reservoir.items():
            if not np.isfinite(embedding[indices]).all():
                raise ValueError("QC-passing sampled cells have nonfinite embedding coordinates.")
            row = pb.iloc[code]
            samples[tuple(row[["assay", "context", "family", "label", "donor"]])] = indices

    references, manifest_rows, fallback_references, fallback_rows = {}, [], {}, []
    mapping_rows, merge_rows, context_rows, merged_qc_rows, noise_rows = [], [], [], [], []
    merge_decisions, context_decisions, unresolved_rows = [], [], []
    for assay in sorted(pb.assay.unique()):
        assay_mask = test_obs.assay.eq(assay).to_numpy()
        assay_counts = test_counts[assay_mask]
        assay_obs = test_obs.loc[assay_mask].reset_index(drop=True)
        source_mask = pb.assay.eq(assay).to_numpy() & (pb.n_counts > 0).to_numpy()
        source_obs = pb.loc[source_mask].copy().reset_index(drop=True)
        source_counts = counts[source_mask]
        global_noise = _global_donor_noise(assay_counts, assay_obs, prior_mass, noise_floor)
        noise_rows.append({"assay": assay, "global_donor_js": global_noise})
        transport = (_CellTransport(embedding, samples, assay_obs,
                                   epsilon=sinkhorn_epsilon, cost_scale=sinkhorn_cost_scale,
                                   max_iter=sinkhorn_max_iter, tolerance=sinkhorn_tolerance,
                                   min_donors=min_donors) if embedding is not None else None)
        original_to_merged = {}
        for family in sorted(source_obs.family.unique()):
            family_mask = assay_obs.family.eq(family).to_numpy()
            raw_mask = source_obs.family.eq(family).to_numpy()
            family_counts = assay_counts[family_mask]
            family_obs = assay_obs.loc[family_mask].reset_index(drop=True)
            family_raw = source_counts[raw_mask]
            family_source = source_obs.loc[raw_mask].reset_index(drop=True)
            labels = sorted(family_source.label.unique())
            if len(labels) == 1:
                clusters, history = [frozenset(labels)], []
            else:
                clusters, history = _agglomerate_family(
                    family_counts, family_obs, labels, family, prior_mass=prior_mass,
                    global_noise=global_noise, ratio_threshold=subtype_ratio_threshold,
                    noise_floor=noise_floor, source_counts=family_raw, source_obs=family_source,
                    min_cells=min_cells, min_counts=min_counts, min_donors=min_donors,
                    max_merge_js=max_merge_js, max_merge_kl=max_merge_kl, eps=eps,
                    unshared_context_policy=unshared_context_policy, transport=transport,
                    sinkhorn_threshold=sinkhorn_threshold)
            merge_decisions.extend({"assay": assay, **row} for row in history)
            merge_rows.extend({"assay": assay, **row} for row in history if row["accepted"])
            if len(clusters) == 1:
                named_clusters = [(clusters[0], family)]
            else:
                named_clusters = [(cluster, next(iter(cluster)) if len(cluster) == 1
                                   else f"{family} group {index}")
                                  for index, cluster in enumerate(clusters, start=1)]
            for cluster, merged_type in named_clusters:
                for label in sorted(cluster):
                    original_to_merged[label] = merged_type
                    mapping_rows.append({"assay": assay, "original_cell_type": label,
                                         "family": family, "merged_cell_type": merged_type})

        source_obs["merged_type"] = source_obs.label.map(original_to_merged)
        merged_counts, merged_obs = _aggregate_rows(
            source_counts, source_obs, ["context", "family", "merged_type", "donor"])
        merged_good = ((merged_obs.n_cells >= min_cells) &
                       (merged_obs.n_counts >= min_counts)).to_numpy()
        merged_obs["qc_pass"] = merged_good; merged_obs["assay"] = assay
        merged_qc_rows.append(merged_obs.copy())
        merged_counts = merged_counts[merged_good]
        merged_obs = merged_obs.loc[merged_good].reset_index(drop=True)
        covered = set()
        suffix = f"|{assay}" if len(pb.assay.unique()) > 1 else ""
        for merged_type, idx in merged_obs.groupby("merged_type", sort=True).indices.items():
            idx = np.asarray(idx); type_counts = merged_counts[idx]
            type_obs = merged_obs.iloc[idx].reset_index(drop=True)
            family = str(type_obs.family.iloc[0])
            raw = source_obs.merged_type.eq(merged_type).to_numpy()
            raw_counts, raw_obs = _supported_sources(
                source_counts[raw], source_obs.loc[raw], min_cells=min_cells, min_counts=min_counts)
            labels = set(raw_obs.label)
            result = _test_context_information(
                type_counts, type_obs, min_context_donors=min_context_donors,
                prior_mass=prior_mass, global_noise=global_noise,
                ratio_threshold=tissue_ratio_threshold, noise_floor=noise_floor,
                source_counts=raw_counts, source_obs=raw_obs, min_donors=min_donors,
                max_merge_js=max_merge_js, max_merge_kl=max_merge_kl, eps=eps,
                labels=labels, transport=transport, sinkhorn_threshold=sinkhorn_threshold)
            context_rows.append({"assay": assay, "cell_type": merged_type, "family": family,
                                 **{k: result[k] for k in ("split", "context_js", "donor_noise", "ratio")},
                                 "eligible_contexts": json.dumps(result["eligible_contexts"]),
                                 "context_groups": json.dumps(result["groups"]),
                                 "deferred_contexts": json.dumps(result["deferred_contexts"])})
            context_decisions.extend({"assay": assay, "cell_type": merged_type, **r}
                                     for r in result["history"])
            for contexts in result["groups"]:
                mask = type_obs.context.astype(str).isin(contexts).to_numpy()
                group_counts, group_obs = type_counts[mask], type_obs.loc[mask]
                reference, donors = _robust_donor_consensus(
                    group_counts, group_obs.reset_index(drop=True), prior_mass=prior_mass, eps=eps)
                raw_mask = raw_obs.context.astype(str).isin(contexts).to_numpy()
                distortion = _profile_distortion(raw_counts[raw_mask], raw_obs.loc[raw_mask],
                                                  units=["label", "context"], prior_mass=prior_mass,
                                                  eps=eps, reference=reference)
                # Re-check against the exact emitted profile and rescued raw rows.
                if not _distortion_allowed(distortion, max_merge_js, max_merge_kl):
                    context_decisions.append({"assay": assay, "cell_type": merged_type,
                                              "merged": json.dumps(contexts), **distortion,
                                              "selected": False, "decision_reason": "final_profile_distortion"})
                    continue
                pan = len(result["groups"]) == 1 and not result["deferred_contexts"]
                scope = "pan_context" if pan else ("context" if len(contexts) == 1 else "context_group")
                if pan:
                    context = "pan_context"; name = f"{merged_type}{suffix}"
                elif len(contexts) == 1:
                    context = contexts[0]; name = f"{merged_type}|{context}{suffix}"
                else:
                    digest = hashlib.sha256(json.dumps(contexts).encode()).hexdigest()[:12]
                    context = f"context_group:{digest}"; name = f"{merged_type}|{context}{suffix}"
                if name in references:
                    raise RuntimeError(f"Reference name collision: {name}")
                references[name] = reference
                members = sorted(set(raw_obs.loc[raw_mask, "label"]))
                manifest_rows.append({
                    "reference": name, "cell_type": merged_type, "family": family,
                    "assay": assay, "scope": scope, "context": context,
                    "contexts": json.dumps(contexts),
                    "tissues": json.dumps(sorted({x.split("::", 1)[0] for x in contexts})),
                    "donors": json.dumps(sorted(donors)), "n_donors": len(donors),
                    "n_cells": int(group_obs.n_cells.sum()), "n_counts": float(group_obs.n_counts.sum()),
                    "source_cell_types": json.dumps(members), "reference_role": "primary",
                    "context_js": result["context_js"], "context_noise": result["donor_noise"],
                    "context_ratio": result["ratio"], **distortion})
                covered.update(zip(raw_obs.loc[raw_mask, "label"], raw_obs.loc[raw_mask, "context"]))

        # Parent profiles are explicit coarse fallbacks, not accepted subtype merges.
        for family, idx in source_obs.groupby("family", sort=True).indices.items():
            idx = np.asarray(idx); fc, fo = source_counts[idx], source_obs.iloc[idx].reset_index(drop=True)
            fc, fo = _supported_sources(fc, fo, min_cells=min_cells, min_counts=min_counts)
            family_units = set(zip(source_obs.iloc[idx].label, source_obs.iloc[idx].context))
            missing_units = sorted(family_units - covered)
            name = f"{family}|family_fallback{suffix}"
            available = False
            if fallback_mode != "off" and len(fo):
                profile, donors = _robust_donor_consensus(fc, fo, prior_mass=prior_mass, eps=eps)
                available = len(donors) >= min_donors
                if available:
                    contexts = sorted(set(fo.context.astype(str)))
                    distortion = _profile_distortion(fc, fo, units=["label", "context"],
                                                      prior_mass=prior_mass, eps=eps, reference=profile)
                    row = {"reference": name, "cell_type": family, "family": family,
                           "assay": assay, "scope": "family_fallback", "context": "pan_context",
                           "contexts": json.dumps(contexts),
                           "tissues": json.dumps(sorted({x.split("::", 1)[0] for x in contexts})),
                           "donors": json.dumps(sorted(donors)), "n_donors": len(donors),
                           "n_cells": int(fo.n_cells.sum()), "n_counts": float(fo.n_counts.sum()),
                           "source_cell_types": json.dumps(sorted(set(fo.label))),
                           "reference_role": "family_fallback", **distortion,
                           "within_distortion_bounds": _distortion_allowed(distortion, max_merge_js, max_merge_kl),
                           "active": fallback_mode == "missing" and bool(missing_units)}
                    fallback_references[name] = profile; fallback_rows.append(row)
                    if row["active"]:
                        if name in references:
                            raise RuntimeError(f"Reference name collision: {name}")
                        references[name] = profile; manifest_rows.append(row)
            for label, context in missing_units:
                unresolved_rows.append({"assay": assay, "family": family,
                                        "original_cell_type": label, "context": context,
                                        "reason": "no_supported_primary_within_bounds",
                                        "fallback_reference": name if available else None,
                                        "fallback_active": available and fallback_mode == "missing"})

    if not references:
        raise ValueError("No final RNA reference profiles were generated.")
    reference = pd.DataFrame(references, index=pd.Index(gene_gtf.gene, name="gene"),
                             dtype=np.float64)
    if not np.isfinite(reference.to_numpy()).all() or (reference <= 0).any().any():
        raise RuntimeError("Reference profiles must be finite and positive.")
    if not np.allclose(reference.sum(axis=0), 1.0, atol=1e-12):
        raise RuntimeError("Reference normalization failed.")
    manifest = pd.DataFrame(manifest_rows)
    diagnostics = {
        "cell_type_map": pd.DataFrame(mapping_rows),
        "merge_history": pd.DataFrame(merge_rows),
        "merge_decisions": pd.DataFrame(merge_decisions),
        "context_merge_decisions": pd.DataFrame(context_decisions),
        "unresolved_units": pd.DataFrame(unresolved_rows),
        "fallback_reference": pd.DataFrame(fallback_references,
                                             index=pd.Index(gene_gtf.gene, name="gene")),
        "fallback_manifest": pd.DataFrame(fallback_rows),
        "context_tests": pd.DataFrame(context_rows),
        "initial_pseudobulk_qc": pb,
        "merged_pseudobulk_qc": (pd.concat(merged_qc_rows, ignore_index=True)
                                  if merged_qc_rows else pd.DataFrame()),
        "global_donor_noise": pd.DataFrame(noise_rows),
        "feature_mapping": feature_map,
        "run_info": {
            "builder_version": __version__, "algorithm": "distortion_bounded_RNA_with_optional_Sinkhorn",
            "n_input_cells": int(adata.n_obs), "n_input_genes": int(adata.n_vars),
            "n_genes": int(reference.shape[0]), "n_references": int(reference.shape[1]),
            "cell_qc": cell_qc, "subtype_ratio_threshold": subtype_ratio_threshold,
            "tissue_ratio_threshold": tissue_ratio_threshold,
            "max_merge_js": max_merge_js, "max_merge_kl": max_merge_kl,
            "unshared_context_policy": unshared_context_policy, "fallback_mode": fallback_mode,
            "sinkhorn_enabled": embedding is not None, "sinkhorn_embedding_key": embedding_key,
            "sinkhorn_embedding_sha256": embedding_hash,
            "sinkhorn_threshold": sinkhorn_threshold, "sinkhorn_epsilon": sinkhorn_epsilon,
            "sinkhorn_cost_scale": sinkhorn_cost_scale, "sinkhorn_max_cells": sinkhorn_max_cells,
            "sinkhorn_max_iter": sinkhorn_max_iter, "sinkhorn_tolerance": sinkhorn_tolerance,
            "random_state": int(random_state),
            "family_mapping_sha256": hashlib.sha256(
                json.dumps(families, sort_keys=True).encode()).hexdigest(),
        },
    }
    result = (reference, manifest, gene_gtf)
    return (*result, diagnostics) if return_diagnostics else result


def save_rna_reference(reference, manifest, gene_gtf, diagnostics, output_dir):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)
    reference.rename_axis(None).to_csv(output_dir / "ref_rna.tsv", sep="\t", index_label=False)
    manifest.to_csv(output_dir / "reference_manifest.tsv", sep="\t", index=False)
    gene_gtf.to_csv(output_dir / "gene_gtf.tsv", sep="\t", index=False)
    for name, value in diagnostics.items():
        if isinstance(value, pd.DataFrame):
            if value.shape[1] == 0:
                value = pd.DataFrame(columns=["stage", "reason"])
            if name == "fallback_reference":
                value.to_csv(output_dir / f"{name}.tsv", sep="\t", index=True, index_label="gene")
            else:
                value.to_csv(output_dir / f"{name}.tsv", sep="\t", index=False)
    (output_dir / "run_info.json").write_text(json.dumps(diagnostics["run_info"], indent=2) + "\n")
    return output_dir / "ref_rna.tsv"


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5ad"); parser.add_argument("output_dir"); parser.add_argument("--gtf")
    parser.add_argument("--assays", nargs="+", default=["10x"])
    parser.add_argument("--donor-col", default="donor")
    parser.add_argument("--tissue-col", default="tissue")
    parser.add_argument("--cell-type-col", default="free_annotation")
    parser.add_argument("--assay-col", default="method")
    parser.add_argument("--anatomy-col"); parser.add_argument("--gene-col"); parser.add_argument("--layer")
    parser.add_argument("--family-json"); parser.add_argument("--min-cells", type=int, default=50)
    parser.add_argument("--min-counts", type=int, default=50_000)
    parser.add_argument("--min-donors", type=int, default=2)
    parser.add_argument("--min-context-donors", type=int, default=2)
    parser.add_argument("--subtype-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--tissue-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--chunk-size", type=int, default=8192); parser.add_argument("--memmap-path")
    parser.add_argument("--max-merge-js", type=float, default=0.02)
    parser.add_argument("--max-merge-kl", type=float, default=0.05)
    parser.add_argument("--unshared-context-policy", choices=["defer", "compare"], default="defer")
    parser.add_argument("--fallback-mode", choices=["missing", "separate", "off"], default="missing")
    parser.add_argument("--sinkhorn-embedding", help="Common dense adata.obsm embedding key")
    parser.add_argument("--sinkhorn-threshold", type=float, default=0.05)
    parser.add_argument("--sinkhorn-epsilon", type=float, default=0.05)
    parser.add_argument("--sinkhorn-cost-scale", type=float, default=1.0)
    parser.add_argument("--sinkhorn-max-cells", type=int, default=64)
    parser.add_argument("--sinkhorn-max-iter", type=int, default=1000)
    parser.add_argument("--sinkhorn-tolerance", type=float, default=1e-7)
    parser.add_argument("--random-state", type=int, default=0)
    args = parser.parse_args()
    import anndata as ad
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    atlas = ad.read_h5ad(args.h5ad, backed="r")
    try:
        result = build_rna_reference(
            atlas, gtf=pd.read_csv(args.gtf, sep="\t") if args.gtf else None,
            donor_col=args.donor_col, tissue_col=args.tissue_col,
            cell_type_col=args.cell_type_col, assay_col=args.assay_col,
            assays=None if args.assays == ["ALL"] else args.assays,
            anatomy_col=args.anatomy_col, gene_col=args.gene_col, layer=args.layer,
            family_mapping=json.loads(Path(args.family_json).read_text()) if args.family_json else None,
            min_cells=args.min_cells, min_counts=args.min_counts, min_donors=args.min_donors,
            min_context_donors=args.min_context_donors,
            subtype_ratio_threshold=args.subtype_ratio_threshold,
            tissue_ratio_threshold=args.tissue_ratio_threshold,
            chunk_size=args.chunk_size, memmap_path=args.memmap_path, return_diagnostics=True,
            max_merge_js=args.max_merge_js, max_merge_kl=args.max_merge_kl,
            unshared_context_policy=args.unshared_context_policy, fallback_mode=args.fallback_mode,
            sinkhorn_embedding=args.sinkhorn_embedding, sinkhorn_threshold=args.sinkhorn_threshold,
            sinkhorn_epsilon=args.sinkhorn_epsilon, sinkhorn_cost_scale=args.sinkhorn_cost_scale,
            sinkhorn_max_cells=args.sinkhorn_max_cells, sinkhorn_max_iter=args.sinkhorn_max_iter,
            sinkhorn_tolerance=args.sinkhorn_tolerance, random_state=args.random_state)
        save_rna_reference(*result, args.output_dir)
    finally:
        atlas.file.close()


if __name__ == "__main__":
    _main()
