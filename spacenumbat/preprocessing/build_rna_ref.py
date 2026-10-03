#!/usr/bin/env python3
"""Build a donor-balanced scRNA reference with the ATAC reference algorithm.

The algorithm is a RNA adaptation of build_atac_ref.py:
donor/context/type pseudobulks, within-family subtype compression, a context
information test, and equal-donor robust final profiles.
"""

#from __future__ import annotations

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
from scipy.special import rel_entr


log = logging.getLogger(__name__)
__version__ = "1.0.0"


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
                      min_cell_counts, min_cell_genes, max_pct_counts_mt, memmap_path):
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


def _subtype_pair_score(counts, obs, cluster_a, cluster_b, *, prior_mass,
                        fallback_noise, noise_floor):
    mask_a = obs.label.isin(cluster_a); mask_b = obs.label.isin(cluster_b)
    shared = sorted(set(obs.loc[mask_a, "context"]) & set(obs.loc[mask_b, "context"]))
    if not shared:
        return {"between_js": 0.0, "donor_noise": fallback_noise, "ratio": 0.0,
                "shared_contexts": 0, "reason": "no_shared_context"}
    between, weights, donor_noise = [], [], []
    for context in shared:
        a = (mask_a & obs.context.eq(context)).to_numpy()
        b = (mask_b & obs.context.eq(context)).to_numpy()
        pa, _ = _donor_profiles(counts[a], obs.loc[a], prior_mass)
        pb, _ = _donor_profiles(counts[b], obs.loc[b], prior_mass)
        weight = len(pa) / (len(pa) + len(pb))
        between.append(_js(pa.mean(axis=0), pb.mean(axis=0), weight=weight))
        weights.append(min(len(pa), len(pb)))
        donor_noise.extend(_pairwise_js(pa)); donor_noise.extend(_pairwise_js(pb))
    noise = max(float(np.median(donor_noise)) if donor_noise else fallback_noise, noise_floor)
    between_js = float(np.average(between, weights=weights))
    return {"between_js": between_js, "donor_noise": noise,
            "ratio": between_js / noise, "shared_contexts": len(shared), "reason": "observed"}


def _agglomerate_family(counts, obs, labels, family, *, prior_mass,
                        global_noise, ratio_threshold, noise_floor):
    clusters = [frozenset([label]) for label in sorted(labels)]
    family_noise = _within_context_donor_noise(counts, obs, prior_mass)
    if not np.isfinite(family_noise):
        family_noise = global_noise
    history = []
    while len(clusters) > 1:
        candidates = []
        for i, j in itertools.combinations(range(len(clusters)), 2):
            score = _subtype_pair_score(counts, obs, clusters[i], clusters[j],
                                        prior_mass=prior_mass, fallback_noise=family_noise,
                                        noise_floor=noise_floor)
            candidates.append((score["ratio"], i, j, score))
        ratio, i, j, score = min(candidates, key=lambda item: item[0])
        if ratio > ratio_threshold:
            break
        left, right = clusters[i], clusters[j]; merged = left | right
        history.append({"family": family, "left": json.dumps(sorted(left)),
                        "right": json.dumps(sorted(right)),
                        "merged": json.dumps(sorted(merged)), **score})
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [merged]
    return sorted(clusters, key=lambda cluster: sorted(cluster)), history


def _test_context_information(counts, obs, *, min_context_donors, prior_mass,
                              global_noise, ratio_threshold, noise_floor):
    profiles, weights, eligible, donor_noise = [], [], [], []
    for context, idx in obs.groupby("context", sort=True).indices.items():
        donor_profiles, donor_obs = _donor_profiles(counts[np.asarray(idx)], obs.iloc[idx],
                                                    prior_mass)
        if len(donor_obs) < min_context_donors:
            continue
        consensus = donor_profiles.mean(axis=0); consensus /= consensus.sum()
        profiles.append(consensus); weights.append(len(donor_obs)); eligible.append(str(context))
        donor_noise.extend(_pairwise_js(donor_profiles))
    if len(eligible) < 2:
        return {"split": False, "context_js": np.nan, "donor_noise": np.nan,
                "ratio": np.nan, "eligible_contexts": eligible}
    context_js = _generalized_js(np.vstack(profiles), weights)
    local_noise = float(np.median(donor_noise)) if donor_noise else global_noise
    noise = max(local_noise, global_noise, noise_floor); ratio = context_js / noise
    return {"split": bool(ratio > ratio_threshold), "context_js": context_js,
            "donor_noise": noise, "ratio": ratio, "eligible_contexts": eligible}


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
):
    """Construct an RNA reference using the scATAC builder's hierarchy.

    Increase `subtype_ratio_threshold` to merge more labels within curated
    families. Increase `tissue_ratio_threshold` to retain fewer context-specific
    profiles. No requested reference count is used.
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
                        "chunk_size": chunk_size}.items():
        if value < 1 or int(value) != value:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in {"subtype_ratio_threshold": subtype_ratio_threshold,
                        "tissue_ratio_threshold": tissue_ratio_threshold,
                        "noise_floor": noise_floor, "prior_mass": prior_mass,
                        "eps": eps}.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
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

    counts, pb, cell_qc = _aggregate_counts(
        adata, meta, projection, layer=layer, chunk_size=chunk_size,
        min_cell_counts=min_cell_counts, min_cell_genes=min_cell_genes,
        max_pct_counts_mt=max_pct_counts_mt, memmap_path=memmap_path)
    expressed = counts.sum(axis=0) > 0; counts = counts[:, expressed]
    gene_gtf = gene_gtf.loc[expressed].reset_index(drop=True)
    if counts.shape[1] < 2:
        raise ValueError("Fewer than two expressed autosomal genes remain.")
    initial_good = ((pb.n_cells >= min_cells) & (pb.n_counts >= min_counts)).to_numpy()
    pb["initial_qc_pass"] = initial_good
    test_counts = counts[initial_good]; test_obs = pb.loc[initial_good].reset_index(drop=True)
    if test_obs.empty:
        raise ValueError("No donor/context/cell-type pseudobulks passed QC.")

    references, manifest_rows = {}, []
    mapping_rows, merge_rows, context_rows, merged_qc_rows, noise_rows = [], [], [], [], []
    assay_values = sorted(test_obs.assay.unique())
    for assay in assay_values:
        assay_mask = test_obs.assay.eq(assay).to_numpy()
        assay_counts = test_counts[assay_mask]
        assay_obs = test_obs.loc[assay_mask].reset_index(drop=True)
        assay_all_obs = pb.loc[pb.assay.eq(assay)].reset_index(drop=True)
        global_noise = _global_donor_noise(assay_counts, assay_obs, prior_mass, noise_floor)
        noise_rows.append({"assay": assay, "global_donor_js": global_noise})
        original_to_merged = {}
        for family in sorted(assay_all_obs.family.unique()):
            family_mask = assay_obs.family.eq(family).to_numpy()
            family_counts = assay_counts[family_mask]
            family_obs = assay_obs.loc[family_mask].reset_index(drop=True)
            labels = sorted(assay_all_obs.loc[assay_all_obs.family.eq(family), "label"].unique())
            if len(labels) == 1:
                clusters, history = [frozenset(labels)], []
            else:
                clusters, history = _agglomerate_family(
                    family_counts, family_obs, labels, family, prior_mass=prior_mass,
                    global_noise=global_noise, ratio_threshold=subtype_ratio_threshold,
                    noise_floor=noise_floor)
            merge_rows.extend({"assay": assay, **row} for row in history)
            if len(clusters) == 1:
                named_clusters = [(clusters[0], family)]
            else:
                named_clusters = [(cluster, next(iter(cluster)) if len(cluster) == 1
                                   else f"{family} group {index}")
                                  for index, cluster in enumerate(clusters, start=1)]
            for cluster, merged_type in named_clusters:
                for label in cluster:
                    original_to_merged[label] = merged_type
                    mapping_rows.append({"assay": assay, "original_cell_type": label,
                                         "family": family, "merged_cell_type": merged_type})

        source_mask = pb.assay.eq(assay).to_numpy()
        source_obs = pb.loc[source_mask].copy().reset_index(drop=True)
        source_counts = counts[source_mask]
        source_obs["merged_type"] = source_obs.label.map(original_to_merged)
        valid = source_obs.merged_type.notna().to_numpy()
        merged_counts, merged_obs = _aggregate_rows(
            source_counts[valid], source_obs.loc[valid].reset_index(drop=True),
            ["context", "family", "merged_type", "donor"])
        merged_good = ((merged_obs.n_cells >= min_cells) &
                       (merged_obs.n_counts >= min_counts)).to_numpy()
        merged_obs["qc_pass"] = merged_good; merged_obs["assay"] = assay
        merged_qc_rows.append(merged_obs.copy())
        merged_counts = merged_counts[merged_good]
        merged_obs = merged_obs.loc[merged_good].reset_index(drop=True)
        source_types = (pd.DataFrame([{"original": k, "merged": v}
                                     for k, v in original_to_merged.items()])
                        .groupby("merged").original.agg(list).to_dict())

        for merged_type, idx in merged_obs.groupby("merged_type", sort=True).indices.items():
            idx = np.asarray(idx); type_counts = merged_counts[idx]
            type_obs = merged_obs.iloc[idx].reset_index(drop=True)
            family = str(type_obs.family.iloc[0])
            result = _test_context_information(
                type_counts, type_obs, min_context_donors=min_context_donors,
                prior_mass=prior_mass, global_noise=global_noise,
                ratio_threshold=tissue_ratio_threshold, noise_floor=noise_floor)
            context_rows.append({"assay": assay, "cell_type": merged_type, "family": family,
                                 "split": result["split"], "context_js": result["context_js"],
                                 "donor_noise": result["donor_noise"], "ratio": result["ratio"],
                                 "eligible_contexts": json.dumps(result["eligible_contexts"])})
            groups = []
            if result["split"]:
                for context in result["eligible_contexts"]:
                    mask = type_obs.context.astype(str).eq(context).to_numpy()
                    groups.append(("context", context, type_counts[mask], type_obs.loc[mask]))
            else:
                groups.append(("pan_context", "pan_context", type_counts, type_obs))
            for scope, context, group_counts, group_obs in groups:
                reference, donors = _robust_donor_consensus(
                    group_counts, group_obs.reset_index(drop=True), prior_mass=prior_mass, eps=eps)
                if len(donors) < min_donors:
                    continue
                suffix = f"|{assay}" if len(assay_values) > 1 else ""
                name = (f"{merged_type}|{context}{suffix}" if scope == "context"
                        else f"{merged_type}{suffix}")
                if name in references:
                    raise RuntimeError(f"Reference name collision: {name}")
                references[name] = reference
                contexts = sorted(group_obs.context.astype(str).unique())
                tissues = sorted({value.split("::", 1)[0] for value in contexts})
                manifest_rows.append({
                    "reference": name, "cell_type": merged_type, "family": family,
                    "assay": assay, "scope": scope, "context": context,
                    "contexts": json.dumps(contexts), "tissues": json.dumps(tissues),
                    "donors": json.dumps(sorted(donors)), "n_donors": len(donors),
                    "n_cells": int(group_obs.n_cells.sum()),
                    "n_counts": float(group_obs.n_counts.sum()),
                    "source_cell_types": json.dumps(sorted(source_types.get(merged_type, []))),
                    "context_js": result["context_js"],
                    "context_noise": result["donor_noise"],
                    "context_ratio": result["ratio"],
                })

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
        "context_tests": pd.DataFrame(context_rows),
        "initial_pseudobulk_qc": pb,
        "merged_pseudobulk_qc": (pd.concat(merged_qc_rows, ignore_index=True)
                                  if merged_qc_rows else pd.DataFrame()),
        "global_donor_noise": pd.DataFrame(noise_rows),
        "feature_mapping": feature_map,
        "run_info": {
            "builder_version": __version__, "algorithm": "direct_ATAC_hierarchy_RNA",
            "n_input_cells": int(adata.n_obs), "n_input_genes": int(adata.n_vars),
            "n_genes": int(reference.shape[0]), "n_references": int(reference.shape[1]),
            "cell_qc": cell_qc, "subtype_ratio_threshold": subtype_ratio_threshold,
            "tissue_ratio_threshold": tissue_ratio_threshold,
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
            chunk_size=args.chunk_size, memmap_path=args.memmap_path, return_diagnostics=True)
        save_rna_reference(*result, args.output_dir)
    finally:
        atlas.file.close()


if __name__ == "__main__":
    _main()


