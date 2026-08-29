#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 29 16:21:05 2026

@author: carlino.calogero
"""

import itertools

import numpy as np
import pandas as pd
import scipy.sparse as sp


# Biological merge families
#
# These define which atlas labels are ALLOWED to merge.
# This does not force the merge: JSD + donor variability decide.

FAMILY_MAPPING = {

    # Fibroblasts
    "Fibroblast (General)": "Fibroblast",
    "Cardiac Fibroblasts": "Fibroblast",
    "Fibroblast (Peripheral Nerve)": "Fibroblast",
    "Fibroblast (Sk Muscle Associated)": "Fibroblast",
    "Fibroblast (Epithelial)": "Fibroblast",
    "Fibroblast (Gastrointestinal)": "Fibroblast",
    "Fibroblast (Liver Adrenal)": "Fibroblast",

    # Pericytes
    "Pericyte (General) 1": "Pericyte",
    "Pericyte (General) 2": "Pericyte",
    "Pericyte (General) 3": "Pericyte",
    "Pericyte (General) 4": "Pericyte",
    "Cardiac Pericyte 1": "Pericyte",
    "Cardiac Pericyte 2": "Pericyte",
    "Cardiac Pericyte 3": "Pericyte",
    "Cardiac Pericyte 4": "Pericyte",
    "Pericyte (Esophageal Muscularis)": "Pericyte",

    # Vascular smooth muscle
    "Vascular Smooth Muscle 1": "Vascular Smooth Muscle",
    "Vascular Smooth Muscle 2": "Vascular Smooth Muscle",

    # Smooth muscle
    "Smooth Muscle (General)": "Smooth Muscle",
    "Smooth Muscle (Esophageal Muscularis) 1": "Smooth Muscle",
    "Smooth Muscle (Esophageal Muscularis) 2": "Smooth Muscle",
    "Smooth Muscle (Esophageal Muscularis) 3": "Smooth Muscle",
    "Smooth Muscle (Esophageal Mucosal)": "Smooth Muscle",
    "Smooth Muscle (Colon) 1": "Smooth Muscle",
    "Smooth Muscle (Colon) 2": "Smooth Muscle",
    "Smooth Muscle (General Gastrointestinal)": "Smooth Muscle",
    "Smooth Muscle (GE Junction)": "Smooth Muscle",
    "Smooth Muscle (Vaginal)": "Smooth Muscle",
    "Smooth Muscle (Uterine)": "Smooth Muscle",

    # General endothelial
    "Endothelial Cell (General) 1": "Endothelial",
    "Endothelial Cell (General) 2": "Endothelial",
    "Endothelial Cell (General) 3": "Endothelial",

    # Macrophage
    "Macrophage (General)": "Macrophage",
    "Macrophage (General,Alveolar)": "Macrophage",

    # Closely related numbered populations
    "Pancreatic Alpha Cell 1": "Pancreatic Alpha",
    "Pancreatic Alpha Cell 2": "Pancreatic Alpha",

    "Pancreatic Beta Cell 1": "Pancreatic Beta",
    "Pancreatic Beta Cell 2": "Pancreatic Beta",

    "Mammary Luminal Epithelial Cell 1": "Mammary Luminal Epithelial",
    "Mammary Luminal Epithelial Cell 2": "Mammary Luminal Epithelial",

    "Keratinocyte 1": "Keratinocyte",
    "Keratinocyte 2": "Keratinocyte",

    "Astrocyte 1": "Astrocyte",
    "Astrocyte 2": "Astrocyte",
}


# Cosmetic names only. These do not define merge families.
CLEAN_NAMES = {
    "T Lymphocyte 1 (CD8+)": "CD8 T",
    "T lymphocyte 2 (CD4+)": "CD4 T",
    "Naive T cell": "Naive T",
    "Natural Killer T Cell": "NKT",
    "Memory B Cell": "Memory B",
    "Plasma Cell": "Plasma",
    "Mast Cell": "Mast",
}


# Composite populations that should not define a diploid baseline.
EXCLUDE_CELL_TYPES = {
    "Alverolar Type 2,Immune",
    "CNS,Enteric Neuron",
}



# Basic aggregation

def _aggregate_rows(X, obs, by):
    """Sum rows of a sparse matrix according to metadata groups."""

    keys = pd.MultiIndex.from_frame(obs[by].astype("string"))

    codes, groups = pd.factorize(keys, sort=False,)

    G = sp.csr_matrix((np.ones(len(codes), dtype=np.float64),
                       (codes, np.arange(len(codes)))),
                      shape=(len(groups), len(codes)))

    X_agg = (G @ X).tocsr()

    obs_agg = groups.to_frame(index=False)
    obs_agg.columns = by

    if "n_cells" in obs:
        weights = obs["n_cells"].to_numpy(dtype=float)
    else:
        weights = np.ones(len(obs))

    obs_agg["n_cells"] = np.bincount(codes, weights=weights,).astype(int)
    obs_agg["n_fragments"] = np.asarray(X_agg.sum(axis=1)).ravel()

    return X_agg, obs_agg


def _normalize_rows(X):
    """Convert count pseudobulks into probability distributions."""

    depth = np.asarray(X.sum(axis=1)).ravel().astype(float)

    if np.any(depth <= 0):
        raise ValueError("A pseudobulk contains zero fragments.")

    return X.multiply(1.0 / depth[:, None]).toarray()


def _donor_profiles(X, obs, donor_col):
    """
    Aggregate rows belonging to the same donor and normalize each donor
    independently.
    """

    X_donor, donor_obs = _aggregate_rows(X,
                                         obs.reset_index(drop=True),
                                         [donor_col])
    P = _normalize_rows(X_donor)

    return P, donor_obs


# Information-theoretic distances

def _kl_divergence(p, q):
    """KL divergence in natural-log units (nats)."""

    mask = p > 0

    return float(np.sum(p[mask]* np.log(p[mask] / q[mask])))


def _js_divergence(p, q, weight=0.5):
    """
    Weighted Jensen-Shannon divergence.

    Parameters
    ----------
    weight
        Prior probability assigned to p.
    """

    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)

    p /= p.sum()
    q /= q.sum()

    m = weight * p + (1.0 - weight) * q

    return (weight * _kl_divergence(p, m) + (1.0 - weight) * _kl_divergence(q, m))


def _generalized_js(P, weights=None):
    """
    Generalized Jensen-Shannon divergence among multiple distributions.

    This corresponds to the information carried by the group label about
    the genomic-bin distribution.
    """

    P = np.asarray(P, dtype=float)
    P /= P.sum(axis=1, keepdims=True,)

    n = P.shape[0]

    if weights is None:
        weights = np.full(n, 1.0 / n)

    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()

    mean_profile = np.sum(weights[:, None] * P, axis=0)

    return float(sum(w * _kl_divergence(p, mean_profile) for w, p in zip(weights, P)))


def _pairwise_js(P):
    """All pairwise JSD values between rows of P."""

    return [_js_divergence(P[i], 
                           P[j]) for i, j in itertools.combinations(range(P.shape[0]),
                                                                    2)]


# Donor noise

def _within_tissue_donor_noise(
    X,
    obs,
    donor_col,
    tissue_col,
    ):
    """
    Estimate donor variability while holding tissue constant.
    """

    values = []

    for _, idx in obs.groupby(tissue_col, sort=False).indices.items():

        idx = np.asarray(idx)
        P, _ = _donor_profiles(X[idx], obs.iloc[idx].reset_index(drop=True), donor_col,)
        values.extend(_pairwise_js(P))

    if not values:
        return np.nan

    return float(np.median(values))


def _global_donor_noise(
    X,
    obs,
    donor_col,
    tissue_col,
    type_col,
    noise_floor,
    ):
    """
    Estimate global technical/biological donor noise from comparisons
    where both cell type and tissue are held constant.
    """

    values = []

    groups = obs.groupby([type_col, tissue_col], sort=False).indices

    for _, idx in groups.items():

        idx = np.asarray(idx)

        P, _ = _donor_profiles(X[idx], obs.iloc[idx].reset_index(drop=True), donor_col)
        values.extend(_pairwise_js(P))

    if not values:
        return noise_floor

    return max(float(np.median(values)), noise_floor)


# Subtype merging
# Compare candidate subtypes only within tissues shared by both groups:
#
#          I(bin ; subtype | tissue, family)

def _subtype_pair_score(
    X,
    obs,
    cluster_a,
    cluster_b,
    type_col,
    donor_col,
    tissue_col,
    fallback_noise,
    noise_floor,
    ):
    """
    Compare two subtype clusters after conditioning on tissue.

    A ratio <= ~1 means the subtype difference is no larger than
    donor-to-donor variability.
    """

    mask_a = obs[type_col].isin(cluster_a)
    mask_b = obs[type_col].isin(cluster_b)

    tissues_a = set(obs.loc[mask_a, tissue_col])
    tissues_b = set(obs.loc[mask_b, tissue_col])
    shared_tissues = sorted(tissues_a & tissues_b)

    # No shared tissue -> subtype effect cannot be separated from tissue.
    # Because the labels were placed in the same curated biological family,
    # collapse them and allow the tissue test to decide later.
    if not shared_tissues:
        return {
            "between_js": 0.0,
            "donor_noise": fallback_noise,
            "ratio": 0.0,
            "shared_tissues": 0,
            "reason": "no_shared_tissue",
        }

    between = []
    weights = []
    donor_noise = []

    for tissue in shared_tissues:

        a = (mask_a & obs[tissue_col].eq(tissue)).to_numpy()
        b = (mask_b & obs[tissue_col].eq(tissue)).to_numpy()

        if not np.any(a) or not np.any(b):
            continue

        Pa, _ = _donor_profiles(
            X[a],
            obs.loc[a].reset_index(drop=True),
            donor_col,
        )

        Pb, _ = _donor_profiles(
            X[b],
            obs.loc[b].reset_index(drop=True),
            donor_col,
        )

        ref_a = Pa.mean(axis=0)
        ref_b = Pb.mean(axis=0)

        w = Pa.shape[0] / (Pa.shape[0] + Pb.shape[0])

        between.append(_js_divergence(ref_a, ref_b, weight=w))

        # Give more importance to comparisons supported by more donors.
        weights.append(min(Pa.shape[0], Pb.shape[0]))
        donor_noise.extend(_pairwise_js(Pa))
        donor_noise.extend(_pairwise_js(Pb))

    if not between:
        return {
            "between_js": 0.0,
            "donor_noise": fallback_noise,
            "ratio": 0.0,
            "shared_tissues": 0,
            "reason": "insufficient_overlap",
        }

    between_js = np.average(between, weights=weights)

    if donor_noise:
        noise = float(np.median(donor_noise))
    else:
        noise = fallback_noise

    noise = max(noise, noise_floor)

    return {
        "between_js": float(between_js),
        "donor_noise": float(noise),
        "ratio": float(between_js / noise),
        "shared_tissues": len(shared_tissues),
        "reason": "observed",
    }


def _agglomerate_family(
    X,
    obs,
    labels,
    family,
    type_col,
    donor_col,
    tissue_col,
    global_noise,
    ratio_threshold,
    noise_floor,
    ):
    """
    Agglomerative information-bottleneck-like compression.

    Repeatedly merge the pair whose conditional JSD is smallest relative
    to donor noise.
    """

    clusters = [frozenset([label]) for label in sorted(labels)]

    family_mask = obs[type_col].isin(labels).to_numpy()

    if np.any(family_mask):
        family_noise = _within_tissue_donor_noise(
            X[family_mask],
            obs.loc[family_mask].reset_index(drop=True),
            donor_col,
            tissue_col)
    else:
        family_noise = np.nan

    if not np.isfinite(family_noise):
        family_noise = global_noise

    history = []

    while len(clusters) > 1:

        candidates = []

        for i, j in itertools.combinations(range(len(clusters)), 2):

            score = _subtype_pair_score(
                X=X,
                obs=obs,
                cluster_a=clusters[i],
                cluster_b=clusters[j],
                type_col=type_col,
                donor_col=donor_col,
                tissue_col=tissue_col,
                fallback_noise=family_noise,
                noise_floor=noise_floor,
            )

            candidates.append(
                (score["ratio"],
                 i,
                 j,
                 score))

        ratio, i, j, score = min(candidates, key=lambda x: x[0])

        # The remaining subtype distinction carries more information
        # than permitted by donor-level variability.
        if ratio > ratio_threshold:
            break

        left = clusters[i]
        right = clusters[j]

        merged = left | right

        history.append(
            {
                "family": family,
                "left": ";".join(sorted(left)),
                "right": ";".join(sorted(right)),
                "merged": ";".join(sorted(merged)),
                **score,
            }
        )

        clusters = [cluster for k, cluster in enumerate(clusters) if k not in (i, j)]
        clusters.append(merged)

    return clusters, history


# Tissue information

def _test_tissue_information(
    X,
    obs,
    donor_col,
    tissue_col,
    min_tissue_donors,
    global_noise,
    ratio_threshold,
    noise_floor,
    ):
    """
    Test I(bin ; tissue | merged cell type).

    A tissue split is retained only when:
      1. at least two tissues have sufficient independent donors;
      2. generalized tissue JSD exceeds donor variability.
    """

    tissue_profiles = []
    tissue_weights = []
    eligible_tissues = []
    donor_noise = []

    groups = obs.groupby(tissue_col, sort=True,).indices

    for tissue, idx in groups.items():

        idx = np.asarray(idx)

        P, donor_obs = _donor_profiles(
            X[idx],
            obs.iloc[idx].reset_index(drop=True),
            donor_col,
        )

        n_donors = len(donor_obs)

        if n_donors < min_tissue_donors:
            continue

        consensus = P.mean(axis=0)
        consensus /= consensus.sum()

        tissue_profiles.append(consensus)

        tissue_weights.append(n_donors)

        eligible_tissues.append(str(tissue))

        donor_noise.extend(_pairwise_js(P))

    if len(eligible_tissues) < 2:
        return {
            "split": False,
            "tissue_js": np.nan,
            "donor_noise": np.nan,
            "ratio": np.nan,
            "eligible_tissues": eligible_tissues,
        }

    tissue_js = _generalized_js(
        np.vstack(tissue_profiles),
        weights=tissue_weights,
    )

    if donor_noise:
        local_noise = float(np.median(donor_noise))
    else:
        local_noise = global_noise

    # Conservative: do not allow an unusually quiet cell type to cause
    # tiny tissue differences to become large ratios.
    noise = max(local_noise, global_noise, noise_floor)
    ratio = tissue_js / noise

    return {
        "split": bool(ratio > ratio_threshold),
        "tissue_js": float(tissue_js),
        "donor_noise": float(noise),
        "ratio": float(ratio),
        "eligible_tissues": eligible_tissues,
    }


# Final reference estimator

def _robust_donor_consensus(
    X,
    obs,
    donor_col,
    eps,
    ):
    """
    Equal-donor robust reference.

    Donors are normalized independently, then combined using the median
    in log-probability space.
    """

    P, donor_obs = _donor_profiles(X, obs, donor_col)
    log_P = np.log(P + eps)
    center = np.median(log_P, axis=0)
    reference = np.exp(center)

    reference /= reference.sum()

    return reference, len(donor_obs)


# Bin annotation

def _make_bin_gtf(features):

    coords = pd.Series(features, dtype="string").str.extract(
        r"^(?:chr)?([^:]+):(\d+)-(\d+)$")

    if coords.isna().any().any():
        raise ValueError("var_names must use chrN:start-end format.")

    coords.columns = ["CHROM",
                      "gene_start",
                      "gene_end"]

    coords["gene"] = np.asarray(features, dtype=str,)
    coords["gene_start"] = coords["gene_start"].astype(np.int64)
    coords["gene_end"] = coords["gene_end"].astype(np.int64)

    return coords[["gene",
                   "CHROM",
                   "gene_start",
                   "gene_end"]]


# Main API

def build_atac_reference(
    adata,
    donor_col="donor",
    tissue_col="tissue_ref",
    cell_type_col="cell_type",
    life_stage_col="life_stage",
    life_stage="Adult",
    min_cells=50,
    min_fragments=50_000,
    min_donors=1,
    min_tissue_donors=2,
    subtype_ratio_threshold=1.5,
    tissue_ratio_threshold=2.0,
    noise_floor=1e-8,
    eps=1e-12,
    return_diagnostics=False):
    """
    Construct a donor-balanced, information-theoretically compressed
    normal scATAC reference for SpaceNumbat.

    Subtype compression
    -------------------
    Atlas labels belonging to the same curated biological family are
    agglomeratively merged when their conditional Jensen-Shannon
    divergence:

        I(bin ; subtype | tissue)

    is no larger than `subtype_ratio_threshold` times donor variability.

    Tissue splitting
    ----------------
    After subtype merging, tissue-specific references are generated only
    when:

        I(bin ; tissue | cell_type)

    exceeds `tissue_ratio_threshold` times within-tissue donor variability.

    Returns
    -------
    reference
        Genomic-bin x reference-profile normalized DataFrame. Directly
        usable as SpaceNumbat ``lambdas_ref``.

    manifest
        Metadata describing final references.

    bin_gtf
        Bin annotation compatible with current SpaceNumbat utilities.

    diagnostics
        Returned only when ``return_diagnostics=True``. Contains subtype
        mapping, merge history and tissue-information tests.
    """

    # Validate metadata

    required = [donor_col,
                tissue_col,
                cell_type_col,
                life_stage_col]

    missing = [column for column in required if column not in adata.obs.columns]

    if missing:
        raise ValueError(f"Missing obs columns: {missing}")

    if adata.var_names.has_duplicates:
        raise ValueError("adata.var_names must be unique.")

    obs = adata.obs.copy()
    original_type = obs[cell_type_col].astype("string")

    keep_cells = (
        obs[donor_col].notna()
        & obs[tissue_col].notna()
        & original_type.notna()
        & ~original_type.isin(EXCLUDE_CELL_TYPES)
        )

    if life_stage is not None:
        keep_cells &= obs[life_stage_col].astype("string").eq(life_stage)

    # Autosomal genomic bins
    features_all = pd.Index(adata.var_names.astype(str))

    chrom = pd.Series(features_all).str.extract(r"^(?:chr)?([^:]+):",expand=False)
    keep_bins = chrom.isin({str(i) for i in range(1, 23)}).to_numpy()

    cell_idx = np.flatnonzero(keep_cells.to_numpy())
    bin_idx = np.flatnonzero(keep_bins)
    features = features_all[keep_bins]

    X = adata.X[cell_idx,:][:, bin_idx]

    if sp.issparse(X):
        X = X.tocsr()
    else:
        X = sp.csr_matrix(X)

    if X.data.size and np.min(X.data) < 0:
        raise ValueError("adata.X must contain raw non-negative fragment counts.")

    obs = obs.iloc[cell_idx][[donor_col, tissue_col, cell_type_col]].copy().reset_index(drop=True)
    obs["original_type"] = obs[cell_type_col].astype("string")

    clean_type = obs["original_type"].map(CLEAN_NAMES).fillna(obs["original_type"])

    obs["family"] = obs["original_type"].map(FAMILY_MAPPING).fillna(clean_type)
    obs["n_cells"] = 1

    # donor x tissue x original-type pseudobulk

    pb_X, pb_obs = _aggregate_rows(X,
                                   obs,
                                   [donor_col,
                                    tissue_col,
                                    "original_type",
                                    "family"])

    # Pseudobulks reliable enough to participate in information tests.
    good = ((pb_obs["n_cells"] >= min_cells) & (pb_obs["n_fragments"] >= min_fragments)).to_numpy()
    test_X = pb_X[good]
    test_obs = pb_obs.loc[good].reset_index(drop=True)

    if test_obs.empty:
        raise ValueError("No pseudobulks passed QC.")

    # Empirical donor-noise scale

    global_noise = _global_donor_noise(
        X=test_X,
        obs=test_obs,
        donor_col=donor_col,
        tissue_col=tissue_col,
        type_col="original_type",
        noise_floor=noise_floor,
    )

    # Information-theoretic subtype merging

    original_to_merged = {}
    merge_history = []

    all_families = sorted(pb_obs["family"].unique())

    for family in all_families:

        all_labels = sorted(pb_obs.loc[pb_obs["family"].eq(family),
                                       "original_type"].unique())

        family_test = test_obs["family"].eq(family).to_numpy()
        family_X = test_X[family_test]
        family_obs = test_obs.loc[family_test].reset_index(drop=True)

        # Single annotation -> nothing to test.
        if len(all_labels) == 1:

            original = all_labels[0]
            name = (CLEAN_NAMES.get(original,family))
            original_to_merged[original] = name

            continue

        clusters, history = _agglomerate_family(
            X=family_X,
            obs=family_obs,
            labels=all_labels,
            family=family,
            type_col="original_type",
            donor_col=donor_col,
            tissue_col=tissue_col,
            global_noise=global_noise,
            ratio_threshold=subtype_ratio_threshold,
            noise_floor=noise_floor,
        )

        merge_history.extend(history)

        # If everything collapsed, use the broad family name.
        if len(clusters) == 1:

            for original in clusters[0]:
                original_to_merged[original] = family

        else:

            # Preserve genuinely distinct subclusters.
            clusters = sorted(clusters, key=lambda x: sorted(x))

            for i, cluster in enumerate(clusters, start=1):

                if len(cluster) == 1:

                    original = next(iter(cluster))
                    name = CLEAN_NAMES.get(original, original)

                else:
                    name = f"{family} group {i}"

                for original in cluster:
                    original_to_merged[original] = name

    # 4. Reaggregate using the compressed cell-type assignments

    pb_obs = pb_obs.copy()

    pb_obs["merged_type"] = pb_obs["original_type"].map(original_to_merged)

    valid = pb_obs["merged_type"].notna().to_numpy()

    merged_X, merged_obs = _aggregate_rows(pb_X[valid],
                                           pb_obs.loc[valid].reset_index(drop=True),
                                           [donor_col,tissue_col,"merged_type"])

    # QC AFTER merging as well. This lets several sparse original
    # subtypes collectively form a reliable broad pseudobulk.
    good = ((merged_obs["n_cells"] >= min_cells)
            & (merged_obs["n_fragments"] >= min_fragments)).to_numpy()

    merged_X = merged_X[good]
    merged_obs = (merged_obs.loc[good].reset_index(drop=True))

    # Test whether tissue adds information after cell-type merging

    references = {}
    manifest = []
    tissue_tests = []

    source_types = (
        pd.DataFrame(
            {"original_type": list(original_to_merged.keys()),
             "merged_type": list(original_to_merged.values())}).groupby("merged_type")["original_type"].agg(list).to_dict())

    groups = merged_obs.groupby("merged_type", sort=True).indices

    for merged_type, idx in groups.items():

        idx = np.asarray(idx)
        type_X = merged_X[idx]
        type_obs = (merged_obs.iloc[idx].reset_index(drop=True))

        tissue_result = _test_tissue_information(
            X=type_X,
            obs=type_obs,
            donor_col=donor_col,
            tissue_col=tissue_col,
            min_tissue_donors=min_tissue_donors,
            global_noise=global_noise,
            ratio_threshold=tissue_ratio_threshold,
            noise_floor=noise_floor,
        )

        tissue_tests.append(
            {
                "cell_type": merged_type,
                "split": tissue_result["split"],
                "tissue_js": tissue_result["tissue_js"],
                "donor_noise": tissue_result["donor_noise"],
                "ratio": tissue_result["ratio"],
                "eligible_tissues": ";".join(tissue_result["eligible_tissues"]),
            })

        # Tissue carries reproducible information

        if tissue_result["split"]:

            for tissue in tissue_result["eligible_tissues"]:

                mask = type_obs[tissue_col].astype(str).eq(tissue).to_numpy()

                ref, n_donors = _robust_donor_consensus(
                    X=type_X[mask],
                    obs=type_obs.loc[mask].reset_index(drop=True),
                    donor_col=donor_col,
                    eps=eps,
                )

                if n_donors < min_donors:
                    continue

                name = (f"{merged_type}|{tissue}")

                references[name] = ref

                manifest.append(
                    {
                        "profile": name,
                        "cell_type": merged_type,
                        "scope": "tissue",
                        "tissue": tissue,
                        "n_donors": n_donors,
                        "n_cells": int(type_obs.loc[mask, "n_cells"].sum()),
                        "n_fragments": int(type_obs.loc[mask,"n_fragments"].sum()),
                        "source_cell_types": ";".join(source_types.get(merged_type,[])),
                        "tissue_js": tissue_result["tissue_js"],
                        "tissue_noise": tissue_result["donor_noise"],
                        "tissue_ratio": tissue_result["ratio"]})

        # Tissue does NOT add meaningful information

        else:

            ref, n_donors = _robust_donor_consensus(
                X=type_X,
                obs=type_obs,
                donor_col=donor_col,
                eps=eps,
            )

            if n_donors < min_donors:
                continue

            references[merged_type] = ref
            manifest.append(
                {
                    "profile": merged_type,
                    "cell_type": merged_type,
                    "scope": "pan_tissue",
                    "tissue": "pan_tissue",
                    "n_donors": n_donors,
                    "n_cells": int(type_obs["n_cells"].sum()),
                    "n_fragments": int(type_obs["n_fragments"].sum()),
                    "source_cell_types": ";".join(source_types.get(merged_type,[])),
                    "tissue_js": tissue_result["tissue_js"],
                    "tissue_noise": tissue_result["donor_noise"],
                    "tissue_ratio": tissue_result["ratio"]})

    if not references:
        raise ValueError("No final ATAC references were generated.")

    # Final SpaceNumbat-compatible matrix

    reference = pd.DataFrame(references, index=features)
    manifest = pd.DataFrame(manifest)
    bin_gtf = _make_bin_gtf(features)

    if not return_diagnostics:
        return (reference,manifest,bin_gtf)

    cell_type_map = pd.DataFrame(
        {"original_cell_type": list(original_to_merged.keys()),
         "merged_cell_type": list(original_to_merged.values())})

    cell_type_map["family"] = (cell_type_map["original_cell_type"]
                               .map(FAMILY_MAPPING)
                               .fillna(cell_type_map["original_cell_type"]
                                       .map(CLEAN_NAMES).fillna(cell_type_map["original_cell_type"])))

    diagnostics = {"cell_type_map": cell_type_map,
                   "merge_history": pd.DataFrame(merge_history),
                   "tissue_tests": pd.DataFrame(tissue_tests),
                   "global_donor_js": global_noise}

    return (
        reference,
        manifest,
        bin_gtf,
        diagnostics,
    )