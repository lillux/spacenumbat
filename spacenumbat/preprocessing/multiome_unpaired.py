#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:47:14 2026

@author: carlino.calogero
"""

from typing import List
from collections import OrderedDict
from itertools import chain
from pathlib import Path

import pandas as pd
import numpy as np

import natsort

import anndata as ad
from scipy.sparse import csr_matrix

import copy
from spacenumbat import diagnostics


def get_minimal_chrom_size_from_fasta_index(fasta_fai_path:str,
                                            chrom_accepted:List[str]="auto", 
                                            out_path:str=None):

    if chrom_accepted == "auto":
        chrom_accepted = [str(i) for i in range(1,23)]

    raw_chrom_size_df = pd.read_table(fasta_fai_path, header=None)
    chrom = raw_chrom_size_df[0].astype(str).str.replace(r"^chr", "", regex=True)
    raw_chrom_size_df = raw_chrom_size_df[chrom.isin(chrom_accepted)].copy()
    raw_chrom_size_df = raw_chrom_size_df.iloc[:,:2]
    raw_chrom_size_df.columns = ["CHROM", "length"]
    raw_chrom_size_df = raw_chrom_size_df.sort_values("CHROM", key=natsort.natsort_keygen()).reset_index(drop=True)

    if out_path is not None:
    
        raw_chrom_size_df.to_csv(out_path, 
                                 sep="\t", 
                                 index=False)
    return raw_chrom_size_df


def get_custom_gtf_binning(chrom_size_df:pd.DataFrame, bin_size:int=220_000):

    binned_dict = {}
    for _, row in chrom_size_df.iterrows():
        current_n_bins = np.int64(np.ceil(row.length / bin_size))
        current_cum_sum = 0
    
        for bin_running in range(current_n_bins):
            proposed_bin_end = current_cum_sum + bin_size
            current_bin_end = proposed_bin_end if (proposed_bin_end < row.length) else row.length
            bin_id = f"{row.CHROM}:{current_cum_sum}-{current_bin_end}"
    
            binned_dict[bin_id] = {"bin_id":bin_id,
                                   "CHROM":str(row.CHROM)[3:] if str(row.CHROM).startswith("chr") else str(row.CHROM), 
                                   "start":current_cum_sum, 
                                   "end":current_bin_end,
                                   "width":current_bin_end-current_cum_sum}
            current_cum_sum = proposed_bin_end
    
    bin_genome = pd.DataFrame(binned_dict).T

    return bin_genome


def get_gene_bin_intersection(gtf: pd.DataFrame, gtf_binned: pd.DataFrame) -> pd.DataFrame:

    gtf = gtf.copy()
    bins = gtf_binned.copy()

    gtf["CHROM"] = gtf["CHROM"].astype("string").str.replace(r"^chr", "", regex=True)
    bins["CHROM"] = bins["CHROM"].astype("string").str.replace(r"^chr", "", regex=True)

    out = []

    bins_by_chrom = {chrom: x for chrom, x in bins.groupby("CHROM", sort=False)}

    for chrom, genes in gtf.groupby("CHROM", sort=False):

        chrom_bins = bins_by_chrom.get(chrom)

        if chrom_bins is None:
            continue

        for row_idx, row in genes.iterrows():

            candidate_bins = chrom_bins[
                (chrom_bins["start"] < row.gene_end) &
                (chrom_bins["end"] > row.gene_start)]

            if candidate_bins.empty:
                continue

            overlap = (
                np.minimum(candidate_bins["end"].to_numpy(), row.gene_end)
                - np.maximum(candidate_bins["start"].to_numpy(), row.gene_start))

            best = np.argmax(overlap)
            assigned_bin = candidate_bins.index[best]

            out.append({
                "gene": row.gene,
                "bin_id": assigned_bin,
            })

    return pd.DataFrame(out)


def get_rna_binning(mtx_path:str, barcodes_path:str, features_path:str, gene_binned:pd.DataFrame):
    
    mtx = ad.io.read_mtx(mtx_path)
    barcodes = pd.read_table(barcodes_path, header=None)
    feature = pd.read_table(features_path, header=None)

    adata = ad.AnnData(X=mtx.X.T, obs=barcodes, var=feature)

    adata.obs = adata.obs.rename({0:"barcodes"}, axis=1).set_index("barcodes", drop=True)
    adata.var = adata.var.rename({0:"gene_id", 1:"gene_name", 2:"modality"}, axis=1).set_index("gene_id", drop=False)
    adata = adata[:,~adata.var.gene_name.duplicated()]
    adata = adata[:,adata.var["gene_name"].isin(gene_binned.gene)]
    adata.var = pd.merge(adata.var, gene_binned, left_on="gene_name", right_on="gene").set_index("gene_name", drop=True)

    bin_counts = {}
    for bin_idx, group in adata.var.groupby("bin_id"):
        bin_counts[bin_idx] = np.array(adata[:,group.index].X.sum(1)).ravel()

    binned_rna_df = pd.DataFrame(bin_counts)
    adata_bin = ad.AnnData(X=csr_matrix(binned_rna_df.values), obs=adata.obs, var=pd.DataFrame([pd.Series(binned_rna_df.columns)]).T)
    adata_bin.var = adata_bin.var.rename({0:"bin_id"}, axis=1).set_index("bin_id", drop=True)
    adata_bin = adata_bin[:,adata_bin.var.sort_index(key=natsort.natsort_keygen()).index]

    return adata_bin


def get_atac_binning(fragments_path:str,
                     genomic_regions:List,
                     barcodes:str,
                     counting_strategy:str="fragment",
                     min_num_fragments:int=0):
    try:
        import snapatac2 as snap

    except ImportError as exc:
        raise ImportError("ATAC preprocessing requires SnapATAC2. "
                          "Install SpaceNumbat with ATAC dependencies.") from exc

    adata_atac = snap.pp.import_fragments(fragments_path,
                                          chrom_sizes=snap.genome.hg38,
                                          sorted_by_barcode=False,
                                          whitelist=barcodes, 
                                          min_num_fragments=min_num_fragments)
    
    adata_atac = snap.pp.make_peak_matrix(adata_atac,
                                          inplace=False, 
                                          chunk_size=500,
                                          counting_strategy=counting_strategy,
                                          use_rep=genomic_regions)

    adata_atac = adata_atac[:,adata_atac.var.sort_index(key=natsort.natsort_keygen()).index]

    return adata_atac
    

def get_binned_ref(ref_df, gene_bin_intersection, gene_id="gene", bin_id="bin_id"):
    
    ref = pd.merge(ref_df,
                   gene_bin_intersection,
                   left_index=True, 
                   right_on=gene_id).drop(gene_id, axis=1)
    bin_dict = {}
    for idx, group in ref.groupby(bin_id, sort=False):
        bin_dict[idx] = group.sum(0)

    ref_bin_df = pd.DataFrame(bin_dict).T.drop(bin_id, axis=1).sort_index(key=natsort.natsort_keygen())
    ref_bin_df = ref_bin_df.div(ref_bin_df.sum(axis=0),axis=1)
    
    return ref_bin_df


def get_binned_gtf(binning: pd.DataFrame) -> pd.DataFrame:

    gtf = binning.copy()

    if "bin_id" not in gtf.columns:
        gtf["bin_id"] = gtf.index.astype(str)

    gtf = pd.DataFrame({"CHROM": (gtf["CHROM"].astype(str).str.replace(r"^chr", "", regex=True)),
                        "gene_start": gtf["start"].astype(np.int64),
                        "gene_end": gtf["end"].astype(np.int64),
                        "gene": gtf["bin_id"].astype(str)})

    return diagnostics.validate_annotation(gtf)


def _load_table(x, index_col=None):

    if isinstance(x, pd.DataFrame):

        out = x.copy()

        if index_col is not None:

            if isinstance(index_col, int):
                index_name = out.columns[index_col]
            else:
                index_name = index_col

            out = out.set_index(index_name)

        return out

    if isinstance(x, (str, Path)):
        return pd.read_table(x, index_col=index_col)

    raise TypeError("Expected a pandas.DataFrame or path to a TSV file.")


def apply_cell_manifest(
    adata: ad.AnnData,
    cell_manifest,
    ) -> ad.AnnData:

    manifest = _load_table(cell_manifest)

    required = {"barcode", "cell_id", "modality"}
    missing = required.difference(manifest.columns)

    if missing:
        raise ValueError(f"Cell manifest is missing columns: {sorted(missing)}")

    key_cols = ["modality", "barcode"]

    duplicated = manifest.duplicated(key_cols, keep=False)

    if duplicated.any():
        raise ValueError(
            "The cell manifest contains duplicated "
            "(modality, barcode) combinations. "
            "The current unpaired API expects one RNA and "
            "one ATAC library per analysis."
        )

    obs = adata.obs.copy()

    if "modality" not in obs.columns:
        raise ValueError("adata.obs must contain a 'modality' column.")

    obs["barcode"] = adata.obs_names.astype(str)
    mapping = manifest.set_index(key_cols)["cell_id"].astype(str)
    keys = pd.MultiIndex.from_frame(obs[key_cols].astype(str))
    cell_ids = mapping.reindex(keys)

    if cell_ids.isna().any():
        missing_rows = obs.loc[cell_ids.isna().to_numpy(), key_cols].head()

        raise ValueError("Some count-matrix cells could not be matched "
                         "to the allele cell manifest. Examples:\n"
                         f"{missing_rows}")

    obs.index = pd.Index(cell_ids.to_numpy(), name="cell_id",)

    if obs.index.has_duplicates:
        raise ValueError("Cell IDs are not unique after applying the manifest.")

    adata = adata.copy()
    adata.obs = obs

    return adata


def prepare_unpaired_multiome_inputs(
    mode: str,
    binning: str,
    source_gtf: pd.DataFrame | None,
    rna_reference: pd.DataFrame | None,
    atac_reference=None,
    numbat_binning=None,
    custom_binning=None,
    chrom_size_fai_path: str | None = None,
    bin_size: int | None = None,
    rna_mtx_path: str | None = None,
    rna_barcodes_path: str | None = None,
    rna_features_path: str | None = None,
    atac_fragments_path: str | None = None,
    atac_barcodes_path: str | None = None,
    cell_manifest=None,
    min_num_fragments: int = 0,
    max_cells_per_modality: int | None = None,
    seed: int = 28,
    ):

    aliases = {"atac": "atac_bin",}
    mode = aliases.get(mode, mode)
    valid_modes = {"rna_bin", "atac_bin", "combined"}

    if mode not in valid_modes:
        raise ValueError(f"Unsupported multiome mode {mode!r}. "
                         f"Expected one of {sorted(valid_modes)}.")

    has_rna = mode in {"rna_bin", "combined"}
    has_atac = mode in {"atac_bin", "combined"}

    # Binning
    if binning == "numbat":

        if numbat_binning is None:
            raise ValueError("Numbat binning table was not provided.")

        current_binning = _load_table(numbat_binning)

    elif binning == "fixed":

        if custom_binning is not None:

            current_binning = _load_table(custom_binning)

        else:

            if chrom_size_fai_path is None:
                raise ValueError("chrom_size_fai_path is required for "
                                 "fixed binning unless custom_binning is supplied.")

            if not isinstance(bin_size, int):
                raise ValueError("bin_size must be an integer when "
                                 "binning='fixed'.")

            chrom_sizes = get_minimal_chrom_size_from_fasta_index(chrom_size_fai_path)
            current_binning = get_custom_gtf_binning(chrom_sizes, bin_size=bin_size,)

    else:
        raise ValueError("binning must be 'numbat' or 'fixed'.")

    required_binning = {
        "bin_id",
        "CHROM",
        "start",
        "end",
        }

    missing = required_binning.difference(current_binning.columns)

    if missing:
        raise ValueError(f"Binning table is missing columns: {sorted(missing)}")

    current_binning = current_binning.copy()
    current_binning["bin_id"] = current_binning["bin_id"].astype(str)
    current_binning.index = current_binning["bin_id"]
    expected_bins = pd.Index(current_binning["bin_id"], name="bin_id")

    # RNA
    if has_rna:

        required_rna = {
            "rna_mtx_path": rna_mtx_path,
            "rna_barcodes_path": rna_barcodes_path,
            "rna_features_path": rna_features_path,
        }

        missing = [name for name, value in required_rna.items() if value is None]

        if missing:
            raise ValueError(f"Missing RNA inputs: {missing}")

        if source_gtf is None:
            raise ValueError("source_gtf is required for RNA binning.")

        if rna_reference is None:
            raise ValueError("rna_reference is required for RNA binning.")

        gene_intersect = get_gene_bin_intersection(source_gtf, current_binning)

        adata_rna = get_rna_binning(
            mtx_path=rna_mtx_path,
            barcodes_path=rna_barcodes_path,
            features_path=rna_features_path,
            gene_binned=gene_intersect,
        )

        rna_ref = get_binned_ref(
            rna_reference,
            gene_bin_intersection=gene_intersect,
        )

    # ATAC
    if has_atac:

        if atac_fragments_path is None:
            raise ValueError("atac_fragments_path is required.")

        if atac_barcodes_path is None:
            raise ValueError("atac_barcodes_path is required.")

        if atac_reference is None:
            raise ValueError("An ATAC reference is required.")

        genomic_regions = expected_bins.tolist()

        adata_atac = get_atac_binning(
            fragments_path=atac_fragments_path,
            genomic_regions=genomic_regions,
            barcodes=atac_barcodes_path,
            counting_strategy="fragment",
            min_num_fragments=min_num_fragments,
        )

        atac_ref = _load_table(atac_reference, index_col=0)
        atac_ref.index = atac_ref.index.astype(str)

        # IMPORTANT:
        # every reference feature must belong to the selected
        # genomic binning. Missing bins are allowed because
        # later we take the shared feature space.
        unexpected = atac_ref.index.difference(expected_bins)

        if len(unexpected) > 0:
            raise ValueError(
                "The ATAC reference was generated using an "
                "incompatible genomic binning. "
                f"Examples of incompatible bins: "
                f"{unexpected[:5].tolist()}"
            )

    # Optional cell subsampling
    rng = np.random.default_rng(seed)

    def subsample(adata):
        if (max_cells_per_modality is None or adata.n_obs <= max_cells_per_modality):
            return adata

        idx = np.sort(rng.choice(adata.n_obs, size=max_cells_per_modality, replace=False))

        return adata[idx].copy()

    if has_rna:
        adata_rna = subsample(adata_rna)

    if has_atac:
        adata_atac = subsample(adata_atac)

    # Combine observations
    if mode == "rna_bin":

        count_mat = adata_rna
        count_mat.obs["modality"] = "rna"

        reference = rna_ref

    elif mode == "atac_bin":

        count_mat = adata_atac
        count_mat.obs["modality"] = "atac"

        reference = atac_ref

    else:

        count_mat = ad.concat({"rna": adata_rna,
                               "atac": adata_atac},
                              label="modality",
                              axis="obs",
                              join="inner")

        # Avoid ambiguous reference column names.
        rna_ref = rna_ref.copy()
        atac_ref = atac_ref.copy()

        rna_ref.columns = [f"rna::{x}" for x in rna_ref.columns]
        atac_ref.columns = [f"atac::{x}" for x in atac_ref.columns]
        reference = pd.concat([rna_ref, atac_ref], axis=1, join="inner")

    # Align count/reference feature space
    common_bins = expected_bins[expected_bins.isin(count_mat.var_names)
                                & expected_bins.isin(reference.index)]

    if len(common_bins) == 0:
        raise ValueError("No common genomic bins remain between "
                         "count matrix and reference.")

    count_mat = count_mat[:, common_bins].copy()
    reference = reference.reindex(common_bins).copy()

    # Canonical cell identifiers
    if mode == "combined" and cell_manifest is None:
        raise ValueError("cell_manifest is required for combined unpaired "
                         "RNA/ATAC analysis.")

    if cell_manifest is not None:
        count_mat = apply_cell_manifest(count_mat, cell_manifest)

    # Inference annotation
    bin_gtf = get_binned_gtf(current_binning)

    return {
        "count_mat": count_mat,
        "lambdas_ref": reference,
        "gtf": bin_gtf,
        "binning": current_binning,
        }


def transfer_spatial_info(
    count_adata,
    spatial_adata,
    connectivity_key="spatial_connectivities",
    distance_key="weighted_adjacency",
    barcode_col="barcode"):
    """
    Transfer spatial metadata and graph information to a binned AnnData.

    The binned count object may use canonical cell IDs generated by
    pileup_n_phase.py, while spatial_adata is expected to use raw
    barcodes as obs_names.

    Raw barcodes are taken from count_adata.obs[barcode_col] when
    available; otherwise count_adata.obs_names are assumed to contain
    raw barcodes.
    """

    count_adata = count_adata.copy()

    if count_adata.obs_names.has_duplicates:
        raise ValueError("count_adata.obs_names contains duplicated cell IDs.")

    if spatial_adata.obs_names.has_duplicates:
        raise ValueError("spatial_adata.obs_names contains duplicated barcodes.")

    # Canonical cell IDs may already be present in obs_names.
    # In that case use the raw barcode retained by apply_cell_manifest().
    if barcode_col in count_adata.obs.columns:
        barcodes = pd.Index(count_adata.obs[barcode_col].astype(str), name="barcode")
    else:
        barcodes = pd.Index(count_adata.obs_names.astype(str), name="barcode")

    if barcodes.has_duplicates:
        raise ValueError("Raw barcodes used for spatial alignment are duplicated.")

    spatial_barcodes = pd.Index(spatial_adata.obs_names.astype(str))
    missing = barcodes.difference(spatial_barcodes)

    if len(missing) > 0:
        raise ValueError(f"{len(missing)} count-matrix cells could not be "
                         "matched to spatial_adata by barcode. "
                         f"Examples: {missing[:5].tolist()}")

    # Subset the spatial object to the count-matrix cells
    # and preserve the count-matrix order.
    spatial_view = spatial_adata[barcodes.tolist(), :].copy()

    # Observation metadata
    spatial_obs = spatial_view.obs.copy()
    spatial_obs.index = count_adata.obs_names

    # Preserve count-side columns in case of collisions.
    for col in spatial_obs.columns:
        if col not in count_adata.obs.columns:
            count_adata.obs[col] = spatial_obs[col].to_numpy()

    # Graph
    if connectivity_key not in spatial_view.obsp:
        raise KeyError(f"{connectivity_key!r} is missing from spatial_adata.obsp. "
                       f"Available keys: {list(spatial_adata.obsp.keys())}")

    count_adata.obsp[connectivity_key] = spatial_view.obsp[connectivity_key].copy()

    if distance_key in spatial_view.obsp:
        count_adata.obsp[distance_key] = spatial_view.obsp[distance_key].copy()

    # Coordinates / Visium metadata
    if "spatial" in spatial_view.obsm:
        count_adata.obsm["spatial"] = spatial_view.obsm["spatial"].copy()

    if "spatial" in spatial_view.uns:
        count_adata.uns["spatial"] = copy.deepcopy(spatial_view.uns["spatial"])

    return count_adata



