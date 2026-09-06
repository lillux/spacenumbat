#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:47:14 2026

@author: carlino.calogero
"""

from pathlib import Path

import pandas as pd
import numpy as np

import natsort

import anndata as ad
from scipy.sparse import csr_matrix

import copy
from spacenumbat import diagnostics

from spacenumbat._log import get_logger
log = get_logger(__name__)


def get_chrom_sizes_from_fasta_index(fasta_fai_path: str,
                                     contigs=None,
                                     out_path: str | None = None):
    
    chrom_sizes = pd.read_table(fasta_fai_path, 
                                header=None, 
                                usecols=[0, 1], 
                                names=["CHROM", "length"])

    chrom_sizes["CHROM"] = chrom_sizes["CHROM"].astype(str).str.strip()
    chrom_sizes["length"] = pd.to_numeric(chrom_sizes["length"], errors="raise").astype(np.int64)

    if contigs is not None:
        contigs = [str(x) for x in contigs]
        missing = pd.Index(contigs).difference(chrom_sizes["CHROM"])

        if len(missing):
            raise ValueError(f"Contigs absent from FAI: {missing.tolist()}")

        chrom_sizes = chrom_sizes.set_index("CHROM").loc[contigs].reset_index()

    if out_path is not None:
        chrom_sizes.to_csv(out_path, sep="\t", index=False)

    return chrom_sizes


def _get_snap_chrom_sizes(value):

    if isinstance(value, dict):
        sizes = value

    else:
        sizes = getattr(value, "chrom_sizes", None)

        if not isinstance(sizes, dict):
            raise TypeError("snap_chrom_sizes must be either "
                            "a SnapATAC2 Genome object or "
                            "dict[str, int].")

    sizes = {str(chrom): int(length) for chrom, length in sizes.items()}

    if not sizes:
        raise ValueError("snap_chrom_sizes is empty.")

    if any(length <= 0 for length in sizes.values()):
        raise ValueError("All SnapATAC chromosome lengths "
                         "must be positive.")
    return sizes


def get_custom_gtf_binning(genome_spec, bin_size: int = 220_000):

    rows = []

    for row in genome_spec.chrom_sizes.itertuples():

        chrom = str(row.CHROM)
        source_chrom = str(row.source_chrom)
        chrom_length = int(row.length)

        for start in range(0, chrom_length, bin_size):

            end = min(start + bin_size, chrom_length)

            # Internal representation.
            bin_id = (f"{chrom}:{start}-{end}")

            # Representation used for files whose contigs
            # follow the original FASTA naming.
            source_bin_id = (f"{source_chrom}:{start}-{end}")

            rows.append({
                "bin_id": bin_id,
                "source_bin_id": source_bin_id,
                "CHROM": chrom,
                "start": start,
                "end": end,
                "width": end - start,
            })

    return pd.DataFrame(rows)


def get_gene_bin_intersection(gtf: pd.DataFrame, gtf_binned: pd.DataFrame) -> pd.DataFrame:

    gtf = gtf.copy()
    bins = gtf_binned.copy()

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
    log.info(f"RNA binning completed | cells={adata_bin.n_obs} | bins={adata_bin.n_vars}")
    return adata_bin


def get_atac_binning(fragments_path: str,
                     genomic_regions: list[str],
                     barcodes,
                     chrom_sizes,
                     counting_strategy: str = "fragment",
                     min_num_fragments: int = 0):
    try:
        import snapatac2 as snap
    except ImportError as exc:
        raise ImportError("ATAC preprocessing requires SnapATAC2.") from exc

    if isinstance(chrom_sizes, pd.DataFrame):
        chrom_sizes = dict(zip(chrom_sizes["CHROM"].astype(str),
                               chrom_sizes["length"].astype(int)))

    adata_atac = snap.pp.import_fragments(fragments_path,
                                          chrom_sizes=chrom_sizes,
                                          sorted_by_barcode=False,
                                          whitelist=barcodes,
                                          min_num_fragments=min_num_fragments)

    adata_atac = snap.pp.make_peak_matrix(adata_atac,
                                          inplace=False,
                                          chunk_size=500,
                                          counting_strategy=counting_strategy,
                                          use_rep=genomic_regions)

    return adata_atac[:,adata_atac.var.sort_index(key=natsort.natsort_keygen()).index]
    

def get_binned_ref(ref_df, gene_bin_intersection, gene_id="gene", bin_id="bin_id",):
    
    ref_df = ref_df.apply(pd.to_numeric, errors="raise").astype(np.float64)
    ref_cols = ref_df.columns
    ref = pd.merge(ref_df, gene_bin_intersection, left_index=True, right_on=gene_id)
    ref_bin_df = ref.groupby(bin_id, sort=False)[ref_cols].sum().sort_index(key=natsort.natsort_keygen())
    col_sums = ref_bin_df.sum(axis=0)

    if (col_sums <= 0).any():
        bad = col_sums.index[col_sums <= 0].tolist()
        raise ValueError("RNA reference profiles with zero total signal "
                         f"after genomic binning: {bad}")

    ref_bin_df = ref_bin_df.div(col_sums, axis=1)

    return ref_bin_df.astype(np.float64)


def get_binned_gtf(binning):

    return diagnostics.validate_annotation(
        pd.DataFrame({"CHROM": binning["CHROM"].astype("string"),
                      "gene_start": binning["start"].astype(np.int64),
                      "gene_end": binning["end"].astype(np.int64),
                      "gene": binning["bin_id"].astype(str)}))


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


def apply_cell_manifest(adata: ad.AnnData, cell_manifest) -> ad.AnnData:

    manifest = _load_table(cell_manifest).copy()
    required = {"barcode", "cell_id"}
    missing = required.difference(manifest.columns)

    if missing:
        raise ValueError(f"Cell manifest is missing columns: {sorted(missing)}")
    obs = adata.obs.copy()

    obs["barcode"] = pd.Index(adata.obs_names).astype(str).str.strip()
    manifest["barcode"] = manifest["barcode"].astype(str).str.strip()
    manifest["cell_id"] = manifest["cell_id"].astype(str).str.strip()

    has_modality = "modality" in manifest.columns
    # Multiple modalities cannot safely be matched using barcode alone.
    n_modalities = obs["modality"].nunique() if "modality" in obs.columns else 1

    if n_modalities > 1 and not has_modality:
        raise ValueError("A 'modality' column is required in the cell manifest "
                         "when the count matrix contains multiple modalities.")

    if has_modality:

        if "modality" not in obs.columns:
            raise ValueError("Cell manifest contains modality information but "
                             "adata.obs does not contain a 'modality' column.")

        manifest["modality"] = manifest["modality"].astype(str).str.strip().str.lower()
        obs["modality"] = obs["modality"].astype(str).str.strip().str.lower()
        key_cols = ["modality", "barcode"]
        duplicated = manifest.duplicated(key_cols, keep=False)

        if duplicated.any():
            raise ValueError("The cell manifest contains duplicated "
                             "(modality, barcode) combinations.")

        mapping = manifest.set_index(key_cols)["cell_id"].astype(str)
        keys = pd.MultiIndex.from_frame(obs[key_cols])

    else:
        key_cols = ["barcode"]
        duplicated = manifest["barcode"].duplicated(keep=False)

        if duplicated.any():
            raise ValueError("The cell manifest contains duplicated barcodes. "
                             "A modality-aware manifest is required when "
                             "barcodes are not unique.")

        mapping = manifest.set_index("barcode")["cell_id"].astype(str)
        keys = pd.Index(obs["barcode"], name="barcode")

    cell_ids = mapping.reindex(keys)

    if cell_ids.isna().any():

        missing_mask = cell_ids.isna().to_numpy()
        missing_rows = obs.loc[missing_mask, key_cols].head()

        raise ValueError("Some count-matrix cells could not be matched "
                         f"using {key_cols}. Examples:\n"
                         f"{missing_rows}")

    obs.index = pd.Index(cell_ids.to_numpy(), name="cell_id")

    if obs.index.has_duplicates:
        raise ValueError("Cell IDs are not unique after applying the manifest.")

    adata = adata.copy()
    adata.obs = obs

    return adata


def validate_chrom_sizes(chrom_sizes):
    
    chrom_sizes = chrom_sizes.copy()
    required = {"CHROM", "length"}
    missing = required.difference(chrom_sizes.columns)

    if missing:
        raise ValueError(f"chrom_sizes is missing columns: {sorted(missing)}")

    chrom_sizes["CHROM"] = chrom_sizes["CHROM"].astype(str).str.strip()
    chrom_sizes["length"] = pd.to_numeric(chrom_sizes["length"],errors="raise",).astype(np.int64)

    if (chrom_sizes["length"] <= 0).any():
        raise ValueError("All chromosome lengths must be positive.")
    if chrom_sizes["CHROM"].duplicated().any():
        raise ValueError("chrom_sizes contains duplicated chromosomes.")

    return chrom_sizes


def prepare_unpaired_multiome_inputs(
    mode: str,
    binning: str,
    genome_spec,
    snap_chrom_sizes,
    source_gtf: pd.DataFrame | None,
    rna_reference: pd.DataFrame | None,
    atac_reference=None,
    numbat_binning=None,
    custom_binning=None,
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
    
    log.info(f"Preparing binned input | mode={mode} | binning={binning} | "
             f"RNA={has_rna} | ATAC={has_atac}")

    # Binning
    if binning == "numbat":

        if numbat_binning is None:
            raise ValueError("Numbat binning table was not provided.")

        current_binning = _load_table(numbat_binning)
        
        log.info(f"Genomic binning prepared | bins={len(current_binning)} | "
                 f"chromosomes={current_binning['CHROM'].nunique()}")

    elif binning == "fixed":

        if custom_binning is not None:
    
            current_binning = _load_table(custom_binning)
        
        else:
    
            if not isinstance(bin_size, int):
                raise ValueError("bin_size must be an integer when binning='fixed'")
                
            if bin_size <= 0:
                raise ValueError("bin_size must be positive.")

            current_binning = get_custom_gtf_binning(genome_spec=genome_spec,
                                                     bin_size=bin_size)
                
    else:
        raise ValueError("binning must be 'numbat' or 'fixed'.")
        
    current_binning = genome_spec.normalize_table(current_binning,
                                                  table_name="genomic binning")
    
    if "source_bin_id" not in current_binning.columns:

        source_chrom = current_binning["CHROM"].map(genome_spec.canonical_to_source)
        
        current_binning["source_bin_id"] = (
            source_chrom.astype(str)
            + ":"
            + current_binning["start"].astype(str)
            + "-"
            + current_binning["end"].astype(str)
            )

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

        required_rna = {"rna_mtx_path": rna_mtx_path,
                        "rna_barcodes_path": rna_barcodes_path,
                        "rna_features_path": rna_features_path}

        missing = [name for name, value in required_rna.items() if value is None]

        if missing:
            raise ValueError(f"Missing RNA inputs: {missing}")

        if source_gtf is None:
            raise ValueError("source_gtf is required for RNA binning.")

        if rna_reference is None:
            raise ValueError("rna_reference is required for RNA binning.")

        gene_intersect = get_gene_bin_intersection(source_gtf, current_binning)
        
        log.info(f'RNA gene-to-bin mapping | input_genes={source_gtf["gene"].nunique()} | '
                 f'mapped_genes={gene_intersect["gene"].nunique()} | '
                 f'populated_bins={gene_intersect["bin_id"].nunique()}')
        
        mapped_fraction = (gene_intersect["gene"].nunique() / source_gtf["gene"].nunique())
        if mapped_fraction < 0.8:
            log.warning(f"Only {100 * mapped_fraction}% of annotated genes mapped to genomic bins")
        
        adata_rna = get_rna_binning(mtx_path=rna_mtx_path,
                                    barcodes_path=rna_barcodes_path,
                                    features_path=rna_features_path,
                                    gene_binned=gene_intersect)

        rna_ref = get_binned_ref(rna_reference, gene_bin_intersection=gene_intersect)
        log.info(f"RNA reference binned | bins={rna_ref.shape[0]} | reference_profiles={rna_ref.shape[1]}")
        
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
            genomic_regions=current_binning["source_bin_id"].tolist(),
            barcodes=atac_barcodes_path,
            chrom_sizes=genome_spec.source_chromosome_lengths,
            counting_strategy="fragment",
            min_num_fragments=min_num_fragments,
        )

        atac_ref = _load_table(atac_reference, index_col=0)
        atac_ref.index = atac_ref.index.astype(str)

        source_to_internal = current_binning.set_index("source_bin_id")["bin_id"]
        if source_to_internal.index.has_duplicates:
            raise ValueError("source_bin_id contains duplicated bins.")
        
        expected_set = set(expected_bins)
        normalized_index = []
        unknown = []
        
        for feature in atac_ref.index:
        
            if feature in expected_set:
                # Already using internal bin IDs.
                normalized_index.append(feature)
            
            elif feature in source_to_internal.index:
                # External/source chromosome namespace.
                normalized_index.append(source_to_internal.loc[feature])
        
            else:
                unknown.append(feature)
        
        if unknown:
        
            raise ValueError("The ATAC reference contains bins that "
                             "do not belong to the selected genome/binning. "
                             f"Examples: {unknown[:5]}")
        
        atac_ref.index = pd.Index(normalized_index, name="bin_id")
        
        if atac_ref.index.has_duplicates:
            raise ValueError("The ATAC reference contains duplicated "
                             "bins after genome normalization.")
            
        snap_features = pd.Index(adata_atac.var_names.astype(str))
        unexpected = snap_features.difference(source_to_internal.index)
        
        if len(unexpected):
            raise ValueError("SnapATAC2 returned regions absent from "
                             "the selected SpaceNumbat binning. "
                             f"Examples: {unexpected[:5].tolist()}")
        
        adata_atac.var_names = pd.Index(
            source_to_internal
            .reindex(snap_features)
            .to_numpy(),
            name="bin_id",
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



