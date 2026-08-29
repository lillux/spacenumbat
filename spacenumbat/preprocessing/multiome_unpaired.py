#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 19:47:14 2026

@author: carlino.calogero
"""

from typing import List
import pandas as pd
import numpy as np

import natsort

import anndata as ad
import snapatac2 as snap
from scipy.sparse import csr_matrix

from spacenumbat import diagnostics

def get_minimal_chrom_size_from_fasta_index(fasta_fai_path:str,
                                            chrom_accepted:List[str]="auto", 
                                            out_path:str=None):

    if chrom_accepted == "auto":
        chrom_accepted = [str(i) for i in range(1,23)]

    raw_chrom_size_df = pd.read_table(fasta_fai_path, header=None)
    raw_chrom_size_df = raw_chrom_size_df[[i[3:] in chrom_accepted for i in raw_chrom_size_df.loc[:,0]]]
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


