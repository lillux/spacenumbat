#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:35:09 2024

@author: carlino.calogero
"""

from typing import Dict, List, Union, Sequence, Any, Optional, Iterable, Literal
from numpy.typing import NDArray, ArrayLike

import string
import re
import itertools
from collections import Counter
from functools import reduce, partial

import tqdm

import numpy as np
import pandas as pd
import anndata as ad

import pyranges as pr
from pyranges import PyRanges

import scipy
from scipy.stats import ttest_ind

import natsort
from natsort import natsorted




from statsmodels.stats.multitest import multipletests
import networkx as nx

from joblib import cpu_count, Parallel, delayed

from spacenumbat import dist_prob
from spacenumbat import hmm as hmmlib
from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("This is an info message.")


## Prepare bulk data

def annotate_genes(
    df: pd.DataFrame,
    gtf: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Annotate SNPs in `df` with gene names based on overlaps with gene regions from `gtf`.

    This function uses PyRanges to find overlaps between SNP positions and gene coordinates,
    adding a 'gene' column to the input SNP dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        SNP dataframe containing at least columns ['snp_id', 'CHROM', 'POS'].
    gtf : pandas.DataFrame
        Gene annotation dataframe containing at least columns ['gene', 'gene_start', 'gene_end', 'CHROM'].

    Returns
    -------
    pandas.DataFrame
        The input SNP dataframe `df` with an added 'gene' column, indicating the gene overlapping
        each SNP if any; otherwise, NaN.

    Notes
    -----
    - The function renames columns to conform to PyRanges expectations ('Chromosome', 'Start', 'End').
    - SNP positions are treated as zero-length intervals for overlap detection.
    - Duplicate SNPs are removed during processing to avoid redundant annotations.
    """
    
    snps = df.loc[:, ["snp_id", "CHROM", "POS"]].drop_duplicates().reset_index(drop=True)  
    snps["snp_index"] = [i for i in range(snps.shape[0])]

    snps_pr_df = pd.DataFrame({"Chromosome": snps["CHROM"].astype(str),
                               "Start": snps["POS"].astype(np.int64),
                               "End": snps["POS"].astype(np.int64),
                               "snp_index": snps["snp_index"].astype(np.int64),
                               "snp_id": snps["snp_id"],
                              })

    gtf2 = gtf.copy()
    gtf2["gene_index"] = [i for i in range(gtf2.shape[0])]
    gtf_pr_df = pd.DataFrame({"Chromosome": gtf2["CHROM"].astype(str),
                              "Start": gtf2["gene_start"].astype(np.int64),
                              "End": gtf2["gene_end"].astype(np.int64),
                              "gene": gtf2["gene"],
                              "gene_index": gtf2["gene_index"].astype(np.int64),
                             })

    snps_pr = pr.PyRanges(snps_pr_df)
    gtf_pr = pr.PyRanges(gtf_pr_df)

    # overlaps
    hits = snps_pr.join(gtf_pr).df

    if not hits.empty:
        hits = hits.sort_values(["snp_index", "gene"], 
                                kind="mergesort").drop_duplicates(subset=["snp_index"], 
                                                                  keep="first").loc[:, ["snp_index", 
                                                                                        "gene"]]
    else:
        hits = pd.DataFrame({"snp_index": snps["snp_index"], "gene": pd.NA}).iloc[0:0]

    # left join gene onto snps (by snp_index)
    snps_annot = snps.merge(hits, on="snp_index", how="left")

    # drop existing gene columns then left join
    out = df.drop(columns=[c for c in ["gene", "gene_start", "gene_end"] if c in df.columns])
    out = out.merge(snps_annot.loc[:, ["snp_id", "CHROM", "POS", "gene"]],
                    on=["snp_id", "CHROM", "POS"],
                    how="left",)
    
    out.CHROM = out.CHROM.astype("string")
    out.cell = out.cell.astype("string")
    out.gene = out.gene.astype("string")
    out.snp_id = out.snp_id.astype("string")

    return out
    

def check_anndata(count_ad:ad.AnnData, count_to_int:bool=True, fix_names:bool=True) -> ad.AnnData:
    """
    Validate and preprocess an AnnData object for downstream analysis.

    This function performs several checks and modifications on an AnnData object to ensure it is
    properly formatted for analysis:
    1. Converts the `.X` attribute to a CSC (Compressed Sparse Column) matrix if it is a dense NumPy array.
    2. Checks for duplicate gene names in `var_names` and makes them unique if required.

    Parameters
    ----------
    count_ad : AnnData
        The AnnData object containing the count matrix and associated metadata.
    fix_names : bool, optional (default: True)
        If True, modifies duplicate gene names to make them unique.
        If False, raises a ValueError when duplicate gene names are found.

    Returns
    -------
    AnnData
        The validated and possibly modified AnnData object.

    Raises
    ------
    ValueError
        If `.X` is neither a NumPy array nor a SciPy sparse matrix.
        If `fix_names` is False and duplicate gene names are present.

    """
    # Convert .X to CSC matrix if it's a dense NumPy array
    if isinstance(count_ad.X, np.ndarray):
        count_ad.X = scipy.sparse.csc_matrix(count_ad.X)
    # Raise an error if .X is neither a NumPy array nor a SciPy sparse matrix
    elif not scipy.sparse.issparse(count_ad.X):
        msg = (f'You passed an object with an .X of type {type(count_ad.X)}. '
               'count_ad.X should be a NumPy array or a SciPy sparse CSC matrix.')
        raise ValueError(msg)

    # Check for duplicate gene names and make them unique if required
    if count_ad.var_names.shape[0] != np.unique(count_ad.var_names).shape[0]:
        if fix_names:
            count_ad.var_names_make_unique()
        else:
            msg = ('Some gene names in var_names are not unique. '
                   'Please make them unique or set the argument fix_names=True.')
            raise ValueError(msg)

    return count_ad


def check_allele_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean an allele-count DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Allele dataframe expected to contain:
        ['cell','snp_id','CHROM','POS','cM','REF','ALT','AD','DP','GT','gene']

    Returns
    -------
    pandas.DataFrame
        The same dataframe, filtered to autosomes 1-22.

    Raises
    ------
    ValueError
        If mandatory columns are missing or SNP genotypes are inconsistent.
    """
    df = df.copy()
    # check column
    expected: List[str] = ["cell","snp_id","CHROM",
                           "POS","cM","REF",
                           "ALT","AD","DP",
                           "GT","gene"]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            "The allele count dataframe appears to be malformed; "
            f"missing column(s): {', '.join(missing)}. Please fix.")

    # genotype check
    snp_n_unique = df.loc[df["GT"] != "", ["snp_id", "GT"]].groupby("snp_id", 
                                                  sort=False)["GT"].nunique()
    if (snp_n_unique > 1).any():
        msg = ("Inconsistent SNP genotypes; "
               "Are cells from two different individuals mixed together?")
        #log.error(msg)
        raise ValueError(msg)

    # Strip 'chr' prefix
    # Only check if the first entry starts with "chr"
    if df["CHROM"].astype("string").str.contains(r"^chr").iloc[0]:
        df = df.assign(CHROM=df["CHROM"].astype("string").str.replace(r"^chr", "", regex=True))

    # Keep chr 1-22
    autosomes = [str(i) for i in range(1, 23)]
    df = df[df["CHROM"].astype('string').isin(autosomes)]    
    df["CHROM"] = df["CHROM"].astype('string')
    return df


def check_exp_ref(lambdas_ref: Union[pd.DataFrame, Sequence, np.ndarray]) -> pd.DataFrame:
    """
    Validate a reference-expression profile.

    Parameters
    ----------
    lambdas_ref : pandas.DataFrame | array-like
        Gene-by-reference matrix of (normalised) expression magnitudes.

    Returns
    -------
    pandas.DataFrame
        The same matrix, free of NA, free of duplicated genes,
        and confirmed to contain non-integer (normalised) values.

    Raises
    ------
    ValueError
        When any of the data-quality checks fail.
    """
    
    # check it is a 2-D DataFrame
    if not isinstance(lambdas_ref, pd.DataFrame):
        lambdas_ref = pd.DataFrame(lambdas_ref)
        lambdas_ref.columns = ['ref']

    # remove NA
    if lambdas_ref.isna().any(axis=None):
        msg = ("The reference expression matrix 'lambdas_ref' "
               "should not contain any NA values.")
        #log.error(msg)
        raise ValueError(msg)

    # Reject integer-only matrices (raw counts)
    arr = lambdas_ref.to_numpy(copy=False)
    if np.all(arr == arr.astype(int)):
        msg = ("The reference expression matrix 'lambdas_ref' appears to "
               "contain only integer values. Please normalise raw counts "
               "with aggregate_counts() before calling this routine.")
        #log.error(msg)
        raise ValueError(msg)

    # check that Gene IDs (row index) are unique
    if lambdas_ref.index.has_duplicates:
        msg = "Please remove duplicated genes in reference profile."
        #log.error(msg)
        raise ValueError(msg)


    return lambdas_ref.copy()


def fit_ref_sse_ad(
    count_mat: ad.AnnData,
    lambdas_ref: pd.DataFrame,
    gtf: pd.DataFrame,
    min_lambda: float = 2e-6,
    verbose: bool = False
    ) -> Dict[str, Any]:
    """
    Fit a reference expression profile to a count matrix using sum-of-squared-errors on log-scaled values.

    Parameters
    ----------
    count_mat : ad.AnnData
        AnnData object containing sample counts (cells x genes).
    lambdas_ref : pd.DataFrame
        Reference expression profiles; genes as index, profiles as columns.
    gtf : pd.DataFrame
        Genome annotation with a 'gene' column listing gene names.
    min_lambda : float, optional
        Minimum mean expression threshold for genes to include (default 2e-6).
    verbose : bool, optional
        If True, show optimization progress (default False).

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys:
        - 'w': np.ndarray of optimized weights per reference profile.
        - 'lambdas_bar': np.ndarray weighted combination of reference profiles.
        - 'mse': float mean squared error of the fit per gene.
    """
    count_mat = count_mat[:, np.array(count_mat.X.sum(0) > 0).flatten()]
    common_genes = set(
        gtf.loc[:,'gene']).intersection(
        set(count_mat.var_names)).intersection(
            set(lambdas_ref[lambdas_ref.mean(1) > min_lambda].index))
    common_genes = [g for g in gtf.loc[:, 'gene'] if g in common_genes]

    count_mat = count_mat[:, common_genes]
    lambdas_obs = np.exp(np.log(np.array(count_mat.X.sum(0)).flatten()) -
                         np.log(np.array(count_mat.X.sum(0)).flatten().sum()))
    lambdas_ref = lambdas_ref.loc[common_genes, :]

    n_ref = lambdas_ref.shape[1]

    def kl_to_min(x):
        return np.sum(np.power(np.log(lambdas_obs) -
                               np.log(np.matmul(lambdas_ref, x / np.sum(x))),
                               2))
    bounds = [(1e-6, None)] * n_ref
    par = np.ones(n_ref) / n_ref
    fit = scipy.optimize.minimize(
        fun=kl_to_min,
        x0=par,
        method='L-BFGS-B',
        tol=1e-6,
        bounds=bounds,
        #options={'disp': verbose}
    )

    x = fit.x
    x /= np.sum(x)
    lambdas_bar = np.matmul(lambdas_ref, x)
    lambdas_mse = fit.fun / len(lambdas_obs)

    return {'w': x, 'lambdas_bar': lambdas_bar, 'mse': lambdas_mse}


def filter_genes(
    count_mat: ad.AnnData,
    lambdas_bar: Union[Dict[str, float], pd.Series],
    gtf: Union[pd.DataFrame, dict],
    filter_segments: Optional[pd.DataFrame] = None,
    filter_hla: bool = True,
    verbose: bool = False
    ) -> List[str]:
    """
    Filter genes based on expression and annotation criteria.

    Parameters
    ----------
    count_mat : AnnData
        Single-cell count matrix with `.var_names` representing gene names.
    lambdas_bar : dict or pd.Series
        Reference expression profile keyed/indexed by gene names.
    gtf : pd.DataFrame or dict
        Genome annotation containing at least columns ['gene', 'CHROM', 'gene_start', 'gene_end'].
    filter_segments : pd.DataFrame, optional
        Optional segments dataframe used to exclude genes overlapping these regions.
        Must contain columns ['CHROM', 'seg_start', 'seg_end'].
    filter_hla : bool, optional
        Whether to exclude genes in the HLA region on chromosome 6 (default True).
        Exclude hg38 chr6:28510120-33480577.
        !!! SET TO False IF USED IN GENOMES OTHER THAN hg38 and hg19 !!!
    verbose : bool, optional
        If True, prints the number of retained genes (default False).

    Returns
    -------
    List[str]
        List of gene names retained after filtering.

    Notes
    -----
    - Genes are initially filtered to those present in all three inputs: gtf, count_mat, and lambdas_bar.
    - Optionally excludes genes overlapping the human HLA region on chromosome 6, hg38 coordinates.
    - Optionally excludes genes overlapping regions defined in `filter_segments`.
    - Retention is based on expression thresholds applied to both the reference profile (`lambdas_bar`) and observed counts.
    """
    gtf_df = pd.DataFrame(gtf)

    # Get genes to keep - intersection of gtf genes, count_mat genes, and lambdas_bar keys
    genes_keep = set(gtf_df['gene']).intersection(set(count_mat.var_names)).intersection(set(lambdas_bar.keys()))
    # Sort genes following gtf ordering
    genes_keep = [gene for gene in gtf_df['gene'] if gene in genes_keep]

    if filter_hla:
        # Exclude genes in HLA region (chr6: 28,510,120 - 33,480,577) in hg38 and hg19
        genes_exclude = gtf_df[(gtf_df['CHROM'].astype("string").isin(["6", "chr6"]) &
                               (gtf_df['gene_start'] < 33480577) &
                               (gtf_df['gene_end'] > 28510120))]['gene'].tolist()
        genes_keep = [gene for gene in genes_keep if gene not in genes_exclude]

    if filter_segments is not None and not filter_segments.empty:
        genes_exclude = []
        for _, row in filter_segments.iterrows():
            overlapping = gtf_df[(gtf_df['CHROM'].astype("string") == str(row.CHROM)) &
                                 (gtf_df['gene_start'] < row.seg_end) &
                                 (gtf_df['gene_end'] > row.seg_start)]['gene'].tolist()
            genes_exclude.extend(overlapping)
        genes_keep = [gene for gene in genes_keep if gene not in genes_exclude]

    # Filter count matrix and lambdas_bar
    count_mat_filtered = count_mat[:, genes_keep]
    lambdas_bar_filtered = lambdas_bar.loc[genes_keep] if isinstance(lambdas_bar, pd.Series) else {g: lambdas_bar[g] for g in genes_keep}
    if isinstance(lambdas_bar_filtered, dict):
        # Convert dict to pd.Series for consistent indexing below
        lambdas_bar_filtered = pd.Series(lambdas_bar_filtered)

    lambdas_obs = pd.Series(
        np.array(count_mat_filtered.X.sum(0) / count_mat_filtered.X.sum()).ravel(),
        index=count_mat_filtered.var_names
    )

    # Thresholds and means
    min_both = 2
    mean_lambdas_bar = lambdas_bar_filtered[lambdas_bar_filtered > 0].mean()
    mean_lambdas_obs = lambdas_obs[lambdas_obs > 0].values.mean(dtype=np.float64)

    # Identify genes passing expression filters
    mut_expressed = pd.DataFrame(
        (((lambdas_bar_filtered.values.flatten() * 1e6 > min_both) & (lambdas_obs.values * 1e6 > min_both)) |
         (lambdas_bar_filtered.values.flatten() > mean_lambdas_bar) |
         (lambdas_obs.values > mean_lambdas_obs)) & 
        (lambdas_bar_filtered.values.flatten() > 0))
    mut_expressed.index = lambdas_bar_filtered.index

    # Retain genes that meet the filter criteria
    retained = [gene for gene, expressed in zip(genes_keep, mut_expressed[0]) if expressed]

    if verbose:
        log.info(f'number of genes left: {len(retained)}')
    return retained


def get_exp_bulk(
    count_mat: ad.AnnData,
    lambdas_bar: pd.Series,
    gtf: pd.DataFrame,
    verbose: bool = False,
    filter_hla: bool = False,
    filter_segments: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
    """
    Compute bulk gene expression metrics by filtering genes and combining observed counts
    with reference expression profiles and genome annotations.

    Parameters
    ----------
    count_mat : ad.AnnData
        Single-cell count matrix with genes as `.var_names`.
    lambdas_bar : pd.Series
        Reference expression profile indexed by gene names.
    gtf : pd.DataFrame
        Genome annotation containing at least ['gene', 'CHROM'] columns.
    verbose : bool, optional
        Whether to print progress messages (default False).
    filter_hla : bool, optional
        Whether to exclude genes in the HLA region on chromosome 6 (default False).
    filter_segments : pd.DataFrame or None, optional
        Optional dataframe defining genomic segments to filter out overlapping genes (default None).

    Returns
    -------
    pd.DataFrame
        DataFrame with bulk gene expression statistics per gene including:
        - 'Y_obs': observed counts sum
        - 'd_obs': total library depth before filtering
        - 'lambda_obs': normalized observed expression (counts / depth)
        - 'lambda_ref': reference expression
        - gene annotation columns from `gtf`
        - 'logFC': log2 fold change between observed and reference expression
        - 'lnFC': natural log fold change between observed and reference expression

    Notes
    -----
    - Genes are filtered using `filter_genes` function based on expression and annotations.
    - Log fold changes with infinite values are filtered out.
    """
    depth_obs_before_filt = count_mat.X.sum()
    mut_expressed = filter_genes(count_mat, lambdas_bar, gtf, filter_hla=filter_hla, filter_segments=filter_segments)
    count_mat = count_mat[:, mut_expressed]
    lambdas_bar = lambdas_bar.loc[mut_expressed]

    bulk_obs = pd.DataFrame({'Y_obs': count_mat.X.sum(0).T.A.ravel()}, index=count_mat.var_names)
    bulk_obs = bulk_obs.rename_axis('gene').reset_index()

    # Use pre-filter library depth for normalization
    bulk_obs['d_obs'] = depth_obs_before_filt
    bulk_obs['lambda_obs'] = bulk_obs['Y_obs'] / bulk_obs['d_obs']
    bulk_obs['lambda_ref'] = lambdas_bar[bulk_obs['gene']].values.astype(np.float64)

    gtf = gtf.copy()
    #gtf['gene_index'] = gtf.index
    bulk_obs.gene = bulk_obs.gene.astype("string")
    gtf.gene = gtf.gene.astype("string")
    bulk_obs = bulk_obs.merge(gtf, on='gene', how='left', sort=False)

    bulk_obs['CHROM'] = bulk_obs['CHROM'].astype('string') #was category
    bulk_obs['gene'] = bulk_obs['gene'].astype('string') #was category
    bulk_obs['logFC'] = np.log2(bulk_obs['lambda_obs']) - np.log2(bulk_obs['lambda_ref'])
    bulk_obs['lnFC'] = np.log(bulk_obs['lambda_obs']) - np.log(bulk_obs['lambda_ref'])

    # Filter out infinite log fold changes
    bulk_obs = bulk_obs[~bulk_obs['logFC'].isin([np.inf, -np.inf]) & ~bulk_obs['lnFC'].isin([np.inf, -np.inf])]
    bulk_obs["gene_index"] = [i for i in range(bulk_obs.shape[0])]
    return bulk_obs


def get_inter_cm(cM: pd.Series) -> NDArray:
    """
    Compute inter-marker genetic distances from cumulative cM positions.

    Parameters
    ----------
    cM : pandas.Series
        A series of cumulative genetic positions in centiMorgans (cM),
        ordered along a chromosome.

    Returns
    -------
    numpy.ndarray
        Array of inter-marker distances with the same length as `cM`.
        The first element is NaN (no preceding marker).
        If input length <= 1, returns np.nan scalar.
    """
    if len(cM) <= 1:
        return np.nan
    else:
        return np.hstack([np.nan, cM.values[1:] - cM.values[:-1]])


def switch_prob(
    distance: NDArray,
    nu: float = 1,
    min_p: float = 1e-10
    ) -> NDArray:
    """
    Calculate switch probabilities based on genetic distances and parameter nu.

    Parameters
    ----------
    distance : numpy.ndarray
        Array of inter-marker distances (genetic distances in cM or recombination units).
    nu : float, optional
        Recombination rate parameter. If zero, returns zeros array. Default is 1.
    min_p : float, optional
        Minimum threshold for switch probabilities to avoid zeros. Default is 1e-10.

    Returns
    -------
    numpy.ndarray
        Array of switch probabilities, thresholded to minimum `min_p`.
        NaN values are replaced by 0.
    """
    if nu == 0:
        p = np.zeros(len(distance))
    else:
        #p = np.exp(np.log(1 - np.exp(-2 * nu * distance)) - np.log(2))
        p = (1 - np.exp(-2 * nu * distance)) * 0.5

        p = np.maximum(p, min_p)

    # p[np.isnan(p)] = 0 
    p[np.isnan(distance)] = 0 # Faithful
    return p


def get_allele_bulk(
    df_allele: pd.DataFrame,
    nu: float = 1,
    min_depth: int = 0
    ) -> pd.DataFrame:
    """
    Process allele count data to produce a bulk allele summary table with
    allele ratios, positional indexing, and switch probabilities.

    Parameters
    ----------
    df_allele : pd.DataFrame
        Allele-level DataFrame expected to contain columns:
        ['snp_id', 'CHROM', 'POS', 'cM', 'REF', 'ALT', 'AD', 'DP', 'GT', 'gene'].
        'GT' genotype values must be in {'1|0', '0|1'} for inclusion.
    nu : float, optional
        Parameter for recombination rate used in switch probability calculation,
        by default 1.
    min_depth : int, optional
        Minimum depth (DP) threshold for including SNPs, by default 0.

    Returns
    -------
    pd.DataFrame
        Processed allele bulk DataFrame including:
        - Aggregated allele depths ('AD') and total depths ('DP')
        - Allelic ratio ('AR')
        - SNP positional index within each chromosome ('snp_index')
        - Probabilistic B-allele frequency ('pBAF') and adjusted depth ('pAD')
        - Inter-SNP genetic distance ('inter_snp_cm')
        - Switch probability ('p_s')
        - 'gene' column cast to string or NaN if missing

    Notes
    -----
    - SNPs with genotypes other than '1|0' or '0|1' are excluded.
    - SNPs with NaN genetic map positions ('cM') are excluded.
    - The function computes inter-SNP distances per chromosome.
    - Switch probabilities are computed using `switch_prob` function (nu parameter).
    """
    df_allele = df_allele.loc[:, ['snp_id', 'CHROM', 'POS', 'cM', 'REF', 'ALT', 'AD', 'DP', 'GT', 'gene']]
    df_allele = df_allele[df_allele.GT.isin({'1|0', '0|1'})]
    df_allele = df_allele[~np.isnan(df_allele.cM)]
    
    # Sum AD and DP grouped by SNP attributes
    df_allele = df_allele.groupby(['snp_id', 'CHROM', 'POS', 'cM', 'REF', 'ALT', 'GT', 'gene'],
                                    sort=False, as_index=False, dropna=False).sum(['AD', 'DP'])
    
    df_allele['AR'] = df_allele.AD / df_allele.DP
    df_allele = df_allele.sort_values(['CHROM', 'POS'], key=natsort.natsort_keygen())

    # Assign SNP index per chromosome
    flat_list = []
    for chrom in df_allele.CHROM.unique():
        snps = df_allele[df_allele.CHROM == chrom].snp_id
        flat_list.extend(range(len(snps)))
    df_allele['snp_index'] = flat_list

    # Filter by minimum depth
    df_allele = df_allele[df_allele.DP >= min_depth]

    # Compute probabilistic B-allele frequency and adjusted depth
    pBAF = []
    pAD = []
    for _, data in df_allele.iterrows():
        if data.GT == '1|0':
            pBAF.append(data.AR)
            pAD.append(data.AD)
        else:
            pBAF.append(1 - data.AR)
            pAD.append(data.DP - data.AD)
    df_allele['pBAF'] = pBAF
    df_allele['pAD'] = pAD

    df_allele = df_allele.sort_values(['CHROM', 'POS'], key=natsort.natsort_keygen()) # REMOVED NOW
    df_allele['CHROM'] = df_allele['CHROM'].astype('string')

    # Compute inter-SNP genetic distances chromosome-wise
    inter_snp_cm = np.zeros(df_allele.shape[0])
    start_idx = 0
    for chrom in df_allele.CHROM.unique():
        df_chrom = df_allele.loc[df_allele.CHROM == chrom]
        end_idx = start_idx + df_chrom.shape[0]
        inter_snp_cm[start_idx:end_idx] = get_inter_cm(df_chrom.cM)
        start_idx = end_idx
    df_allele['inter_snp_cm'] = inter_snp_cm
    
    # Compute switch probabilities
    df_allele['p_s'] = switch_prob(df_allele['inter_snp_cm'].values, nu=nu)

    # Ensure 'gene' column has string or NaN values
    df_allele['gene'] = df_allele['gene'].apply(lambda i: i if isinstance(i, str) else np.nan)
    return df_allele


def combine_bulk(
    allele_bulk: pd.DataFrame,
    exp_bulk: pd.DataFrame,
    filter_hla: bool = True,
    filter_segments: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
    """
    Combine allele-level bulk data and expression-level bulk data into a unified DataFrame,
    with optional filtering for HLA region and specified genomic segments.

    Parameters
    ----------
    allele_bulk : pd.DataFrame
        DataFrame containing allele bulk data, expected to include columns
        like ['snp_id', 'CHROM', 'POS', 'gene', 'p_s', 'Y_obs'].
    exp_bulk : pd.DataFrame
        DataFrame containing expression bulk data with columns including
        ['gene', 'CHROM', 'gene_start', 'lambda_obs', 'lambda_ref', 'd_obs'].
    filter_hla : bool, optional
        Whether to exclude SNPs/genes located in the HLA region on chromosome 6
        (default True).
    filter_segments : pd.DataFrame or None, optional
        DataFrame of genomic segments with columns ['CHROM', 'seg_start', 'seg_end']
        used to exclude overlapping SNPs/genes (default None).

    Returns
    -------
    pd.DataFrame
        Combined and filtered DataFrame with updated allele counts,
        normalized expression values, log fold changes, and SNP indices.

    Notes
    -----
    - SNP IDs missing in `allele_bulk` are filled with corresponding gene names.
    - Positions missing in `allele_bulk` are filled with gene start positions from `exp_bulk`.
    - Switch probabilities missing in allele data are set to zero.
    - Filters out HLA region on chromosome 6 and any segments specified in `filter_segments`.
    - SNP indices are reassigned per chromosome in genomic order.
    - Fold changes (`logFC` and `lnFC`) are computed with infinity values replaced by NaN.
    """
    # Outer merge on CHROM and gene
    allele_bulk.CHROM = allele_bulk.CHROM.astype("string")
    allele_bulk.gene = allele_bulk.gene.astype("string")
    exp_bulk.CHROM = exp_bulk.CHROM.astype("string")
    exp_bulk.gene = exp_bulk.gene.astype("string")
    bulk = pd.merge(allele_bulk, exp_bulk, how='outer', on=['CHROM', 'gene'])
    # Fill missing SNP ids with gene names
    bulk['snp_id'] = np.where(bulk['snp_id'].isna(), bulk['gene'], bulk['snp_id'])
    # Fill missing POS with gene_start from expression data
    bulk['POS'] = np.where(bulk['POS'].isna(), bulk['gene_start'], bulk['POS'])
    # Fill missing switch probabilities with zero
    bulk['p_s'] = np.where(bulk['p_s'].isna(), 0, bulk['p_s'])
    
    # Sort by chromosome and position using natural sorting
    bulk = bulk.sort_values(by=['CHROM', 'POS'], key=natsort.natsort_keygen())
    
    # Exclude genes in HLA region (chr6: 28,510,120 - 33,480,577) in hg38 and hg19
    if filter_hla:
        to_filter = bulk[(bulk['CHROM'].astype("string").isin(["6", "chr6"]) & 
                    (bulk['POS'] > 28510120) & 
                    (bulk['POS'] < 33480577))].index
        bulk = bulk.drop(index=to_filter)
    
    # Filter segments overlap if provided
    if filter_segments is not None and not filter_segments.empty:
        genes_exclude = []
        for _, row in filter_segments.iterrows():
            to_filter = bulk[(bulk['CHROM'].astype("string") == str(row.CHROM)) &
                             (bulk['POS'] < row.seg_end) &
                             (bulk['POS'] > row.seg_start)].index.tolist()
            genes_exclude.extend(to_filter)
        bulk = bulk.drop(index=genes_exclude)
    
    # get rid of duplicate gene expression values, collapsing multiple SNPs per gene
    dup_mask = bulk["gene"].notna() & bulk.duplicated(subset=["CHROM", "gene"], keep="first")
    bulk.loc[dup_mask, "Y_obs"] = np.nan
    
    # Calculate fold changes and normalize lambda_obs
    bulk['lambda_obs'] = bulk['Y_obs'] / bulk['d_obs']
    fc = np.exp(np.log(bulk['lambda_obs']) - np.log(bulk['lambda_ref']))
    bulk['logFC'] = np.log2(fc)
    bulk['logFC'] = bulk['logFC'].replace([np.inf, -np.inf], np.nan)
    bulk['lnFC'] = np.log(fc)
    bulk['lnFC'] = bulk['lnFC'].replace([np.inf, -np.inf], np.nan)
    
    # Resort by chromosome and position with natural sorting
    bulk = bulk.sort_values(by=['CHROM', 'POS'], key=natsort.natsort_keygen()).reset_index(drop=True)
    
    # Assign SNP index per chromosome
    snp_index = []
    for chrom in bulk['CHROM'].unique():
        current_snp_num = bulk[bulk['CHROM'] == chrom].shape[0]
        snp_index.extend(range(current_snp_num))
    bulk['snp_index'] = snp_index
    
    return bulk


def annot_consensus(bulk, segs_consensus, join_mode='inner'):
    
    """
    Annotate consensus segments on a pseudobulk dataframe.
    
    Args:
        bulk (pd.DataFrame): Pseudobulk profile.
        segs_consensus (pd.DataFrame): Consensus segment dataframe.
        join_mode (str): 'inner' or 'left' join mode.
        
    Returns:
        pd.DataFrame: Annotated pseudobulk profile.
     """
    
    # Set the join mode
    if join_mode == 'inner':
        how = 'inner'
    else:
        how = 'left'
    
    # If 'seg_cons' not in segs_consensus columns, create it
    if 'seg_cons' not in segs_consensus.columns:
        segs_consensus = segs_consensus.copy()
        segs_consensus['seg_cons'] = segs_consensus['seg']
    
    # Copy bulk
    bulk = bulk.copy()
    
    # Alternative pyranges usage
    # bulk_ranges
    bulk.loc[:,'End'] = bulk.POS
    bulk = bulk.rename(columns={'CHROM':'Chromosome', 'POS':'Start'})
    #bulk["Start"] = bulk["Start"]
    
    bulk_ranges = pr.PyRanges(df=bulk) 
    
    # segs_consensus_ranges
    segs_consensus = segs_consensus.rename(columns={'CHROM':'Chromosome', 'seg_start':'Start', 'seg_end':'End'})
    segs_consensus_ranges = pr.PyRanges(df=segs_consensus) 
    
    # Find overlaps between bulk and segs_consensus
    overlaps = segs_consensus_ranges.join(bulk_ranges, how='left', slack=1) # original implementation from me
    #overlaps = bulk_ranges.join(segs_consensus_ranges, how='left', slack=0) # TODO: just added
    #overlaps = bulk_ranges.join(segs_consensus_ranges, how='left', slack=1) # TODO: just added # last
    overlaps_df = overlaps.df
    
    # # renaming
    bulk = bulk.rename(columns={'Chromosome':'CHROM', 'Start':'POS'})
    bulk = bulk.drop(columns='End')
    segs_consensus = segs_consensus.rename(columns={'Chromosome':'CHROM', 'Start':'seg_start', 'End':'seg_end'})
    if ('seg_start' in overlaps_df.columns) and ('seg_end' in overlaps_df.columns):
        overlaps_df = overlaps_df.drop(columns=['seg_start', 'seg_end'])
    if ('Start_b' in overlaps_df.columns) and ('End_b' in overlaps_df.columns):
        overlaps_df = overlaps_df.drop(columns=['Start_b', 'End_b'])
    overlaps_df = overlaps_df.rename(columns={'Chromosome':'CHROM','Start':'seg_start', 'End':'seg_end'})
    # # Remove duplicates of snp_id, keeping the first occurrence
    overlaps_df = overlaps_df.drop_duplicates(subset='snp_id')
    overlaps_df["CHROM"] = overlaps_df["CHROM"].astype("string")

    
    # Drop unnecessary columns
    columns_to_exclude = ['sample']
    overlaps_df = overlaps_df.drop(columns=[col for col in columns_to_exclude if col in overlaps_df.columns])
    overlaps_df = overlaps_df.loc[:,['snp_id'] + [col for col in segs_consensus if col not in columns_to_exclude]]
    # Exclude overlapping columns from bulk except 'snp_id' and 'CHROM'
    exclude_from_bulk = [col for col in overlaps_df.columns if col not in ['snp_id', 'CHROM']]
    bulk = bulk.drop(columns=[col for col in exclude_from_bulk if col in bulk.columns])
    
    # # Merge bulk and overlaps_df
    bulk.snp_id = bulk.snp_id.astype("string")
    bulk.CHROM = bulk.CHROM.astype("string")
    overlaps_df.snp_id = overlaps_df.snp_id.astype("string")
    overlaps_df.CHROM = overlaps_df.CHROM.astype("string")
    bulk = bulk.merge(overlaps_df, on=['snp_id', 'CHROM'], how=how)    
    # # Assign 'seg' from 'seg_cons'
    bulk.loc[:,'seg'] = bulk.loc[:,'seg_cons']
    # TODO: add type string for "seg"
    return bulk


def sanityze_df(df):

    group_collector = []
    for idx, group in df.groupby("gene", sort=False):

        if group.shape[0] > 1:
            group.loc[group.index[1]:,"logFC"] = np.nan
            group.loc[group.index[1]:,"lnFC"] = np.nan
    
        group_collector.append(group)
        
    nan_remove = pd.concat(group_collector, axis=0)
    pd.concat([nan_remove,df[df.gene.isna()]],axis=0)

    return df


def get_bulk(
    count_mat: ad.AnnData,
    lambdas_ref: Union[pd.DataFrame, pd.Series],
    df_allele: pd.DataFrame,
    gtf: pd.DataFrame,
    subset: Optional[Sequence[str]] = None,
    min_depth: int = 0,
    nu: float = 1,
    segs_loh: Optional[pd.DataFrame] = None,
    verbose: bool = True,
    disp: bool = False,
    filter_hla: bool = True,
    filter_segments: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
    """
    Compute combined bulk allele and expression data with filtering and clonal LOH annotation.

    Parameters
    ----------
    count_mat : anndata.AnnData
        Single-cell count matrix.
    lambdas_ref : pd.DataFrame or pd.Series
        Reference expression profile.
    df_allele : pd.DataFrame
        Allele-level data frame with SNP genotype info.
    gtf : pd.DataFrame
        Genome annotation with gene info.
    subset : Optional[Sequence[str]], optional
        Optional list of cell barcodes to subset `count_mat` and `df_allele`.
        Default is None (no subsetting).
    min_depth : int, optional
        Minimum depth threshold for filtering alleles (default 0).
    nu : float, optional
        Parameter for switch probability calculation (default 1).
    segs_loh : Optional[pd.DataFrame], optional
        Clonal LOH segments for annotation (default None).
    verbose : bool, optional
        Verbosity flag for internal functions (default True).
    disp : bool, optional
        Display flag for optimization steps (default False).
    filter_hla : bool, optional
        Whether to filter HLA region genes (default True).
    filter_segments : Optional[pd.DataFrame], optional
        Additional segments to filter genes/SNPs overlapping these (default None).

    Returns
    -------
    pd.DataFrame
        Combined and annotated bulk data with gene, allele, expression, and LOH information.

    Raises
    ------
    KeyError
        If requested cell barcodes in `subset` are not all present in `count_mat.obs_names`.
    ValueError
        If duplicated SNP IDs are found after merging allele and expression data.
    """

    # ***EXPLICIT*** COPY OF THE ANNDATA BEFORE YOU WRITE ON IT!
    count_mat = check_anndata(count_mat.copy())
    if subset is not None: 
        if not set(subset).issubset(set(count_mat.obs_names)):
            raise KeyError('All the requested cell barcodes must be present in count_mat')
        else:
            count_mat = count_mat[subset]
            df_allele_subset_mask = [i in subset for i in df_allele.cell]
            df_allele = df_allele[df_allele_subset_mask]
    fit = fit_ref_sse_ad(count_mat, lambdas_ref, gtf, verbose=disp)
    exp_bulk = get_exp_bulk(count_mat, fit['lambdas_bar'], gtf, verbose=verbose, filter_hla=filter_hla, filter_segments=filter_segments)
    exp_bulk = exp_bulk[((exp_bulk.loc[:,'logFC'] > -5) & (exp_bulk.loc[:,'logFC'] < 5)) | (exp_bulk.loc[:,'Y_obs'] == 0)]
    exp_bulk.loc[:,'mse'] = fit['mse']
    allele_bulk = get_allele_bulk(df_allele, nu=nu, min_depth=min_depth)
    bulk = combine_bulk(allele_bulk, exp_bulk, filter_hla=filter_hla, filter_segments=filter_segments)
    if np.unique(bulk.loc[:,'snp_id']).shape[0] != bulk.loc[:,'snp_id'].shape[0]:
        raise ValueError('Duplicated SNPs found, please check genotypes')
    
    # Filter out rows where lambda_ref is zero or gene is not NaN
    bulk = bulk[(bulk.loc[:, 'lambda_ref'] != 0) | (bulk.loc[:,'gene'].isna())]

    #bulk = sanityze_df(bulk) # TODO: remove it!!!!!!!!!!!!!!!!!!!!!!!!
    
    #bulk.loc[:,'CHROM'] = np.where(bulk.loc[:, 'CHROM'] == 'X', "23", bulk.loc[:,'CHROM'])
    bulk = bulk.sort_values(by=['CHROM','POS'], key=natsort.natsort_keygen())
    bulk = bulk.reset_index(drop=True)

    # Annotate clonal LOH regions
    if segs_loh is None:
        bulk.loc[:,'loh'] = False
    else:
        # Annotate consensus segments
        bulk = annot_consensus(bulk, segs_loh, join_mode='left')
        # Set 'loh' to False where it's NaN
        bulk.loc[:,'loh'] = bulk.loc[:,'loh'].fillna(0).astype(bool)
    
    return bulk


## Fit snp rate on loh calling

def fit_snp_rate(
    gene_snps: ArrayLike,
    gene_length: float
    ) -> np.ndarray:
    """
    Fit SNP mutation rate parameters to observed SNP counts per gene using Negative Binomial likelihood.

    Parameters
    ----------
    gene_snps : array-like
        Observed SNP counts for a single gene across samples or regions (non-negative integers).
    gene_length : float
        Length of the gene in base pairs.

    Returns
    -------
    np.ndarray
        Estimated parameters array `[v, sig]`:
        - v: scaled mutation rate parameter (unit depends on gene length scaling)
        - sig: dispersion parameter of the Negative Binomial distribution

    Notes
    -----
    - The mean `mu` of the Negative Binomial is modeled as `v * gene_length / 1e6`.
    - Uses L-BFGS-B optimizer with positivity constraints on parameters.
    - Returns the optimized parameters minimizing the negative log-likelihood.
    """
    # Define the objective function to minimize
    def objective(params):
        v = params[0]
        sig = params[1]
        mu = v * gene_length / 1e6
        log_likelihood = np.sum(scipy.stats.nbinom.logpmf(gene_snps, sig, sig / (mu + sig)))
        return -log_likelihood

    # Initial parameters
    initial_params = [10, 1]
    # Constraints on the parameters (lower bounds)
    bounds = [(1e-10, None), (1e-10, None)]

    # Minimize the objective function
    result = scipy.optimize.minimize(objective, initial_params, method='L-BFGS-B', bounds=bounds)

    return result.x


## Annotate loh call
def generate_postfix(n:List):
    '''
    Generate alphabetical postfixes for a list of positive integers.

    Parameters
    ----------
    n : List
        A list of positive integers.

    Raises
    ------
    ValueError
        Raise ValueError if any of the integers are None.

    Returns
    -------
    postfixes : List[str]
        Alphabetical postfixes corresponding to the integers.

    '''

    if any(x is None for x in n):
        raise ValueError("Segment number cannot contain NA")
    
    alphabet = list(string.ascii_lowercase)
    len_alphabet = len(alphabet)
    postfixes = []
    for number in n:
        i = int(number)  # Ensure the number is an integer
        postfix = ''
        while True:
            i, remainder = divmod(i, len_alphabet)
            postfix = alphabet[remainder] + postfix
            if i == 0:
                break
        postfixes.append(postfix)
    return postfixes


def annot_segs(bulk: pd.DataFrame, var: str = "cnv_state") -> pd.DataFrame:
    """
    Annotate contiguous segments along each chromosome based on a state column.

    This function scans rows within each chromosome in their existing order and
    starts a new segment whenever the value in `var` changes from the previous row.
    It then assigns a segment identifier per row and computes per-segment attributes.

    Parameters
    ----------
    bulk : pandas.DataFrame
        Long-form table containing at least the following columns:
          - "CHROM" : chromosome identifier (will be cast to pandas "string" dtype)
          - "POS" : genomic coordinate (integer-like)
          - "snp_index" : monotone index along the chromosome (integer-like)
          - "gene" : gene symbol (string-like or NA)
          - "pAD" : allele depth for the “alternate” allele (numeric, may be NA)
          - "{var}" : state used to split segments (e.g., "cnv_state")
        Rows are assumed to be already ordered by genomic position within each
        chromosome.

    var : str, default 'cnv_state'
        Name of the column in "bulk" that defines segment boundaries; a new
        segment starts whenever this value changes across adjacent rows within
        a chromosome.

    Returns
    -------
    pandas.DataFrame
        A copy of "bulk" with the following additional columns:
          - "boundary" : 0/1 indicator; 1 where a new segment starts
          - "seg" : segment identifier (string) per row
          - "seg_start" / "seg_end" : min/max "POS" within the segment (int64)
          - "seg_start_index" / "seg_end_index" : min/max "snp_index" in the segment
          - "n_genes" : number of unique, non-null genes in the segment
          - "n_snps" : number of rows in the segment with non-null "pAD"

    Notes
    -----
    - The algorithm does NOT sort rows; it treats the current row order within each
      chromosome as the traversal order for segmenting. An input sorted by "['CHROM', 'POS']" is expected.
    - The return value is a new DataFrame (input is not modified in place).

    Examples
    --------
    >>> df = df.sort_values(["CHROM", "POS"])
    >>> out = annot_segs(df, var="cnv_state")
    >>> out[["CHROM", "POS", "cnv_state", "seg"]].head()
    """

    # you need to reset index so you can pass portion of list (groups)
    bulk = bulk.copy().reset_index(drop=True)
    bulk.CHROM = bulk.CHROM.astype('string')
    boundary = []
    postfix = []
    cum_sum_test = 0
    for chrom in bulk.CHROM.unique():
        temp_sorted = bulk[bulk.loc[:, 'CHROM'] == chrom]
        cum_sum_test += temp_sorted.shape[0]
        boundary += [0]+[1 if temp_sorted.loc[:,var].iloc[i] != temp_sorted.loc[:,var].iloc[i - 1] else 0 for i in range(1,temp_sorted.shape[0])]
        current_postfix = generate_postfix(np.cumsum(boundary[temp_sorted.index[0]:temp_sorted.index[-1]+1]))
        postfix += [str(chrom)+i for i in current_postfix]
    
    # Natural sorting and cast to Categorical to avoid warnings
    postfix = pd.Series(postfix)
    bulk.loc[:,'boundary'] = boundary
    bulk.loc[:,'seg'] = postfix
    
    seg_start = []
    seg_end = []
    seg_start_index = []
    seg_end_index = []
    n_genes = []
    n_snps = []
    
    for seg in bulk.seg.unique():
        current_seg = bulk[bulk.loc[:, 'seg'] == seg]
        seg_len = current_seg.shape[0]
        seg_start += list(np.repeat(current_seg.POS.min(), seg_len))
        seg_end += list(np.repeat(current_seg.POS.max(), seg_len))
        seg_start_index += list(np.repeat(current_seg.snp_index.min(), seg_len))
        seg_end_index += list(np.repeat(current_seg.snp_index.max(), seg_len))
        n_genes += list(np.repeat(current_seg.gene[~current_seg.gene.isnull()].unique().shape[0], seg_len))
        n_snps += list(np.repeat(np.count_nonzero(~current_seg.pAD.isna()), seg_len))
    
    bulk.loc[:, 'seg_start'] = np.array(seg_start, dtype=np.int64)
    bulk.loc[:, 'seg_end'] = np.array(seg_end, dtype=np.int64)
    bulk.loc[:, 'seg_start_index'] = seg_start_index
    bulk.loc[:, 'seg_end_index'] = seg_end_index
    bulk.loc[:, 'n_genes'] = n_genes
    bulk.loc[:, 'n_snps'] = n_snps

    return bulk

def t_test_pval(x: ArrayLike, y: ArrayLike) -> float:
    """
    Two-sample t-test p-value with small-sample guard.

    If either sample has ≤ 1 observation, returns 1.0. Otherwise returns the
    p-value from a two-sided Welch's t-test **assuming equal variances**
    and NaNs omitted.

    Parameters
    ----------
    x, y : ArrayLike
        1-D arrays (or array-like) of numeric observations.

    Returns
    -------
    float
        p-value in [0, 1]. Returns 1.0 if `x.size <= 1` or `y.size <= 1`.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Check length conditions
    if x.size <= 1 or y.size <= 1:
        return 1.0
    # Perform two-sample t-test (assuming equal var)
    #_, pvalue = ttest_ind(x, y, equal_var=True, nan_policy='omit')
    _, pvalue = ttest_ind(x, y, equal_var=False, nan_policy='omit')

    return pvalue


def simes_p(p_vals: ArrayLike, n_dim: int) -> float:
    """
    Compute Simes' combined p-value.

    Parameters
    ----------
    p_vals : ArrayLike
        Iterable of individual p-values (expected in [0, 1]).
    n_dim : int
        Scaling factor.

    Returns
    -------
    float
        Simes-adjusted p-value.

    Notes
    -----
    Classic Simes uses 1-based ranks k=1..m (i.e., ``min(m * p_(k) / k)``).
    This implementation currently constructs ``indices = np.arange(len(sorted_p))``,
    which starts at 0 and can cause division by zero. Keep this in mind if you
    rely on this function; adjust the indexing if you revise the code.
    """
    p_vals = np.asarray(p_vals)
    sorted_p = np.sort(p_vals)
    indices = np.arange(1, len(sorted_p)+1) # start at 1
    return n_dim * np.min(sorted_p / indices)

def Modes(x: Iterable[Any]) -> List[Any]:
    """
    Return all modes (most frequent values) of a 1-D iterable.

    Parameters
    ----------
    x : Iterable[Any]
        1-D data (hashable elements).

    Returns
    -------
    list
        List of elements achieving the maximum observed frequency. If multiple
        values tie for the highest count, all are returned (order arbitrary).
    """
    x = np.asarray(x)
    # Count occurrences using Counter
    c = Counter(x)
    # Find max frequency
    max_freq = max(c.values())
    # Return all elements that have this frequency
    return [k for k, v in c.items() if v == max_freq]


def detect_clonal_loh(
    bulk: pd.DataFrame,
    t: float = 1e-5,
    snp_rate_loh: float = 5,
    min_depth: int = 0,
    use_pbar: bool = False
    ) -> Optional[pd.DataFrame]:
    """
    Detect clonal Loss of Heterozygosity (LOH) segments from bulk-level allelic data.

    This function summarizes SNP counts and allelic metrics per gene, fits statistical
    models for allelic expression and SNP rates, and applies a Hidden Markov Model (HMM)
    to call clonal LOH segments. Outputs a DataFrame of detected LOH segments with
    estimated SNP rates.

    Parameters
    ----------
    bulk : pd.DataFrame
        DataFrame containing bulk-level allele and gene information, with required columns:
        ['CHROM', 'gene', 'gene_start', 'gene_end', 'AD', 'DP', 'Y_obs', 'lambda_ref',
         'logFC', 'd_obs'].
    t : float, optional
        HMM transition probability parameter, by default 1e-5.
    snp_rate_loh : float, optional
        Reference SNP mutation rate for LOH state in the HMM, by default 5.
    min_depth : int, optional
        Minimum depth (DP) for SNP inclusion, by default 0.
    use_pbar : bool, optional
        If True, show tqdm progress while iterating chromosomes during clonal
        LOH detection. Default is False.

    Returns
    -------
    Optional[pd.DataFrame]
        DataFrame of LOH segments with columns:
        ['CHROM', 'seg', 'seg_start', 'seg_end', 'snp_rate', 'loh'],
        or None if no LOH segments are detected.

    Notes
    -----
    - Genes are summarized per chromosome, with per-gene and per-segment SNP statistics.
    - Statistical fitting is performed using Negative Binomial and Poisson lognormal models.
    - HMM is used to segment genes into neutral and LOH states.
    - Only segments classified as 'loh' are retained in the output.
    - If no LOH segments are found, returns None.
    """
    
    bulk = bulk[(~bulk.loc[:, 'lambda_ref'].isna()) & (~bulk.loc[:,'gene'].isna())].copy()
    
    bulk_snps = {'CHROM':[],
             'gene':[],
             'gene_start':[],
             'gene_end':[],
             'gene_snps':[],
             'Y_obs':[],
             'lambda_ref':[],
             'logFC':[],
             'd_obs':[],
             'gene_length':[]}
    
    chrom_unique = bulk.CHROM.unique()
    for chrom in tqdm.tqdm(chrom_unique, disable=not use_pbar):
        gene_unique = bulk[bulk.loc[:, 'CHROM'] == chrom].gene.unique()
        for gene in gene_unique:
            tmp_bulk = bulk[(bulk.loc[:,'CHROM'] == chrom) &
                            (bulk.loc[:,'gene'] == gene)]
            gene_snps = tmp_bulk[(~tmp_bulk.loc[:,'AD'].isna()) &
                                (tmp_bulk.loc[:,'DP'] > min_depth)].shape[0]
            Y_obs = tmp_bulk.loc[:,'Y_obs'].dropna().unique().astype(np.int64).sum()
            lambda_ref = tmp_bulk.loc[:,'lambda_ref'].dropna().unique().item()
            logFC = tmp_bulk.loc[:,'logFC'].dropna().unique().item()
            d_obs = tmp_bulk.loc[:,'d_obs'].dropna().unique().item()
            bulk_snps['gene_snps'].append(gene_snps)
            bulk_snps['Y_obs'].append(Y_obs)
            bulk_snps['lambda_ref'].append(lambda_ref)
            bulk_snps['logFC'].append(logFC)
            bulk_snps['d_obs'].append(np.int64(d_obs))
            bulk_snps['gene'].append(tmp_bulk.gene.values[0])
            bulk_snps['gene_start'].append(tmp_bulk.gene_start.astype(np.int64).values[0])
            bulk_snps['gene_end'].append(tmp_bulk.gene_end.astype(np.int64).values[0])
            bulk_snps['gene_length'].append(np.array(tmp_bulk.gene_end.values[0] - tmp_bulk.gene_start.values[0]).astype(np.int64))
            bulk_snps['CHROM'].append(chrom)
    
    bulk_snps_df = pd.DataFrame(bulk_snps)
    bulk_snps_df = bulk_snps_df[(bulk_snps_df.logFC < 8) & (bulk_snps_df.logFC > -8)]
    bulk_snps_df = bulk_snps_df.sort_values(['CHROM','gene_start'],
                                            key=natsort.natsort_keygen()).reset_index(drop=True)
    
    fit = dist_prob.fit_lnpois(bulk_snps_df.Y_obs.values,
                     bulk_snps_df.lambda_ref.values,
                     bulk_snps_df.d_obs.unique())
    
    mu, sig = fit
    bulk_snps_df.gene_length = bulk_snps_df.gene_length.values.astype(np.int64)
    snp_fit = fit_snp_rate(bulk_snps_df.gene_snps.values, bulk_snps_df.gene_length.values)
    snp_rate_ref, snp_sig = snp_fit
    
    n = bulk_snps_df.shape[0]
    A = np.array([[1-t, t],[t, 1-t]])
    As = np.tile(A, (n, 1, 1)).transpose(1, 2, 0)
    
    HMM = {
    'x': bulk_snps_df.gene_snps,
    'Pi': As,
    'delta': [1-t, t],
    'pm': np.array([snp_rate_ref, snp_rate_loh]),
    'pn': np.array(bulk_snps_df.gene_length / 1e6),
    'snp_sig': snp_sig,
    'y': bulk_snps_df.Y_obs,
    'phi': [1, 0.5],
    'lambda_star': bulk_snps_df.lambda_ref,
    'd': bulk_snps_df.d_obs.unique(),
    'mu': mu,
    'sig':sig,
    'states': ['neu', 'loh']
    }
    
    vtb = hmmlib.viterbi_loh(HMM)
    bulk_snps_df.loc[:,'cnv_state'] = vtb
    
    bulk_snps_df.loc[:,'snp_index'] = bulk_snps_df.index
    bulk_snps_df.loc[:,'POS'] = bulk_snps_df.gene_start
    bulk_snps_df.loc[:,'pAD'] = 1

    segs_loh = annot_segs(bulk_snps_df)  # you may want this output for plot in bulk

    snp_rate = []
    for chrom in segs_loh.CHROM.unique():
        chrom_bulk = segs_loh[segs_loh.CHROM == chrom]
        for seg in chrom_bulk.seg.unique():
            seg_bulk = chrom_bulk[chrom_bulk.seg == seg]
            snp_rate.append(fit_snp_rate(seg_bulk.gene_snps, seg_bulk.gene_length)[0])
    
    segs_loh = segs_loh.groupby(['CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state'], 
                                observed=True, sort=False).sum()
    segs_loh = segs_loh.reset_index().loc[:,['CHROM', 'seg', 'seg_start', 'seg_end', 
                                             'cnv_state', 'gene_snps', 'gene_length']]
    
    segs_loh.loc[:,'snp_rate'] = snp_rate
    segs_loh = segs_loh[segs_loh.cnv_state == 'loh'].copy()
    
    if segs_loh.shape[0] == 0:
       segs_loh = None

    else:
        segs_loh.loc[:, 'loh'] = True
        segs_loh = segs_loh.reset_index().loc[:,['CHROM', 'seg', 'seg_start', 'seg_end', 'snp_rate', 'loh']]
    
    return segs_loh


def get_segs_neu(bulks: pd.DataFrame) -> pd.DataFrame:
    """
    Derive neutral genomic segments across pseudobulks and reduce overlaps.

    Filters rows with ``cnv_state == 'neu'``, computes per-(sample, seg, CHROM)
    start/end by min/max POS, then reduces overlapping/adjacent intervals using
    PyRanges.

    Parameters
    ----------
    bulks : pd.DataFrame
        Pseudobulk table with at least the columns:
        'sample', 'seg', 'CHROM', 'POS', and 'cnv_state'.

    Returns
    -------
    pd.DataFrame
        Reduced neutral segments with columns:
        - 'CHROM' : chromosome (string),
        - 'seg_start' : start coordinate (int),
        - 'seg_end' : end coordinate (int),
        - 'seg_length' : length in bases (int).

    Notes
    -----
    - Reduction is performed via ``PyRanges.merge()``.
    - Coordinates are taken directly from 'POS' min/max per neutral segment.
    """
    neu = bulks[bulks['cnv_state'] == 'neu'].copy()

    neu = neu.groupby(['sample','seg','CHROM'], sort=False, as_index=False, observed=True)
    segs_neu = neu.min('POS').dropna().loc[:, ['sample', 'seg', 'CHROM', 'POS']]
    segs_neu = segs_neu.rename({'POS': 'seg_start'}, axis=1)
    segs_neu['seg_end'] = neu.max('POS').dropna().loc[:,'POS']

    # Use PyRanges to reduce intervals (PyRanges uses 0-based, end-exclusive coordinates).
    gr = pr.PyRanges(chromosomes=segs_neu['CHROM'].astype("string"),
                     starts=segs_neu['seg_start'],
                     ends=segs_neu['seg_end'] + 1)
    gr_reduced = gr.merge()  # equivalent to reduce in GenomicRanges
    segs_neu_reduced = gr_reduced.as_df()

    segs_neu_reduced = segs_neu_reduced.rename(columns={'Chromosome':'CHROM','Start':'seg_start','End':'seg_end'})
    segs_neu_reduced['seg_end'] = segs_neu_reduced['seg_end'] - 1
    segs_neu_reduced['seg_length'] = segs_neu_reduced['seg_end'] - segs_neu_reduced['seg_start'] + 1

    return segs_neu_reduced


def fill_neu_segs(segs_consensus: pd.DataFrame, segs_neu: pd.DataFrame) -> pd.DataFrame:
    """
    Insert neutral intervals into a consensus set of segments and re-label.

    Computes gaps as "neutral - consensus" via PyRanges subtraction, appends
    these gaps to the consensus, sets missing "cnv_state" to "neu", and assigns
    a per-chromosome consensus segment ID 'seg_cons' using
    "generate_postfix" helper.

    Parameters
    ----------
    segs_consensus : pd.DataFrame
        Consensus segments with at least:
        - 'CHROM' (string-like),
        - 'seg_start' (int),
        - 'seg_end' (int),
        - optional 'cnv_state' (string); if absent, it will be created and filled.
    segs_neu : pd.DataFrame
        Neutral segments with:
        - 'CHROM' (string-like),
        - 'seg_start' (int),
        - 'seg_end' (int).

    Returns
    -------
    pd.DataFrame
        Combined segments containing original consensus plus inserted neutral
        gaps, with columns including:
        - 'CHROM', 'seg_start', 'seg_end', 'seg_length', 'cnv_state', 'seg_cons'.

    """
    # Convert segs_neu and segs_consensus to PyRanges    
    gr_neu = pr.PyRanges(chromosomes=segs_neu['CHROM'].astype('string'),
                         starts=segs_neu['seg_start'],
                         ends=segs_neu['seg_end'])
    gr_cons = pr.PyRanges(chromosomes=segs_consensus['CHROM'].astype('string'),
                          starts=segs_consensus['seg_start'],
                          ends=segs_consensus['seg_end'])
    
    gr_gaps = gr_neu.subtract(gr_cons)
    gaps = gr_gaps.as_df().rename(columns={'Chromosome':'CHROM','Start':'seg_start','End':'seg_end'})
    gaps['seg_length'] = gaps['seg_end'] - gaps['seg_start']
    gaps = gaps[gaps['seg_length']>0]
    segs_consensus['seg_length'] = segs_consensus['seg_end'] - segs_consensus['seg_start']
    combined = pd.concat([segs_consensus, gaps], ignore_index=True)
    if 'cnv_state' not in combined.columns:
        combined['cnv_state'] = np.nan
    combined['cnv_state'] = combined['cnv_state'].fillna('neu')
    combined.loc[:,'CHROM'] = combined.CHROM.astype('string')
    combined = combined.sort_values(['CHROM','seg_start'], key=natsort.natsort_keygen()).reset_index(drop=True)
    
    combined_group = combined.groupby('CHROM', group_keys=False, observed=True, sort=False)
    seg_cons = pd.Series(np.repeat(np.nan, combined.shape[0]), dtype='string')
    for k, group in combined_group:
        len_group = group.shape[0]
        postfix = generate_postfix(range(len_group))
        seg_cons[group.index] = group.CHROM.astype('string') + postfix
    combined.loc[:,'seg_cons'] = seg_cons
    # combined.loc[:,'CHROM'] = combined.CHROM.astype('category')

    return combined


def remove_up_down(s: str) -> str:
    """
    Remove a trailing suffix "_up" or "_down" from a state label.

    Parameters
    ----------
    s : str
        Input state label, e.g. "amp_up", "del_down", "neu".

    Returns
    -------
    str
        The input string with a terminal "_up"/"_down" stripped if present;
        otherwise the original string unchanged.

    Examples
    --------
    >>> remove_up_down("amp_up")
    'amp'
    >>> remove_up_down("neu")
    'neu'
    """
    return re.sub(r'(_up|_down)$', '', s)


def extract_up_down(s: str) -> Optional[str]:
    """
    Extract a trailing direction token ("up" or "down") from a state label.

    Parameters
    ----------
    s : str
        Input state label, e.g. "amp_up", "del_down", "neu".

    Returns
    -------
    Optional[str]
        "up" or "down" if the string ends with that token; otherwise "None".

    Examples
    --------
    >>> extract_up_down("amp_up")
    'up'
    >>> extract_up_down("neu") is None
    True
    """
    match = re.search(r'(up|down)$', s)
    return match.group(1) if match else None


def make_group_bulks(groups: Dict[str, Dict[str, Any]],
                     count_mat: Any,
                     df_allele: pd.DataFrame,
                     lambdas_ref: pd.DataFrame,
                     gtf: pd.DataFrame,
                     min_depth: int = 0,
                     nu: float = 1,
                     segs_loh: pd.DataFrame = None,
                     ncores: int = None,
                     filter_hla: bool = True,
                     filter_segments = None,
                     ) -> pd.DataFrame:
    """
    Build pseudobulk profiles for a collection of groups, in parallel.

    Parameters
    ----------
    groups
        Mapping from arbitrary group keys to group specifications. Each value must
        include `sample`, `members`, `cells`, and `size`.
    count_mat
        Gene-count container in `anndata.AnnData` format.
    df_allele
        Allelic counts table used for allele-mode emissions in the HMM. Must include
        the columns required by `get_bulk`.
    lambdas_ref
        Reference expression profiles (per-gene baseline λ). Columns/indices should
        match the genes in `count_mat` as expected by `get_bulk`.
    gtf
        Gene annotation metadata used by `get_bulk` to align features.
    min_depth
        Minimum allele depth (DP) threshold; loci below this are typically excluded.
    nu
        Phase switch rate or related parameter consumed by `get_bulk` to compute
        switch probabilities.
    segs_loh
        Optional table of segments with clonal LOH to be excluded from allelic tests.
    ncores
        Number of worker processes. Defaults to `min(len(groups), cpu_count())`.
        The value is clipped to `cpu_count()`.

    Returns
    -------
    pd.DataFrame
        Concatenated pseudobulk profiles for all groups. Rows are sorted
        “naturally” by `CHROM` and `POS`, then by `sample`, `snp_id`, `POS`. The
        output includes:
          - all columns returned by `get_bulk` (must include `CHROM`, `POS`,
            and `snp_id`),
          - `n_cells` (from `GroupSpec.size`),
          - `members` (semicolon-joined),
          - `sample` (from `GroupSpec.sample`),
          - `snp_index` (categorical code derived from `snp_id`).
        If `groups` is empty or all jobs fail, returns an empty DataFrame.

    Notes
    -----
    - This function delegates the per-group work to `process_group`, which in turn
      calls `get_bulk`.
    - Any group that raises an exception yields an error`; the error is
      logged via `log.error` and skipped in the final concatenation.
    """
    if not groups:
        return pd.DataFrame()

    if ncores is None:
        ncores = min(len(groups), cpu_count())
    # Ensure that ncores does not exceed the number of available CPUs
    ncores = min(ncores, cpu_count())

    # Prepare arguments to pass to the process_group function
    process_group_partial = partial(
        process_group,
        count_mat=count_mat,
        df_allele=df_allele,
        lambdas_ref=lambdas_ref,
        gtf=gtf,
        min_depth=min_depth,
        nu=nu,
        segs_loh=segs_loh,
        filter_hla=filter_hla,
        filter_segments=filter_segments
    )

    # Use joblib's Parallel and delayed for parallel processing
    results = Parallel(n_jobs=ncores)(
        delayed(process_group_partial)(g) for g in groups.values()
    )

    # Check for errors in the results
    bulks_list = []
    for res in results:
        if isinstance(res, dict) and 'error' in res:
            g = res['group']
            log.error(f"Job for sample {g['sample']} failed")
            log.error(str(res['error']))
        else:
            bulks_list.append(res)

    if not bulks_list:
        return pd.DataFrame()

    # Combine all bulks into a single DataFrame # TODO DONE!!! check the 3 lines in the middle
    bulks = pd.concat(bulks_list, ignore_index=True)
    # Arrange the DataFrame by 'CHROM' and 'POS'
    bulks = bulks.sort_values(['CHROM', 'POS'], key=natsort.natsort_keygen())
    # Modify 'snp_id' and 'snp_index' columns
    # Create a categorical type for 'snp_id' with categories in order of appearance
    snp_id_cat = pd.Categorical(bulks['snp_id'], categories=bulks['snp_id'].unique())
    bulks['snp_index'] = snp_id_cat.codes
    # Arrange by 'sample'
    bulks = bulks.sort_values(['sample','CHROM', 'POS'], key=natsort.natsort_keygen()).reset_index(drop=True)
    return bulks


def process_group(g: Dict[str, Any],
                  count_mat: Any,
                  df_allele: pd.DataFrame,
                  lambdas_ref: pd.DataFrame,
                  gtf: pd.DataFrame,
                  min_depth: int,
                  nu: float,
                  segs_loh: pd.DataFrame=None,
                  filter_hla=True,
                  filter_segments=None) -> Union[pd.DataFrame, Dict]:
    """
    Build a single group's pseudobulk by delegating to `get_bulk`, augmenting the
    result with group metadata.

    Parameters
    ----------
    g
        Group specification (sample, members, cells, size).
    count_mat
        Gene-count container in `anndata.AnnData` format, passed through to
        `get_bulk`.
    df_allele
        Allelic counts table passed through to `get_bulk`.
    lambdas_ref
        Reference expression profiles passed through to `get_bulk`.
    gtf
        Gene annotation metadata passed through to `get_bulk`.
    min_depth
        Minimum allele depth (DP) threshold for `get_bulk`.
    nu
        Phase switch rate or related parameter for `get_bulk`.
    segs_loh
        Optional clonal LOH segments for `get_bulk`.

    Returns
    -------
    pd.DataFrame
        Pseudobulk for the group with additional columns:
        `n_cells`, `members`, and `sample`.
    Error
        If an exception occurs, returns a dict with keys `error` and `group`.

    Notes
    -----
    The return value is designed to be consumed by `make_group_bulks`, which will
    handle errors and concatenate successful results.
    """
    try:
        # Extract the subset of cells
        subset_cells = g['cells']
        # Call get_bulk function for the group
        bulk = get_bulk(
            count_mat=count_mat,
            df_allele=df_allele,
            subset=subset_cells,
            lambdas_ref=lambdas_ref,
            gtf=gtf,
            min_depth=min_depth,
            nu=nu,
            segs_loh=segs_loh,
            filter_hla=filter_hla,
            filter_segments=filter_segments
        )
        # Add additional columns
        bulk['n_cells'] = g['size']
        bulk['members'] = ';'.join(map(str, g['members']))
        bulk['sample'] = g['sample']
        return bulk
    except Exception as e:
        return {'error': e, 'group': g}
    

def find_common_diploid(
    bulks: pd.DataFrame,
    grouping: Literal["clique", "component"] = "clique",
    gamma: float = 20.0,
    theta_min: float = 0.08,
    theta_max: float = 0.4,
    t: float = 1e-5,
    fc_min: float = 2 ** 0.25,
    alpha: float = 1e-4,
    min_genes: int = 10,
    ncores: int = 1,
    debug: bool = False,
    verbose: bool = True,
    ) -> pd.DataFrame:
    
    """
    Infer common diploid segments across samples from bulk allele data.

    The procedure runs a per-chromosome allelic HMM for each sample, smooths and
    annotates CNV segments, unions imbalanced segments across samples, fills
    neutral regions, and then identifies a common diploid baseline by comparing
    log fold-changes across segments and constructing a graph of mutually
    “similar” segments. The final output flags rows belonging to diploid segments.

    Parameters
    ----------
    bulks : pandas.DataFrame
        Long-form bulk table. Expected columns include:
        - 'sample': sample/group identifier (created as '1' if absent)
        - 'CHROM', 'seg', 'seg_start', 'seg_end'
        - 'pAD', 'DP', 'p_s' (allelic balance, depth, switch probability)
        - 'state', 'cnv_state' (will be created/overwritten)
        - 'loh' (optional, boolean; if present and True, overrides `cnv_state` to 'loh')
        - 'gene', 'POS', 'lnFC' (used for baseline selection)
    grouping : {'clique', 'component'}, default 'clique'
        How to aggregate the inter-segment similarity graph when choosing the
        diploid baseline:
        - 'component': use connected components
        - 'clique'   : use maximal cliques
    gamma : float, default 20.0
        HMM prior/penalty parameter forwarded to `run_allele_hmm_s5`.
    theta_min : float, default 0.08
        Lower bound of allelic imbalance (theta) used by the HMM.
    theta_max : float, default 0.4
        Upper bound of allelic imbalance (theta) used by the HMM.
    t : float, default 1e-5
        HMM state transition probability.
    fc_min : float, default 2**0.25
        Threshold on the maximum inter-segment FC ratio (after exponentiation) to
        optionally enable a “quadruploid state” choice for the baseline.
    alpha : float, default 1e-4
        FDR threshold (after BH correction) used when testing segment similarity
        across samples (kept if q > alpha).
    min_genes : int, default 10
        Minimum number of genes for segment smoothing (`smooth_segs`).
    ncores : int, default 1
        Maximum number of parallel workers (parallelized across samples).
    debug : bool, default False
        Debug flag passed through to internal utilities.
    verbose : bool, default True
        Verbosity flag passed through to internal utilities.

    Returns
    -------
    pandas.DataFrame
        The input table with updated columns. In particular:
        - 'cnv_state' : per-row segment state after HMM + smoothing (+ LOH override)
        - 'seg'       : consensus segment IDs
        - 'diploid'   : boolean flag, True for rows in the inferred diploid segments

    Notes
    -----
    - Parallelization is performed over samples using `joblib.Parallel`. Any
      exception raised within a worker is propagated to the caller.
    - If no balanced segments are detected, all segments are used as baseline,
      and a warning is emitted. If exactly one balanced segment is found, that
      segment is used.

    Raises
    ------
    Exception
        Any error encountered during parallel processing (e.g., within a per-sample
        HMM run) is caught and re-raised to halt the workflow.

    Examples
    --------
    >>> out = find_common_diploid(bulks_df, grouping="component", ncores=8)
    >>> out['diploid'].value_counts()
    """
    
    # Ensure 'sample' column exists
    if 'sample' not in bulks.columns:
        bulks = bulks.copy()
        bulks['sample'] = '0'
    
    # Define balanced regions in each sample
    sample_groups = [df for _, df in bulks.groupby('sample', observed=True, sort=False)]
    
    def process_bulk(bulk):
        bulk = bulk.copy()
        # Apply the HMM to each chromosome
        bulk['state'] = ''
        for chrom, group in bulk.groupby('CHROM', observed=True, sort=False):
            indices = group.index
            pAD = group['pAD'].values
            DP = group['DP'].values
            p_s = group['p_s'].values
    
            # Run HMM
            states = hmmlib.run_allele_hmm_s5(
                pAD=pAD,
                DP=DP,
                p_s=p_s,
                t=t,
                theta_min=theta_min,
                theta_max=theta_max,
                gamma=gamma
            )
            bulk.loc[indices, 'state'] = states
    
        # Process states to obtain CNV states
        bulk['cnv_state'] = bulk['state'].str.replace('_down|_up', '', regex=True)
        # Annotate and smooth segments
        bulk = annot_segs(bulk, var='cnv_state')
        bulk = hmmlib.smooth_segs(bulk, min_genes=min_genes)
        bulk = annot_segs(bulk, var='cnv_state')
        return bulk
    
    ncores = np.min((len(sample_groups), cpu_count(), ncores))
    log.info(f'Running diploid inference on {ncores} core')
    
    # Parallel processing with joblib
    if ncores > 1:
        try:
            results_list = Parallel(n_jobs=ncores)(
                delayed(process_bulk)(bulk) for bulk in sample_groups
            )
        except Exception as e:
            log.error(str(e))
            raise e
    else:
        results_list = [process_bulk(bulk) for bulk in sample_groups]
    
    # Combine results
    bulks = pd.concat(results_list,ignore_index=True)
    # If there's any loh:
    if 'loh' in bulks.columns and bulks['loh'].any():
        # Replace cnv_state with 'loh' where loh is True
        bulks['cnv_state'] = np.where(bulks['loh'], 'loh', bulks['cnv_state'])
        # Re-annotate segments per sample
        new_results = []
        for idx, group in bulks.groupby('sample', observed=True, sort=False):
            group = annot_segs(group, var='cnv_state')
            new_results.append(group)
        bulks = pd.concat(new_results, ignore_index=True)
    
    # Unionize imbalanced segs
    # Extract imbalanced segments:
    imbal = bulks[bulks['cnv_state'] != 'neu'].drop_duplicates(
        subset=['sample', 'CHROM', 'seg', 'seg_start', 'seg_end']
    )
    
    if imbal.shape[0] > 0:
    
        # Convert to PyRanges for union (reduce)
        gr = pr.PyRanges(pd.DataFrame({
            'Chromosome': imbal['CHROM'],
            'Start': imbal['seg_start'],
            'End': imbal['seg_end'] + 1
        }))
        
        reduced = gr.merge()  # Union of imbalanced segments
        segs_imbal = reduced.df.rename(columns={'Chromosome': 'CHROM', 'Start': 'seg_start', 'End': 'seg_end'})
        segs_imbal['seg_end'] = segs_imbal['seg_end'] - 1
        segs_imbal['seg_length'] = segs_imbal['seg_end'] - segs_imbal['seg_start'] + 1
        segs_imbal.CHROM = segs_imbal['CHROM'].astype("string")
        # Assign segment names and cnv_state
        segs_imbal = segs_imbal.sort_values(['CHROM', 'seg_start'], key=natsort.natsort_keygen())
        segs_imbal['seg'] = segs_imbal.groupby('CHROM', sort=False, observed=True).cumcount() + 1
        segs_imbal['seg'] = segs_imbal['CHROM'] + '_' + segs_imbal['seg'].astype("string")
        segs_imbal['cnv_state'] = 'theta_1'
        
        #TODO remove this
        log.info(f"segs_imbal shape is: {segs_imbal.shape}")
        log.info(f"bulks shape is: {bulks.shape}")
        #log.info(f"get_segs_neu(bulks) shape is: {get_segs_neu(bulks).shape}")


        segs_consensus = fill_neu_segs(segs_imbal, get_segs_neu(bulks))
        segs_consensus.loc[:,'seg'] = segs_consensus.loc[:,'seg_cons']
        
        bulks = annot_consensus(bulks, segs_consensus)
    
    segs_bal = bulks[bulks.cnv_state == 'neu']
    segs_bal_groups = segs_bal.groupby(['seg', 'sample'], observed=True, sort=False)
    
    seg = []
    sample = []
    n_snps = []
    n_genes = []
    for name, group in segs_bal_groups:
        seg.append(name[0])
        sample.append(name[1])
        n_snps.append(group[(group.DP >= 5) & (~group.DP.isna())].DP.shape[0])
        n_genes.append(group[(~group.gene.isna())].gene.shape[0])
        
    segs_bal_df = pd.DataFrame({'seg':seg,
                                'sample':sample,
                                'n_snps':n_snps,
                                'n_genes':n_genes})
    seg = []
    for name, group in segs_bal_df.groupby('seg', sort=False, observed=True):
        if np.any((group.n_genes > 50) & (group.n_snps > 50)): # any on both
            seg.append(name)
    segs_bal = natsorted(np.unique(seg))
    
    bulks_bal = bulks[(np.array([seg in segs_bal for seg in bulks.seg])) & (~bulks.lnFC.isna())]
    
    if len(segs_bal) == 0:
        msg = 'No balanced segments, using all segments as baseline'
        log.warning(msg)
        diploid_segs = bulks.seg.unique()
    elif len(segs_bal) == 1:
        diploid_segs = segs_bal
    else:
        bulk_temp = bulks_bal.loc[:,['gene', 'seg', 'POS', 'lnFC', 'sample']].copy()
        seg = []
        gene = []
        pos = []
        samples = {}
        for name, group in bulk_temp.groupby(by=['seg','gene','POS'], sort=False, observed=True):
            seg.append(name[0])
            gene.append(name[1])
            pos.append(name[2])
            for idx, row in group.iterrows():
                try:
                    samples[name[1]][row.loc['sample']] = row.lnFC
                except:
                    samples[name[1]] = {}
                    samples[name[1]][row.loc['sample']] = row.lnFC
    
        test_dat = pd.DataFrame({'seg':seg, 'gene':gene, 'POS':pos}).merge(pd.DataFrame(samples).T, left_on='gene', right_index=True)
        test_dat = test_dat.sort_values(['seg','POS'], key=natsort.natsort_keygen())
        test_dat = test_dat.drop_duplicates('gene').reset_index(drop=True)
        test_dat = test_dat.dropna()
        test_dat.seg = test_dat.seg.astype('string')
        test_dat = test_dat.drop(['gene', 'POS'], axis=1)
    
        samples = bulks_bal.loc[:,'sample'].unique()
    
        # Generate all (i, j, s) combos:
        combos = []
        for s in samples:
            for (i, j) in itertools.combinations(segs_bal, 2):
                combos.append((i, j, s))
        
        # Convert to DataFrame
        tests_df = pd.DataFrame(combos, columns=['i','j','s'])
        
        def compute_row(row):
                i = row['i']
                j = row['j']
                s = row['s']
                # filter from bulks_bal
                x = bulks_bal.loc[(bulks_bal['seg']==i) & (bulks_bal['sample']==s), 'lnFC']
                y = bulks_bal.loc[(bulks_bal['seg']==j) & (bulks_bal['sample']==s), 'lnFC']
                pval = t_test_pval(x, y)
                lnFC_i = x.mean() if len(x)>0 else np.nan
                lnFC_j = y.mean() if len(y)>0 else np.nan
                return pd.Series({'p': pval, 'lnFC_i': lnFC_i, 'lnFC_j': lnFC_j})
            
        results = tests_df.apply(compute_row, axis=1)
        tests_df = pd.concat([tests_df, results], axis=1)
        
        grouped = tests_df.groupby(['i','j'], as_index=False, observed=True, sort=False)
        def group_summarize(df):
            # simes_p of p
            p_simes = simes_p(df['p'].values, len(samples))
            lnFC_max_i = df['lnFC_i'].max()
            lnFC_max_j = df['lnFC_j'].max()
            delta_max = abs(lnFC_max_i - lnFC_max_j)
            return pd.Series({
                'p': p_simes,
                'lnFC_max_i': lnFC_max_i,
                'lnFC_max_j': lnFC_max_j,
                'delta_max': delta_max
            })
        
        summary_df = grouped.apply(group_summarize, include_groups=False)
        reject, qvals, _, _ = multipletests(summary_df.loc[:,'p'].values, alpha=0.05, method='fdr_bh')
        summary_df.loc[:,'q'] = qvals
        
        V = segs_bal
        E = summary_df[summary_df.loc[:,'q'] > alpha].copy()
        G = nx.Graph()
        G.add_nodes_from(V)
        for idx, row in E.iterrows():
            G.add_edge(row.i, row.j)
        
        if grouping == 'component':
            comps = list(nx.connected_components(G))
            fc_list = []
            for idx, seg_set in enumerate(comps):
                sub = bulks_bal[bulks_bal.loc[:,'seg'].isin(seg_set)]
                sub_group = sub.groupby('sample', as_index=False, observed=True, sort=False)['lnFC'].mean()
                sub_group.loc[:,'component'] = idx
                fc_list.append(sub_group)
                
            fc_df = pd.concat(fc_list, ignore_index=True)
            fc = fc_df.pivot(index='sample', columns='component', values='lnFC')
        
        else:
            all_cliques = list(nx.algorithms.clique.find_cliques(G))
            fc_list = []
            for idx, clique_set in enumerate(all_cliques):
                sub = bulks_bal[bulks_bal.loc[:,'seg'].isin(clique_set)]
                sub_group = sub.groupby('sample', as_index=False, observed=True, sort=False)['lnFC'].mean()
                sub_group = sub_group.rename(columns={'lnFC':idx})
                fc_list.append(sub_group)
            
            fc = reduce(lambda df1,df2: pd.merge(df1,df2,on='sample'), fc_list)
            fc = fc.set_index('sample')
        
        # Calculate diploid_cluster and fc_max
        rowmins = fc.apply(np.argmin, axis=1)
        c = Counter(rowmins)
        diploid_cluster = c.most_common(1)[0][0]
        fc_diff = fc.subtract(fc[diploid_cluster], axis=0)
        fc_max = fc_diff.values.ravel().max()
        fc_exp = np.exp(fc_max)
        
        if fc_exp > fc_min:
            log.info("quadruploid state enabled") # check if logging level is correct
            if grouping=='component':
                # find which segs are in that cluster
                comp_idx = diploid_cluster if diploid_cluster > 0 else 0
                diploid_segs = list(comps[comp_idx])
            else:
                # cliques
                comp_idx = diploid_cluster if diploid_cluster > 0 else 0
                diploid_segs = list(all_cliques[comp_idx])
            #bamp=True
        else:
            diploid_segs = segs_bal
            #bamp=False
    
    log.info(f"Diploid regions: {', '.join(natsort.natsorted(diploid_segs))}")
    
    bulks['diploid'] = bulks['seg'].isin(diploid_segs)

    return bulks


def theta_hat_seg(major_count: NDArray, minor_count: NDArray) -> float:
    """
    Estimate allelic-imbalance (theta) for a single segment.

    Theta is defined here as ``MAF - 0.5`` where
    ``MAF = major_total / (major_total + minor_total)``.

    Parameters
    ----------
    major_count : numpy.ndarray
        1-D array of non-negative counts for the *major* haplotype in the segment.
    minor_count : numpy.ndarray
        1-D array of non-negative counts for the *minor* haplotype in the segment.

    Returns
    -------
    float
        ``(major_total / (major_total + minor_total)) - 0.5``.
        Returns ``0.0`` if the total depth in the segment is zero.

    Notes
    -----
    - The result is negative when the minor haplotype dominates,
      positive when the major haplotype dominates, and 0 at balance.
    """
    major_total = major_count.sum()
    minor_total = minor_count.sum()
    denom = major_total + minor_total
    if denom == 0:
        # if no reads: define as 0
        return 0.0
    MAF = major_total / denom
    return MAF - 0.5


def theta_hat_roll(major_count: NDArray, minor_count: NDArray, h: int = 100) -> NDArray:
    """
    Rolling estimate of allelic-imbalance (theta) along a sequence.

    For each position ``c``, computes ``theta_hat_seg`` over a local window
    centered near ``c`` (clipped at the array boundaries). The window covers
    approximately ``2*h + 1`` positions around ``c``.

    Parameters
    ----------
    major_count : numpy.ndarray
        1-D array of non-negative counts for the *major* haplotype (length ``n``).
    minor_count : numpy.ndarray
        1-D array of non-negative counts for the *minor* haplotype (length ``n``).
    h : int, optional
        Half-window size controlling the local neighborhood used around each
        position, by default ``100``.

    Returns
    -------
    numpy.ndarray
        1-D array of length ``n`` with the rolling theta estimates.
    """
    n = len(major_count)
    out = np.zeros(n, dtype=float)
    for c in range(n):
        left_idx = max(c - h - 1, 0)
        right_idx = min(c + h, n - 1)
        seg_major = major_count[left_idx : right_idx + 1]
        seg_minor = minor_count[left_idx : right_idx + 1]
        out[c] = theta_hat_seg(seg_major, seg_minor)
    return out
    


def annot_theta_roll(df: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate haplotype major/minor counts and compute a per–chromosome rolling
    allelic-imbalance statistic (``theta_hat_roll``).

    The function:
      1) infers, for each SNP, which haplotype is *major* vs *minor*,
         using the optional Viterbi state when available and otherwise falling
         back to B-allele frequency;
      2) derives per-SNP ``major_count`` and ``minor_count`` from (pAD, DP);
      3) computes a rolling imbalance measure ``theta_hat_roll`` within each
         chromosome on rows with non-missing pAD, then merges it back and
         forward-fills missing values within each chromosome.

    Parameters
    ----------
    df : pandas.DataFrame
        Long-form allele table with at least the following columns:
        - ``CHROM`` : chromosome identifier (grouping key)
        - ``snp_id`` : SNP identifier (merge key)
        - ``pAD`` : alternate-allele count (numeric; NaN allowed)
        - ``DP`` : total read depth (numeric)
        - ``pBAF`` : B-allele frequency in [0, 1] (numeric)
        Optional:
        - ``state`` : string labels *without NaNs* used to define haplotype
          direction; values must end with "_up" or "_down" if present
          (e.g., "theta_1_up" / "theta_1_down"). If provided, it is
          used to assign major/minor; otherwise assignment is based on
          ``pBAF > 0.5`` (major) vs ``<= 0.5`` (minor).

    Returns
    -------
    pandas.DataFrame
        A copy of the input with the following additional columns:
        - ``haplo_theta_min`` : 'major' / 'minor' assignment per SNP
        - ``major_count`` : per-SNP count for the major haplotype
        - ``minor_count`` : per-SNP count for the minor haplotype
        - ``theta_hat_roll`` : rolling imbalance statistic (forward-filled
          within each chromosome)

    Notes
    -----
    - The rolling statistic is computed only on rows where ``pAD`` is non-missing
      and uses an external helper ``theta_hat_roll(major, minor, h=100)`` with a
      fixed window parameter ``h=100``.
    - Forward-filling is performed *within* each chromosome to propagate nearby
      estimates to missing positions.

    Raises
    ------
    AttributeError
        If a ``state`` column is present but contains non-strings/NaNs, due to
        the required string parsing (``*_up`` / ``*_down``).
    KeyError
        If required columns are missing.

    Examples
    --------
    >>> out = annot_theta_roll(df)
    >>> out[['CHROM', 'snp_id', 'major_count', 'minor_count', 'theta_hat_roll']].head()
    """

    # If Viterbi was run, define 'haplo_theta_min' from 'state',
    # else fallback to pBAF to decide major/minor.
    annot_test = pd.Series(np.repeat(np.nan, df.shape[0]), index=df.index, dtype='string')
    annot_test[df[[i.split('_')[-1] == 'up' for i in df.state]].index] = 'major'
    annot_test[df[[i.split('_')[-1] == 'down' for i in df.state]].index] = 'minor'
    annot_test[(annot_test.isna()) & (df.pBAF > 0.5)] = 'major'
    annot_test[(annot_test.isna()) & (df.pBAF <= 0.5)] = 'minor'
    df.loc[:,'haplo_theta_min'] = annot_test
    
    # Define major_count, minor_count
    major_count = pd.Series(np.repeat(np.nan, df.shape[0]), index=df.index, dtype=np.float64)
    major_count_idx = df[df.haplo_theta_min == 'major'].index
    minor_count_idx = df[df.haplo_theta_min == 'minor'].index
    major_count[major_count_idx] = df.pAD[major_count_idx].values
    major_count[minor_count_idx] = df.DP[minor_count_idx] - df.pAD[minor_count_idx]
    df.loc[:,'major_count'] = major_count
    df.loc[:,'minor_count'] = df.DP - df.major_count
    
    # Drop 'theta_hat_roll' if it exists
    if 'theta_hat_roll' in df.columns:
        df = df.drop(columns=['theta_hat_roll'])
    
    # We want to compute a rolling measure within each CHROM group,
    # only for rows with pAD != NaN
    def compute_theta_roll_for_group(grp):
        # Filter out rows where pAD is NaN
        sub = grp[grp['pAD'].notna()].copy()
        # Compute rolling measure
        major_arr = sub['major_count'].values
        minor_arr = sub['minor_count'].values
        # we define a new col
        sub['theta_hat_roll'] = theta_hat_roll(major_arr, minor_arr, h=100)
        # Return only the columns we want to merge on
        return sub[['CHROM', 'snp_id', 'theta_hat_roll']]
    
    # group by CHROM, apply the rolling measure
    df_roll = df.groupby('CHROM', group_keys=False, observed=True, sort=False)[df.columns].apply(compute_theta_roll_for_group)
    # Merge (left_join) on ['CHROM', 'snp_id']
    df_merged = df.merge(df_roll, on=['CHROM','snp_id'], how='left')
    # Perform forward fill each CHROM
    df_merged_groups = df_merged.groupby('CHROM', observed=True, sort=False)
    theta_hat_fill = pd.concat([v.theta_hat_roll.ffill() for k, v in df_merged_groups]).rename('theta_hat_roll')
    df_merged = df_merged.drop('theta_hat_roll', axis=1).merge(theta_hat_fill, left_index=True, right_index=True)

    return df_merged


def approx_theta_post(pAD, DP, p_s, lower=0.001, upper=0.499, start=0.25, gamma=20, disp=False):
    """
    Performs a Laplace approximation by optimizing -calc_allele_lik(),
    uses L-BFGS-B, and extracts the Hessian approx for variance.
    
    Parameters
    ----------
    pAD : array-like (list or np.ndarray)
        Variant allele depths (length n).
    DP : array-like
        Total allele depths (same length as pAD).
    p_s : array-like
        Variant allele frequency or phase switch probabilities (same length).
    lower : float
        Lower bound for theta.
    upper : float
        Upper bounds for theta.
    start : float
        Initial guess for theta.
    gamma : float
        Overdispersion param for Beta-binomial.

    Returns
    -------
    A dict with keys:
        'theta_mle': float, the MLE of theta
        'theta_sigma': float, the approximate stdev from Laplace approx
    """

    if len(pAD) <= 10:
        return {"theta_mle": 0.0, "theta_sigma": 0.0}
    
    # negative log-likelihood
    def objective(theta):
        return -hmmlib.calc_allele_lik(pAD, DP, p_s, theta, gamma)
    
    res = scipy.optimize.minimize(
        objective,
        x0=[start],    # initial guess
        method='L-BFGS-B',
        bounds=[(lower, upper)],  # single param => single tuple bound
        options={"maxiter": 1000,
                 #"disp": disp
                 }
    )
    
    mu = res.x[0]  # best param
    # Approx Hessian stuffs
    hess_inv_approx = getattr(res, 'hess_inv', None)
    if hess_inv_approx is not None:
        var_est = res.hess_inv.todense() # this is just res.hess_inv @ np.array([1.])
    else:
        # we can do a numerical approximation of second derivative if needed
        var_est = 0.0
    
    # var_est is actually the approximate inverse of the Hessian, sigma = sqrt(var_est)
    sigma = np.sqrt(var_est)

    return {
        "theta_mle": mu,
        "theta_sigma": sigma.ravel()[0]
    }


def l_bbinom(AD, DP, alpha, beta):
    """
    Calculate beta-binomial log-likelihood.
    Get log PMFs, then sums them.

    Parameters
    ----------
    AD : np.ndarray (or list)
        Variant (paternal) allele depths
    DP : np.ndarray (or list)
        Total allele depths
    alpha : float or np.ndarray
        Alpha parameter(s)
    beta : float or np.ndarray
        Beta parameter(s)

    Returns
    -------
    float
        The total (joint) log-likelihood under the beta-binomial model.
    """

    # Compute log PMF for each observation
    log_pmf = dist_prob.log_beta_binomial_pmf(k=AD,
                                              n=DP, 
                                              alpha=alpha,
                                              beta=beta)
    # Sum up the log PMF values
    total_log_lik = np.sum(log_pmf)

    return total_log_lik


def calc_allele_LLR(
    pAD: ArrayLike,
    DP: ArrayLike,
    p_s: ArrayLike,
    theta_mle: float,
    theta_0: float = 0.0,
    gamma: float = 20,
    ) -> float:
    """
    Compute the log-likelihood ratio (LLR) for allelic imbalance using a 2-state allele HMM.

    The LLR compares:
      - Alternative model (L1): Beta-Binomial emission HMM with imbalance set to theta_mle.
      - Null model (L0): independent Beta-Binomial with symmetric prior alpha = beta = 0.5 * gamma.

    By construction, LLR = L1 - L0.
    A positive LLR supports allelic imbalance relative to the null.

    Parameters
    ----------
    pAD : ArrayLike
        Phased alternate-allele depths per position (length N).
    DP : ArrayLike
        Total read depths per position (length N).
    p_s : ArrayLike
        Phase switch probabilities per position (length N). Used by the alternative
        HMM through calc_allele_lik. Not used in the null model.
    theta_mle : float
        Maximum likelihood estimate of the allelic imbalance parameter for the
        alternative hypothesis.
    theta_0 : float, optional
        Allelic imbalance under the null hypothesis. This parameter is accepted
        for API compatibility but is not used by the current implementation, which
        fixes the null to alpha = beta = 0.5 * gamma (default 0.0).
    gamma : float, optional
        Concentration parameter of the Beta-Binomial emissions. Larger values make
        the Beta prior more concentrated around its mean (default 20).

    Returns
    -------
    float
        Log-likelihood ratio L1 - L0. Returns 0.0 when there is only one or
        no observations.

    Examples
    --------
    >>> llr = calc_allele_LLR(pAD=[5, 8, 10], DP=[10, 16, 20], p_s=[0.01, 0.01, 0.01], theta_mle=0.1)
    >>> isinstance(llr, float)
    True
    """

    # If there's only 1 or fewer observations, return 0
    if len(pAD) <= 1:
        return 0.0

    # Alternative model log-likelihood
    l_1 = hmmlib.calc_allele_lik(pAD=pAD,
                          DP=DP,
                          p_s=p_s,
                          theta=theta_mle,
                          gamma=gamma)

    # Null model
    alpha_0 = gamma * 0.5
    beta_0  = gamma * 0.5

    l_0 = l_bbinom(AD=pAD,
                   DP=DP,
                   alpha=alpha_0,
                   beta=beta_0)

    # Return difference
    return l_1 - l_0


def log1mexp(x: float) -> float:
    """
    Compute log(1 - exp(-x)) in a numerically stable way for x >= 0.

    This chooses between two algebraically equivalent forms to reduce loss of precision:
      - if x <= log(2): use log(1 - exp(-x))
      - else:           use log1p(-exp(-x))

    Parameters
    ----------
    x : float
        Non-negative scalar input. Values must satisfy x >= 0.

    Returns
    -------
    float
        The value of log(1 - exp(-x)). As x -> 0+, the result -> -inf.
        As x -> +inf, the result -> 0.

    Raises
    ------
    ValueError
        If x < 0.
    """
    if x == 0:
        return -np.inf

    if x < 0:
        raise ValueError("Inputs need to be non-negative!")

    if x <= np.log(2):
        return np.log(-np.expm1(-x))
    else:
        return np.log1p(-np.exp(-x))


def pnorm_range_log(lower: float, upper: float, mu: float, sd: float) -> float:
    """
    Return log P(lower <= X <= upper) for X ~ Normal(mu, sd^2).

    This function computes the log of the probability mass of a normal
    distribution over a closed interval using log-CDFs for numerical
    stability. It handles the degenerate case sd == 0 by treating the
    distribution as a point mass at mu.

    Algorithm
    ---------
    1) If sd == 0, return 0.0 if mu is in [lower, upper], else return -np.inf.
    2) Compute l_upper = log CDF(upper; mu, sd).
    3) Compute l_lower = log CDF(lower; mu, sd).
    4) Combine them using a stable log-difference identity via log1mexp:
         log_prob = l_upper + log1mexp(l_upper - l_lower)

    Parameters
    ----------
    lower : float
        Lower bound of the interval. May be -np.inf.
    upper : float
        Upper bound of the interval. May be np.inf. Must satisfy lower <= upper.
    mu : float
        Mean of the normal distribution.
    sd : float
        Standard deviation of the normal distribution. Must be non-negative.

    Returns
    -------
    float
        Log probability that a Normal(mu, sd^2) random variable lies in
        [lower, upper]. Returns -np.inf if the probability is exactly zero.
    """

    if sd == 0:
        # If standard deviation is 0, the distribution is degenerate at mu.
        return 0.0 if (lower <= mu <= upper) else -np.inf # This is consistent in log space

    l_upper = scipy.stats.norm.logcdf(upper, loc=mu, scale=sd)
    l_lower = scipy.stats.norm.logcdf(lower, loc=mu, scale=sd)
    # Calculate log-prob
    log_prob = l_upper + log1mexp(l_upper - l_lower)
    return log_prob


def approx_phi_post(Y_obs, lambda_ref, d, mu=None, sig=None, lower_val=0.2, upper_val=10, start=1.0, disp=False):
    """
    Perform Laplace approximation for the parameter phi in a Poisson-lognormal model.

    This function finds the MLE of phi by minimizing the negative log-likelihood, and
    then extracts an approximate variance from the L-BFGS-B inverted Hessian operator.

    Parameters
    ----------
    Y_obs : array-like
        Gene expression counts.
    lambda_ref : array-like
        Reference expression levels, same length as Y_obs.
    d : float or array
        Total library size.
    mu : float or array, optional
        Mean(s) for the Poisson-lognormal model.
    sig : float or array, optional
        Standard deviation(s) for the Poisson-lognormal model.
    lower_val : float, optional
        Lower bound for phi (defaults to 0.2).
    upper_val : float, optional
        Upper bound for phi (defaults to 10).
    start : float, optional
        Starting guess for phi (defaults to 1.0).

    Returns
    -------
    dict
        A dictionary with two keys:
        - 'phi_mle': float, the maximum-likelihood estimate of phi.
        - 'phi_sigma': float, the approximate standard deviation from the inverse Hessian.
    """
    
    if len(Y_obs) == 0:
        return {"phi_mle": 1.0, "phi_sigma": 0.0}
    
    # Ensure start is within [lower, upper]
    start = max(min(start, upper_val), lower_val)
    
    # negative log-likelihood
    def objective(phi):
        return -dist_prob.l_lnpois(Y_obs, lambda_ref, d, mu, sig, phi=phi)
    
    # Optimize with L-BFGS-B
    res = scipy.optimize.minimize(
        objective,
        x0=[start],
        method='L-BFGS-B',
        bounds=[(lower_val, upper_val)],
        options={'maxiter': 1000,
                 #'disp': disp
                 }
        )
    
    mu = res.x[0]  # best param
    # Approx Hessian stuffs
    hess_inv_approx = getattr(res, 'hess_inv', None)
    if hess_inv_approx is not None:
        var_est = res.hess_inv.todense() # this is just res.hess_inv @ np.array([1.])
    else:
        # we can do a numerical approximation of second derivative if needed
        var_est = 0.0
    
    # var_est is actually the approximate inverse of the Hessian, sigma = sqrt(var_est)
    sigma = np.sqrt(var_est).ravel()[0]

    return {
        "phi_mle": mu,
        "phi_sigma": sigma
    }


def calc_exp_LLR(
    Y_obs: ArrayLike,
    lambda_ref: ArrayLike,
    d: ArrayLike | float,
    phi_mle: float,
    mu: Optional[float | NDArray] = None,
    sig: Optional[float | NDArray] = None,
    alpha: Optional[float | NDArray] = None,
    beta: Optional[float | NDArray] = None,
) -> float:
    """
    Compute the expression log-likelihood ratio (LLR) between an alternative
    overdispersion model and a null model.

    The alternative uses the Poisson lognormal likelihood with phi = phi_mle.
    The null uses the same likelihood with phi = 1. The function returns:
        LLR = l_lnpois(phi = phi_mle) - l_lnpois(phi = 1.0)

    Parameters
    ----------
    Y_obs : ArrayLike
        Observed counts per gene or feature (1-D).
    lambda_ref : ArrayLike
        Reference expression frequencies for the same genes or features (1-D).
        Must align with Y_obs.
    d : ArrayLike or float
        Library depth or scaling factor. May be a scalar or a vector aligned
        with Y_obs.
    phi_mle : float
        Maximum likelihood estimate of the overdispersion multiplier for the
        alternative model.
    mu : float or numpy.ndarray, optional
        Location parameter(s) of the lognormal component. If a scalar is given,
        it is applied to all entries. If an array is given, it must align with Y_obs.
    sig : float or numpy.ndarray, optional
        Scale parameter(s) (standard deviation) of the lognormal component.
        Scalar or array aligned with Y_obs.
    alpha : float or numpy.ndarray, optional
        Unused. Present for API compatibility.
    beta : float or numpy.ndarray, optional
        Unused. Present for API compatibility.

    Returns
    -------
    float
        Log-likelihood ratio. Returns 0.0 if Y_obs has length 0.
    """

    if len(Y_obs) == 0:
        return 0.0

    # Alternative model: phi=phi_mle
    l1 = dist_prob.l_lnpois(Y_obs, lambda_ref, d, mu, sig, phi=phi_mle)
    # Null model: phi=1
    l0 = dist_prob.l_lnpois(Y_obs, lambda_ref, d, mu, sig, phi=1.0)
    return l1 - l0


def classify_alleles(bulk: pd.DataFrame) -> pd.DataFrame:
    """
    Classify SNP haplotypes as major or minor using an allele HMM and a naive rule,
    and attach the results to the input table.

    Workflow
    --------
    1) Subset rows where:
       - cnv_state_post != "neu"
       - AD is not NA
       Then keep only groups (CHROM, seg) that contain more than one SNP.
    2) If the subset is empty, return the original DataFrame unchanged.
    3) For each (CHROM, seg) group in the subset:
       - Build an HMM with get_allele_hmm using pAD, DP, p_s and a single
         theta value per group (theta_mle must be constant within group).
       - Run forward_back_allele(hmm) to get posterior state probabilities
         per SNP; take the first column as p_up (posterior for the "up" state).
       - Define haplo_post per SNP by combining p_up with phased genotype GT:
           if p_up >= 0.5 and GT == "1|0" -> "major"
           if p_up >= 0.5 and GT == "0|1" -> "minor"
           if p_up <  0.5 and GT == "1|0" -> "minor"
           if p_up <  0.5 and GT == "0|1" -> "major"
         otherwise None.
       - Define haplo_naive as "minor" if AR < 0.5, else "major".
       - Return only columns [snp_id, p_up, haplo_post, haplo_naive] for that group.
    4) Left-join these outputs back to the original bulk on snp_id.

    Parameters
    ----------
    bulk : pandas.DataFrame
        Input table with at least the following columns:
        - cnv_state_post, AD, CHROM, seg, snp_id, GT, AR
        - pAD, DP, p_s, theta_mle (needed to build and run the HMM)

    Returns
    -------
    pandas.DataFrame
        A copy of bulk with three added columns for the relevant SNPs:
        - p_up: posterior probability of the "up" state
        - haplo_post: posterior-based haplotype label ("major" or "minor")
        - haplo_naive: naive AR-based haplotype label

    Notes
    -----
    - Groups with only a single SNP are excluded from HMM classification.
    """

    # Create 'allele_bulk'
    allele_bulk = bulk[
        (bulk['cnv_state_post'] != 'neu')
        & (bulk['AD'].notna())
    ].copy()

    # group by (CHROM, seg) and keep only groups with len >1
    group_counts = allele_bulk.groupby(['CHROM','seg'], observed=True, sort=False).size()
    valid_groups = group_counts[group_counts > 1].index 
    allele_bulk = allele_bulk.set_index(['CHROM','seg'])
    allele_bulk = allele_bulk.loc[valid_groups].reset_index()

    # If allele_bulk is empty, return original
    if allele_bulk.shape[0] == 0:
        return bulk

    def aggregator(df_group: pd.DataFrame) -> pd.DataFrame:

        unique_theta = df_group['theta_mle'].unique()
        theta_val = unique_theta[0]

        # Build the HMM dict
        hmm = hmmlib.get_allele_hmm(
            pAD = df_group['pAD'].values,
            DP  = df_group['DP'].values,
            p_s = df_group['p_s'].values,
            theta=theta_val,
            gamma=20
        )
        post_matrix = hmmlib.forward_back_allele(hmm)
       
        p_up = post_matrix[:, 0]

        # define haplo_post
        def define_haplo_post(row, pval):
            gt = row['GT']
            if pval >= 0.5 and gt=='1|0':
                return 'major'
            elif pval >=0.5 and gt=='0|1':
                return 'minor'
            elif pval < 0.5 and gt=='1|0':
                return 'minor'
            elif pval < 0.5 and gt=='0|1':
                return 'major'
            else:
                return None  # or handle other GT

        #p_up_array = []
        #haplo_post_array = []
        #haplo_naive_array= []
        #for i, row in df_group.iterrows():
        #    val_p_up = p_up_array.append(p_up[len(p_up_array)])
        
        # we get the next p_up
        df_out = df_group.copy()
        df_out['p_up'] = p_up
        # define haplo_post rowwise
        df_out['haplo_post'] = [
            define_haplo_post(r, pu)
            for r, pu in zip(df_out.to_dict('records'), df_out['p_up'])
        ]
        # define haplo_naive
        df_out['haplo_naive'] = np.where(df_out['AR']<0.5, 'minor','major')
        
        return df_out[['snp_id','p_up','haplo_post','haplo_naive']]

    # apply aggregator
    allele_post = allele_bulk.groupby(['CHROM','seg'], as_index=False, observed=True, sort=False)[allele_bulk.columns].apply(aggregator)
    # This yields a multi-level index. Flatten:
    allele_post = allele_post.reset_index(drop=True)

    # remove from 'bulk' any col from allele_post except 'snp_id'.
    drop_cols = [c for c in allele_post.columns if c not in ['snp_id']]
    to_remove = [c for c in drop_cols if c in bulk.columns]
    bulk = bulk.drop(columns=to_remove, errors='ignore')

    # do a merge on 'snp_id'
    bulk = bulk.merge(allele_post, on='snp_id', how='left')

    return bulk



def retest_cnv(bulk:pd.DataFrame,
               theta_min:float = 0.08,
               logphi_min:float = 0.25,
               gamma:float = 20,
               allele_only:bool = False,
               exclude_neu:bool = True) -> pd.DataFrame:
    """
    Retest copy-number variations (CNVs) in a pseudobulk profile. This function
    computes segment-level statistics and posterior estimates for CNVs, optionally
    using only allele-based data or including expression-based data.

    Parameters
    ----------
    bulk : pd.DataFrame
        A pseudobulk dataframe containing, at minimum, the following columns:
          - 'cnv_state' : str or categorical, e.g. 'neu', 'loh', etc.
          - 'CHROM' : chromosome indicator (numeric or string)
          - 'seg' : segment identifier
          - 'POS' : genomic position
          - 'gene' : optional gene identifier for counting distinct genes
          - 'pAD', 'DP', 'p_s' : allele data (pAD = paternal allele depth, DP = total
            allele depth, p_s = variant allele frequency or phase switch)
          - 'major_count', 'minor_count' : major/minor allele counts
          - 'Y_obs', 'lambda_ref', 'd_obs', 'mu', 'sig' : for expression-based modeling
            (if allele_only=False, used in expression-based computations)
        The dataframe can have additional columns as needed.

    theta_min : float, optional
        Minimum threshold for the allele imbalance parameter when calculating
        certain likelihood terms (default = 0.08).

    logphi_min : float, optional
        Parameter controlling expression fold-change range. For example, used when
        bounding phi in pnorm_range_log calls (default = 0.25).

    gamma : float, optional
        Dispersion parameter for the Beta-Binomial allele model (default = 20).

    allele_only : bool, optional
        If True, retesting uses only allele-based computations (summaries, LLR for allele).
        If False, includes expression-based computations as well (default = False).

    exclude_neu : bool, optional
        If True, filters out rows where 'cnv_state' == 'neu' before grouping. This can
        remove neutral segments from retesting (default = True).

    Returns
    -------
    pd.DataFrame
        A DataFrame containing summarized segment-level information
        and posterior estimates. Depending on whether `allele_only` is True or False,
        columns may include (but are not limited to):
          - 'CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state'
          - 'n_genes', 'n_snps'
          - 'theta_hat', 'theta_mle', 'theta_sigma'
          - 'p_loh', 'p_amp', 'p_del', 'p_bamp', 'p_bdel'
          - 'LLR_x', 'LLR_y', 'LLR'
          - 'phi_mle', 'phi_sigma' (if not allele_only)
          - Additional posterior probabilities, log-likelihood metrics, etc.
          - 'cnv_state_post' column indicates the post-retesting
            classification of the segment (e.g. 'loh', 'amp', 'del', 'bamp', 'bdel', or 'neu').
    """
    
    bulk = annot_theta_roll(bulk.copy())
    
    if exclude_neu:
        bulk = bulk[bulk.cnv_state != 'neu'].copy()
        bulk = bulk.reset_index(drop=True)
    
    G = {'11': 0.5,
         '20': 0.1,
         '10': 0.1,
         '21': 0.05,
         '31': 0.05,
         '22': 0.1,
         '00': 0.1}
    
    if allele_only:
        
        bulk_group = bulk.groupby(['CHROM', 'seg', 'cnv_state'], observed=True, sort=False)
        group_len = len(bulk_group)
        # chroms = np.zeros(group_len)
        chroms = []
        segs = []
        cnv_state = []
        n_genes = np.zeros(group_len)
        n_snps = np.zeros(group_len)
        seg_start = np.zeros(group_len)
        seg_end = np.zeros(group_len)
        theta_hat = np.zeros(group_len)
        theta_mle = np.zeros(group_len)
        theta_sigma = np.zeros(group_len)
        p_loh = np.repeat(1, group_len)
        p_amp = np.repeat(0, group_len)
        p_del = np.repeat(0, group_len)
        p_bamp = np.repeat(0, group_len)
        p_bdel = np.repeat(0, group_len)
        LLR_x = np.repeat(0, group_len)
        LLR_y = np.zeros(group_len)
        LLR = np.zeros(group_len)
        phi_mle = np.repeat(1, group_len)
        phi_sigma = np.repeat(0, group_len)
        
        for idx, curr_group in enumerate(bulk_group):
            
            group = curr_group[1]
            chroms.append(curr_group[0][0])
            segs.append(curr_group[0][1])
            cnv_state.append(curr_group[0][2])
            
            n_genes[idx] = group.gene[group.gene.notna()].unique().shape[0]
            n_snps[idx] = group.pAD[group.pAD.notna()].shape[0]
            seg_start[idx] = group.POS.min()
            seg_end[idx] = group.POS.max()
            theta_hat[idx] = theta_hat_seg(group.major_count[group.major_count.notna()], group.minor_count[group.minor_count.notna()])
            current_approx_theta_post = approx_theta_post(pAD=group.pAD[group.pAD.notna()].values,
                                                          DP=group.DP[group.pAD.notna()].values,
                                                          p_s=group.p_s[group.pAD.notna()].values,
                                                          gamma=gamma,
                                                          start=theta_hat[idx])
            theta_mle[idx] = current_approx_theta_post['theta_mle']
            theta_sigma[idx] = current_approx_theta_post['theta_sigma']
            LLR_y[idx] = calc_allele_LLR(group.pAD[group.pAD.notna()].values,
                                         group.DP[group.pAD.notna()].values,
                                         group.p_s[group.pAD.notna()].values,
                                         theta_mle[idx],
                                         gamma = gamma)
            LLR[idx] = LLR_x[idx] + LLR_y[idx]
        
        segs_post = pd.DataFrame({
            'CHROM':chroms,
            'seg':segs,
            'cnv_state':cnv_state,
            'n_genes':n_genes,
            'n_snps':n_snps,
            'seg_start':seg_start,
            'seg_end':seg_end,
            'theta_hat':theta_hat,
            'theta_mle':theta_mle,
            'theta_sigma':theta_sigma,
            'p_loh':p_loh,
            'p_amp':p_amp,
            'p_del':p_del,
            'p_bamp':p_bamp,
            'p_bdel':p_bdel,
            'LLR_x':LLR_x,
            'LLR_y':LLR_y,
            'LLR':LLR,
            'phi_mle':phi_mle,
            'phi_sigma':phi_sigma
        })
    
    else:
        bulk_group = bulk.groupby(['CHROM','seg', 'seg_start', 'seg_end', 'cnv_state'], observed=True, sort=False)
        group_len = len(bulk_group)
        # chroms = np.zeros(group_len)
        chroms = []
        segs = []
        cnv_state = []
        
        n_genes = np.zeros(group_len)
        n_snps = np.zeros(group_len)
        seg_start = np.zeros(group_len)
        seg_end = np.zeros(group_len)
        theta_hat = np.zeros(group_len)
        theta_mle = np.zeros(group_len)
        theta_sigma = np.zeros(group_len)
        L_y_n = np.zeros(group_len)
        L_y_d = np.zeros(group_len)
        L_y_a = np.zeros(group_len)
        phi_mle = np.zeros(group_len)
        phi_sigma = np.zeros(group_len)
        L_x_n = np.zeros(group_len)
        L_x_d = np.zeros(group_len)
        L_x_a = np.zeros(group_len)
        Z_cnv = np.zeros(group_len)
        Z_n = np.zeros(group_len)
        Z = np.zeros(group_len)
        logBF = np.zeros(group_len)
        p_neu = np.zeros(group_len)
        p_loh = np.zeros(group_len)
        p_amp = np.zeros(group_len)
        p_del = np.zeros(group_len)
        p_bamp = np.zeros(group_len)
        p_bdel = np.zeros(group_len)
        LLR_x = np.zeros(group_len)
        LLR_y = np.zeros(group_len)
        LLR = np.zeros(group_len)
        
        for idx, curr_group in enumerate(bulk_group):
            
            group = curr_group[1]
            chroms.append(curr_group[0][0])
            segs.append(curr_group[0][1])
            cnv_state.append(curr_group[0][-1])
            
            n_genes[idx] = group.gene[group.gene.notna()].unique().shape[0]
            n_snps[idx] = group.pAD[group.pAD.notna()].shape[0]
            seg_start[idx] = group.seg_start.min()
            seg_end[idx] = group.seg_end.max()
            theta_hat[idx] = theta_hat_seg(major_count = group.major_count[group.major_count.notna()],
                                           minor_count = group.minor_count[group.minor_count.notna()])
            current_approx_theta_post = approx_theta_post(pAD   = group.pAD[group.pAD.notna()].values,
                                                          DP    = group.DP[group.pAD.notna()].values,
                                                          p_s   = group.p_s[group.pAD.notna()].values,
                                                          gamma = gamma,
                                                          start = theta_hat[idx])
            theta_mle[idx] = current_approx_theta_post['theta_mle']
            theta_sigma[idx] = current_approx_theta_post['theta_sigma']
            L_y_n[idx] = pnorm_range_log(0, theta_min, theta_mle[idx], theta_sigma[idx])
            L_y_d[idx] = pnorm_range_log(theta_min, 0.499, theta_mle[idx], theta_sigma[idx])
            L_y_a[idx] = pnorm_range_log(theta_min, 0.375, theta_mle[idx], theta_sigma[idx])
            current_approx_phi_post = approx_phi_post(Y_obs      = group.Y_obs[group.Y_obs.notna()].values,
                                                      lambda_ref = group.lambda_ref[group.Y_obs.notna()].values,
                                                      d          = group.d_obs[group.Y_obs.notna()].unique(),
                                                      mu         = group.mu[group.Y_obs.notna()].values,
                                                      sig        = group.sig[group.Y_obs.notna()].values)
            phi_mle[idx] = current_approx_phi_post['phi_mle']
            phi_sigma[idx] = current_approx_phi_post['phi_sigma']
            L_x_n[idx] = pnorm_range_log(2**(-logphi_min), 2**logphi_min, phi_mle[idx], phi_sigma[idx])
            L_x_d[idx] = pnorm_range_log(0.1, 2**(-logphi_min), phi_mle[idx], phi_sigma[idx])
            L_x_a[idx] = pnorm_range_log(2**logphi_min, 3, phi_mle[idx], phi_sigma[idx])
            Z_cnv[idx] = hmmlib.log_sum_exp((np.log(G['20']) + L_x_n[idx] + L_y_d[idx],
                                      np.log(G['10']) + L_x_d[idx] + L_y_d[idx],
                                      np.log(G['21']) + L_x_a[idx] + L_y_a[idx],
                                      np.log(G['31']) + L_x_a[idx] + L_y_a[idx],
                                      np.log(G['22']) + L_x_a[idx] + L_y_n[idx], 
                                      np.log(G['00']) + L_x_d[idx] + L_y_n[idx]))
            Z_n[idx] = np.log(G['11']) + L_x_n[idx] + L_y_n[idx]
            Z[idx] = hmmlib.log_sum_exp((Z_n[idx], Z_cnv[idx]))
            logBF[idx] = Z_cnv[idx] - Z_n[idx]
            p_neu[idx] = np.exp(Z_n[idx] - Z[idx])
            p_loh[idx] = np.exp(np.log(G['20']) + L_x_n[idx] + L_y_d[idx] - Z_cnv[idx])
            p_amp[idx] = np.exp(np.log(G['31'] + G['21']) + L_x_a[idx] + L_y_a[idx] - Z_cnv[idx])
            p_del[idx] = np.exp(np.log(G['10']) + L_x_d[idx] + L_y_d[idx] - Z_cnv[idx])
            p_bamp[idx] = np.exp(np.log(G['22']) + L_x_a[idx] + L_y_n[idx] - Z_cnv[idx])
            p_bdel[idx] = np.exp(np.log(G['00']) + L_x_d[idx] + L_y_n[idx] - Z_cnv[idx])
            LLR_x[idx] = calc_exp_LLR(Y_obs      = group.Y_obs[group.Y_obs.notna()],
                                      lambda_ref = group.lambda_ref[group.Y_obs.notna()],
                                      d          = group.d_obs[group.Y_obs.notna()].unique(),
                                      phi_mle    = phi_mle[idx],
                                      mu         = group.mu[group.Y_obs.notna()],
                                      sig        = group.sig[group.Y_obs.notna()])
            LLR_y[idx] = calc_allele_LLR(pAD       = group.pAD[group.pAD.notna()].values,
                                         DP        = group.DP[group.pAD.notna()].values,
                                         p_s       = group.p_s[group.pAD.notna()].values,
                                         theta_mle = theta_mle[idx],
                                         gamma     = gamma)
    
        segs_post = pd.DataFrame({
            'CHROM':chroms,
            'seg':segs,
            'seg_start':seg_start,
            'seg_end':seg_end,
            'cnv_state':cnv_state,
            'n_genes':n_genes,
            'n_snps':n_snps,
            'theta_hat':theta_hat,
            'theta_mle':theta_mle,
            'theta_sigma':theta_sigma,
            'L_y_n':L_y_n,
            'L_y_d':L_y_d,
            'L_y_a':L_y_a,
            'phi_mle':phi_mle,
            'phi_sigma':phi_sigma,
            'L_x_n':L_x_n,
            'L_x_d':L_x_d,
            'L_x_a':L_x_a,
            'Z_cnv':Z_cnv,
            'Z_n':Z_n,
            'Z':Z,
            'logBF':logBF,
            'p_neu':p_neu,
            'p_loh':np.where(np.isnan(p_loh), 0, p_loh),
            'p_amp':np.where(np.isnan(p_amp), 0, p_amp),
            'p_del':np.where(np.isnan(p_del), 0, p_del),
            'p_bamp':np.where(np.isnan(p_bamp), 0, p_bamp),
            'p_bdel':np.where(np.isnan(p_bdel), 0, p_bdel),
            'LLR_x':LLR_x,
            'LLR_y':LLR_y,
            #'LLR': LLR_x + LLR_y,
            'LLR': logBF # This is in the original implementation, overwriting 'LLR' as defined before. I guess it is wrong.
        })
    
        segs_post.loc[:,'cnv_state_post'] = segs_post.loc[:,['p_loh', 'p_amp', 'p_del', 'p_bamp', 'p_bdel']].idxmax(axis=1).apply(lambda x: x.split('_')[-1])
        segs_post.loc[:,'cnv_state_post'] = ['neu' if segs_post.p_neu[i] >= 0.5 else segs_post.cnv_state_post[i] for i in segs_post.index]
        segs_post = segs_post.astype({'CHROM':"string", #np.int64,
                  'seg':'string',
                  'seg_start':np.int64,
                  'seg_end':np.int64,
                  'cnv_state':'string',
                  'n_genes':np.int64,
                  'n_snps':np.int64,
                  'theta_hat':np.float64,
                  'theta_mle':np.float64,
                  'theta_sigma':np.float64,
                  'L_y_n':np.float64,
                  'L_y_d':np.float64,
                  'L_y_a':np.float64,
                  'phi_mle':np.float64,
                  'phi_sigma':np.float64,
                  'L_x_n':np.float64,
                  'L_x_d':np.float64,
                  'L_x_a':np.float64,
                  'Z_cnv':np.float64,
                  'Z_n':np.float64,
                  'Z':np.float64,
                  'logBF':np.float64,
                  'p_neu':np.float64,
                  'p_loh':np.float64,
                  'p_amp':np.float64,
                  'p_del':np.float64,
                  'p_bamp':np.float64,
                  'p_bdel':np.float64,
                  'LLR_x':np.float64,
                  'LLR_y':np.float64,
                  'LLR':np.float64,
                  'cnv_state_post':'string'})
    return segs_post


def phi_hat_seg(
    Y_obs: ArrayLike,
    lambda_ref: ArrayLike,
    d: float | ArrayLike,
    mu: ArrayLike,
    sig: ArrayLike,
    ) -> float:
    """
    Estimate a segment-level expression fold change (phi) under a Poisson lognormal view.

    The estimator is:
        phi = exp( mean( log(Y_obs / d) - log(lambda_ref) - mu ) )

    It computes the average log fold change between the observed rate
    (Y_obs divided by depth d) and the reference rate lambda_ref, centers by mu,
    and then returns the value in linear space.

    Parameters
    ----------
    Y_obs : array-like
        Observed counts for all genes in the segment. Must be positive where
        log is applied.
    lambda_ref : array-like
        Reference expression frequencies for the same genes. Must be positive
        where log is applied.
    d : float or array-like
        Library size or scaling factor. May be a scalar or broadcastable to
        the shape of Y_obs.
    mu : array-like
        Log-scale mean term. Must be indexable and broadcastable to the shape
        of Y_obs. This function slices mu, so it should be a one-dimensional
        array for use with rolling callers.
    sig : array-like
        Log-scale standard deviation term. Present for signature consistency;
        not used in the computation here.

    Returns
    -------
    float
        The estimated fold change phi for the segment.
    """
    
    logFC = np.log(Y_obs / d) - np.log(lambda_ref)
    val = np.exp(np.mean(logFC - mu))

    return val


def phi_hat_roll(
    Y_obs: NDArray[np.floating],
    lambda_ref: NDArray[np.floating],
    d_obs: pd.Series | ArrayLike,
    mu: NDArray[np.floating],
    sig: NDArray[np.floating],
    h: int,
    ) -> NDArray[np.floating]:
    """
    Compute a rolling estimate of expression fold change (phi) across positions.

    For each position c in [0, n-1], the function forms a window
    [max(c - h - 1, 0), ..., min(c + h, n - 1)], computes phi_hat_seg on that
    slice, and stores the result at index c. The output is a length-n array of
    rolling phi estimates.

    Parameters
    ----------
    Y_obs : numpy.ndarray of shape (n,)
        Observed counts per position. Must be positive where log is applied.
    lambda_ref : numpy.ndarray of shape (n,)
        Reference expression frequencies per position. Must be positive where
        log is applied.
    d_obs : pandas.Series or array-like
        Library size or scaling factor.
    mu : numpy.ndarray of shape (n,)
        Log-scale mean per position. Must be indexable and aligned with Y_obs.
    sig : numpy.ndarray of shape (n,)
        Log-scale standard deviation per position. Present for signature
        consistency; not used by phi_hat_seg but sliced here to match mu.
    h : int
        Half-window size.

    Returns
    -------
    numpy.ndarray
        Array of length n with rolling phi estimates.

    Examples
    --------
    >>> n = 5
    >>> Y = np.array([10, 12, 9, 11, 13], dtype=float)
    >>> lam = np.array([0.01, 0.012, 0.009, 0.011, 0.013], dtype=float)
    >>> d = np.sum(Y)
    >>> mu = np.zeros(n, dtype=float)
    >>> sig = np.ones(n, dtype=float)
    >>> phi_hat_roll(Y, lam, d, mu, sig, h=1).shape
    (5,)
    """

    n = Y_obs.size
    if mu.size == 1 and sig.size == 1:
        mu  = np.full(n, mu.item(), dtype=float)
        sig = np.full(n, sig.item(), dtype=float)

    # Store results
    out = np.zeros(n, dtype=float)

    for c in range(n):

        left = max((c - h - 1), 0)
        right = min((c + h), n - 1)
        Y_slice       = Y_obs[left : right+1]
        lambda_slice  = lambda_ref[left : right+1]
        mu_slice      = mu[left : right+1]
        sig_slice     = sig[left : right+1]

        out[c] = phi_hat_seg(Y_slice, lambda_slice, np.unique(d_obs[~d_obs.isna()]), mu_slice, sig_slice)

    return out


def analyze_bulk(
    bulk: pd.DataFrame,
    t: float = 1e-5,
    gamma: float = 20,
    theta_min: float = 0.08,
    logphi_min: float = 0.25,
    nu: float = 1,
    min_genes: int = 10,
    exp_only: bool = False,
    allele_only: bool = False,
    bal_cnv: bool = True,
    retest: bool = True,
    find_diploid: bool = True,
    diploid_chroms: list = None,
    classify_allele: bool = False,
    run_hmm: bool = True,
    prior=None,
    exclude_neu: bool = True,
    # phasing: bool = True,
    verbose: bool = True
    ) -> pd.DataFrame:
    """
    Call joint HMM to infer CNVs in a pseudobulk profile.

    Parameters
    ----------
    bulk : pd.DataFrame
        Pseudobulk profile with columns (e.g., DP, pAD, CHROM, Y_obs, lambda_ref, d_obs, etc.).
    t : float
        Transition probability.
    gamma : float
        Dispersion parameter for the Beta-Binomial allele model.
    theta_min : float
        Minimum imbalance threshold.
    logphi_min : float
        Minimum log expression deviation threshold.
    nu : float
        Phase switch rate.
    min_genes : int
        Minimum number of genes to call an event.
    exp_only : bool
        Whether to run expression-only HMM.
    allele_only : bool
        Whether to run allele-only HMM.
    bal_cnv : bool
        Whether to call balanced amplifications/deletions.
    retest : bool
        Whether to retest CNVs after Viterbi decoding (not shown in snippet).
    find_diploid : bool
        Whether to run diploid region identification routine.
    diploid_chroms : list
        Chromosomes known to be diploid.
    classify_allele : bool
        Whether to only classify allele states (internal use).
    run_hmm : bool
        Whether to run HMM (internal use).
    prior : np.ndarray or list or None
        Prior probabilities of states (internal use).
    exclude_neu : bool
        Whether to exclude neutral segments from retesting (internal use).
    # phasing : bool
    #     Whether to use phasing information (internal use).
    verbose : bool
        Verbosity.

    Returns
    -------
    pd.DataFrame
        Updated pseudobulk DataFrame with CNV calls and states.
    """
    
    # checks
    if not isinstance(t, (int, float)):
        raise ValueError("Transition probability (t) is not numeric")
    if bulk['DP'].isna().all():
        raise ValueError("No allele data (all DP are NA)")
    if 'gamma' in bulk.columns:
        bulk = bulk.drop(columns=['gamma'])
        
    # Update transition probability 'p_s'
    if 'inter_snp_cm' not in bulk.columns:
        raise ValueError("'bulk' is missing required column 'inter_snp_cm'")
    bulk = bulk.loc[:,[col != 'p_s' for col in bulk.columns]].copy()
    bulk.loc[:,'p_s'] = switch_prob(bulk['inter_snp_cm'], nu=nu)

    # Determine diploid regions
    if exp_only or allele_only:
        bulk['diploid'] = True
    elif diploid_chroms is not None:
        if verbose:
            log.info(f"Using diploid chromosomes given: {', '.join(diploid_chroms)}")
        bulk['diploid'] = bulk['CHROM'].isin(diploid_chroms)
    else:
        # print(find_diploid)
        if find_diploid:
            bulk = find_common_diploid(
                bulk,
                gamma=gamma,
                t=t,
                theta_min=theta_min,
                min_genes=min_genes,
                fc_min=2**logphi_min
            )
        else:
            if 'diploid' not in bulk.columns:
                raise ValueError("Must define diploid region if not given and not found automatically.")

    # Fit expression baseline if not allele_only
    if not allele_only:
        cond = (
            ~bulk['Y_obs'].isna() &
            (bulk['logFC'] < 8) &
            (bulk['logFC'] > -8) &
            (bulk['diploid'])
        )
        bulk_baseline = bulk[cond].copy()

        if len(bulk_baseline) == 0:
            if verbose:
                log.warning("No genes left in diploid regions, using all genes as baseline")
            bulk_baseline = bulk[~bulk['Y_obs'].isna()].copy()

        d_obs_unique = bulk_baseline['d_obs'].dropna().unique()

        fit = dist_prob.fit_lnpois(
            bulk_baseline['Y_obs'].values,
            bulk_baseline['lambda_ref'].values,
            d_obs_unique
        )
        # fit is (mu, sig)
        mu_hat, sig_hat = fit

        # save mu, sig in DataFrame
        bulk.loc[:,'mu'] = mu_hat
        bulk.loc[:,'sig'] = sig_hat
    else:
        bulk[:,'mu'] = np.nan
        bulk[:,'sig'] = np.nan

    # Run the HMM if run_hmm is True
    if run_hmm:

        def apply_run_hmm(df_group):
            # gamma_val might be per-chrom or use the arg 'gamma'
            gamma_val = gamma

            # run_joint_hmm_s15 returns states for each row
            states = hmmlib.run_joint_hmm_s15(
                pAD            = df_group['pAD'].values,
                DP             = df_group['DP'].values,
                p_s            = df_group['p_s'].values,
                Y_obs          = df_group['Y_obs'].values,
                lambda_ref     = df_group['lambda_ref'].values,
                d_total        = df_group['d_obs'].dropna().unique(),
                phi_amp        = 2**(logphi_min),
                phi_del        = 2**(-logphi_min),
                mu             = df_group['mu'].values,
                sig            = df_group['sig'].values,
                t              = t,
                gamma          = gamma_val,
                theta_min      = theta_min,
                prior          = prior,
                bal_cnv        = bal_cnv,
                exp_only       = exp_only,
                allele_only    = allele_only,
                classify_allele= classify_allele,
                # phasing        = phasing
            )
            return pd.Series(states, index=df_group.index, name='state')

        bulk.loc[:,'state'] = bulk.groupby('CHROM', group_keys=False, observed=True, sort=False).apply(apply_run_hmm, include_groups=False)

        if 'loh' in bulk.columns:
            bulk.loc[bulk['loh'] == True, 'state'] = 'del_up'

        bulk.loc[:,'cnv_state'] = bulk['state'].str.replace(r'_down|_up', '', regex=True)

        # Then annotate, smooth, re-annotate
        bulk = annot_segs(bulk, var='cnv_state')
        bulk = hmmlib.smooth_segs(bulk, min_genes=min_genes)
        bulk = annot_segs(bulk, var='cnv_state')

    if retest and not exp_only:

        if verbose:
            print('Retesting CNVs..')    
        bulk_temp = bulk.copy()
        segs_post = retest_cnv(bulk_temp,
                               gamma=gamma,
                               theta_min=theta_min,
                               logphi_min=logphi_min,
                               exclude_neu=exclude_neu,
                               allele_only=allele_only
                              )
 
        col_to_discard = set(segs_post.columns.difference(('seg', 'CHROM', 'seg_start', 'seg_end')))
        bulk = bulk.loc[:,[i not in col_to_discard for i in bulk.columns]]
        bulk = bulk.merge(segs_post, on=('seg', 'CHROM', 'seg_start', 'seg_end'), how='left')
        bulk.loc[:,'cnv_state_post'] = np.where(bulk.cnv_state_post.isna(), 'neu', bulk.cnv_state_post)
        bulk.loc[:,'cnv_state'] = np.where(bulk.cnv_state.isna(), 'neu', bulk.cnv_state)
        
        # Force segments with clonal LOH to be deletion
        bulk.loc[:,'cnv_state_post'] = np.where(bulk['loh'], 'del', bulk['cnv_state_post'])
        bulk.loc[:,'cnv_state']      = np.where(bulk['loh'], 'del', bulk['cnv_state'])
        bulk.loc[:,'p_del']          = np.where(bulk['loh'], 1, bulk['p_del'])
        bulk.loc[:,'p_amp']          = np.where(bulk['loh'], 0, bulk['p_amp'])
        bulk.loc[:,'p_neu']          = np.where(bulk['loh'], 0, bulk['p_neu'])
        bulk.loc[:,'p_loh']          = np.where(bulk['loh'], 0, bulk['p_loh'])
        bulk.loc[:,'p_bdel']         = np.where(bulk['loh'], 0, bulk['p_bdel'])
        bulk.loc[:,'p_bamp']         = np.where(bulk['loh'], 0, bulk['p_bamp'])
        
        events = bulk.cnv_state_post.isin(set(('amp','del','loh'))) & ~bulk.cnv_state.isin(set(('bamp','bdel')))
        state_suffix = r'(up_1|down_1|up_2|down_2|up|down|1_up|2_up|1_down|2_down)'
        def tryextract(pattern, x):
            try: 
                return re.search(pattern, x).group(1)
            except AttributeError:
                return np.nan
        suffix_out = bulk.state[events].apply(lambda x : tryextract(state_suffix, x)).astype("string")
        naindex = suffix_out[suffix_out.isna()].index
        suffix_out[naindex] = bulk.cnv_state_post[naindex].astype("string")
        new_sp = pd.Series(np.repeat(np.nan, bulk.shape[0]), dtype='string')
        new_sp[events] = ['_'.join((sp, su)) for sp, su in zip(bulk.cnv_state_post[suffix_out.index].values, suffix_out.values)]
        new_sp[~events] = bulk.cnv_state_post[~events].astype('string')
        bulk.loc[:,'state_post'] = new_sp
        bulk['state_post'] = bulk['state_post'].str.replace(r'_NA$', '', regex=True)
        bulk = classify_alleles(bulk)
        bulk = annot_theta_roll(bulk)
    
    else:
        bulk.loc[:,'state_post'] = bulk.state
        bulk.loc[:,'cnv_state_post'] = bulk.cnv_state  ## OK 'till here
    
    
    if not allele_only:
        
        bulk_groups = bulk.groupby(['seg'], observed=True, sort = False)
        phi_post_dict = pd.DataFrame({'phi_mle':np.zeros(bulk.shape[0]), 'phi_sigma':np.zeros(bulk.shape[0])}, index=bulk.index) # recently added index
        for k, group in bulk_groups:
            phi_current = approx_phi_post(Y_obs = group.Y_obs[~group.Y_obs.isna()],
                                          lambda_ref = group.lambda_ref[~group.Y_obs.isna()],
                                          d = group.d_obs[~group.d_obs.isna()].unique(),
                                          mu = group.mu[~group.Y_obs.isna()],
                                          sig = group.sig[~group.Y_obs.isna()]
                                         )
            for phi_status, val in phi_current.items():
                phi_post_dict.loc[group.index, phi_status] = val
        bulk = bulk.loc[:,[i not in set(('phi_mle', 'phi_sigma')) for i in bulk.columns]]
        bulk = bulk.merge(phi_post_dict, left_index=True, right_index=True)
        
        bulk = bulk.loc[:,[i != 'phi_mle_roll' for i in bulk.columns]]
        bulk_groups = bulk.groupby('CHROM', observed=True, sort=False)
        phi_mle_roll = pd.Series(np.repeat(np.nan, bulk.shape[0]), name='phi_mle_roll')
        
        for k, group in bulk_groups:
            tmp_group = group[(~group.Y_obs.isna()) & (group.Y_obs > 0)].copy()
            phi_mle_roll[tmp_group.index] = phi_hat_roll(Y_obs = tmp_group.Y_obs,
                                                         lambda_ref = tmp_group.lambda_ref,
                                                         d_obs = tmp_group.d_obs,
                                                         mu = tmp_group.mu,
                                                         sig = tmp_group.sig,
                                                         h = 50)
        
        bulk = bulk.merge(phi_mle_roll, left_index=True, right_index=True)
        bulk.loc[:,'phi_mle_roll'] = bulk.phi_mle_roll.ffill()
    
    bulk.loc[:,'nu'] = nu
    bulk.loc[:,'gamma'] = gamma

    return bulk


def annot_theta_mle(bulk: pd.DataFrame) -> pd.DataFrame:
    """
    Annotate per-segment theta MLE and sigma.

    Parameters
    ----------
    bulk : pd.DataFrame
        Must contain columns: 'CHROM','seg','cnv_state','pAD','DP','p_s'.

    Returns
    -------
    pd.DataFrame
        Original bulk with two new columns merged by ['CHROM','seg']:
        'theta_mle', 'theta_sigma'.
    """
    required = {"CHROM", "seg", "cnv_state", "pAD", "DP", "p_s"}
    missing = required - set(bulk.columns)
    if missing:
        raise ValueError(f"bulk is missing required columns: {sorted(missing)}")

    def _fit_group(g: pd.DataFrame) -> pd.Series:
        # filter non-neutral rows
        g2 = g[g["cnv_state"] != "neu"]
        if g2.empty:
            return pd.Series({"theta_mle": np.nan, "theta_sigma": np.nan})

        # make mask pAD[pAD.notna()] to index all three arrays
        mask = g2["pAD"].notna().to_numpy()
        if not mask.any():
            return pd.Series({"theta_mle": np.nan, "theta_sigma": np.nan})

        pAD = g2.loc[mask, "pAD"].to_numpy()
        DP  = g2.loc[mask, "DP"].to_numpy()
        p_s = g2.loc[mask, "p_s"].to_numpy()

        est = approx_theta_post(pAD, DP, p_s, gamma=30, start=0.1)
        # approx_theta_post returns dict with 'theta_mle' and 'theta_sigma'
        return pd.Series({"theta_mle": est["theta_mle"], "theta_sigma": est["theta_sigma"]})

    # compute per-segment estimates
    theta_est = (
        bulk.groupby(["CHROM", "seg"], sort=False, as_index=False)
            .apply(lambda g: _fit_group(g))
            .reset_index(drop=True)
    )

    # left-join back to bulk
    out = bulk.merge(theta_est, on=["CHROM", "seg"], how="left")
    return out



