#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:35:09 2024

@author: carlino.calogero
"""
import string
import re

import numpy as np
import pandas as pd
import anndata as ad

import pyranges as pr
from pyranges import PyRanges

import scipy
from scipy.stats import ttest_ind

import natsort
from natsort import natsorted

from typing import Dict, List, Union, Sequence, Any, Optional, Iterable
from numpy.typing import NDArray, ArrayLike

from collections import Counter
import tqdm
import warnings

from spacenumbat import hmm, dist_prob
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
    # Rename some gtf columns. Needed for PyRanges.
    gtf = gtf.rename(columns={'gene_start':'Start', 'gene_end':'End', 'CHROM':'Chromosome'})

    # Take unique SNPs and rename columns for PyRanges
    snps = df[['snp_id', 'CHROM', 'POS']].drop_duplicates()
    snps = snps.rename(columns={'CHROM':'Chromosome', 'POS':'Start'})
    snps.loc[:, 'End'] = snps.loc[:, 'Start']

    # Create PyRanges for SNPs and GTF
    snps_pr = PyRanges(df=snps)
    gtf_pr = PyRanges(df=gtf)

    # Find overlaps between SNPs and genes, remove duplicates
    hits = snps_pr.join(gtf_pr).df.drop_duplicates('snp_id')

    # Add gene names to snps
    snps = snps.merge(hits.loc[:, ['snp_id', 'gene']], on='snp_id', how='left')

    # Left join with SNPs to original df (dropping existing gene columns)
    df = df.loc[:, df.columns.difference(['gene', 'gene_start', 'gene_end'], sort=False)].merge(
        snps.loc[:, ['snp_id', 'gene']], on='snp_id', how='left')

    return df
    

def check_anndata(count_ad:ad.AnnData, count_to_int:bool=True, fix_names:bool=True) -> ad.AnnData:
    """
    Validate and preprocess an AnnData object for downstream analysis.

    This function performs several checks and modifications on an AnnData object to ensure it is
    properly formatted for analysis:
    1. Converts the `.X` attribute to a CSC (Compressed Sparse Column) matrix if it is a dense NumPy array.
    2. Ensures that the data in `.X` are of integer type, converting if necessary and allowed.
    3. Checks for duplicate gene names in `var_names` and makes them unique if required.

    Parameters
    ----------
    count_ad : AnnData
        The AnnData object containing the count matrix and associated metadata.
    count_to_int : bool, optional (default: True)
        If True, converts the count matrix to 32-bit integers if it is not already of integer type.
        If False, raises a ValueError when the count matrix is not of integer type.
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
        If `count_to_int` is False and the count matrix is not of integer type.
        If `fix_names` is False and duplicate gene names are present.

    Notes
    -----
    This function is intended to standardize the format of AnnData objects before analysis,
    ensuring consistency in data types and uniqueness of gene identifiers.
    """
    # Convert .X to CSC matrix if it's a dense NumPy array
    if isinstance(count_ad.X, np.ndarray):
        count_ad.X = scipy.sparse.csc_matrix(count_ad.X)
    # Raise an error if .X is neither a NumPy array nor a SciPy sparse matrix
    elif not scipy.sparse.issparse(count_ad.X):
        msg = (f'You passed an object with an .X of type {type(count_ad.X)}. '
               'count_ad.X should be a NumPy array or a SciPy sparse CSC matrix.')
        raise ValueError(msg)

    # Check if the data in .X are of integer type
    if not np.issubdtype(count_ad.X.dtype, np.integer):
        if count_to_int:
            msg = (f'The count matrix in the supplied count_ad is of type {count_ad.X.dtype}. '
                   f'Converting to {np.int32}.')
            warnings.warn(msg)
            count_ad.X = count_ad.X.astype(np.int32)
            warnings.warn(f'Conversion to {np.int32} completed.')
        else:
            msg = (f'Supplied matrix is not of dtype integer, but it is {count_ad.X.dtype}. '
                   'Please supply an integer count matrix.')
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
        # logging.error(msg)
        raise ValueError(msg)

    # Strip 'chr' prefix
    # Only check if the first entry starts with "chr"
    if df["CHROM"].astype(str).str.contains(r"^chr").iloc[0]:
        df = df.assign(CHROM=df["CHROM"].astype(str).str.replace(r"^chr", "", regex=True))

    # Keep chr 1-22
    autosomes = [str(i) for i in range(1, 23)]
    df = df[df["CHROM"].astype('string').isin(autosomes)]    
    df["CHROM"] = df["CHROM"].astype('string')
    return df.copy()


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
        # logging.error(msg)
        raise ValueError(msg)

    # Reject integer-only matrices (raw counts)
    arr = lambdas_ref.to_numpy(copy=False)
    if np.all(arr == arr.astype(int)):
        msg = ("The reference expression matrix 'lambdas_ref' appears to "
               "contain only integer values. Please normalise raw counts "
               "with aggregate_counts() before calling this routine.")
        # logging.error(msg)
        raise ValueError(msg)

    # check that Gene IDs (row index) are unique
    if lambdas_ref.index.has_duplicates:
        msg = "Please remove duplicated genes in reference profile."
        # logging.error(msg)
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
    common_genes = set(gtf.loc[:,'gene']).intersection(set(count_mat.var_names)).intersection(set(lambdas_ref[lambdas_ref.mean(1) > min_lambda].index))
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

    par = np.ones(n_ref) / n_ref
    fit = scipy.optimize.minimize(
        fun=kl_to_min,
        x0=par,
        method='L-BFGS-B',
        tol=1e-6,
        options={'disp': verbose}
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
    - Optionally excludes genes overlapping the human HLA region on chromosome 6.
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
    gtf['gene_index'] = gtf.index
    bulk_obs = bulk_obs.merge(gtf, on='gene', how='left', sort=False)

    bulk_obs['CHROM'] = bulk_obs['CHROM'].astype('string') #was category
    bulk_obs['gene'] = bulk_obs['gene'].astype('string') #was category
    bulk_obs['logFC'] = np.log2(bulk_obs['lambda_obs']) - np.log2(bulk_obs['lambda_ref'])
    bulk_obs['lnFC'] = np.log(bulk_obs['lambda_obs']) - np.log(bulk_obs['lambda_ref'])

    # Filter out infinite log fold changes
    bulk_obs = bulk_obs[~bulk_obs['logFC'].isin([np.inf, -np.inf]) & ~bulk_obs['lnFC'].isin([np.inf, -np.inf])]
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
        p = np.exp(np.log(1 - np.exp(-2 * nu * distance)) - np.log(2))
        p = np.maximum(p, min_p)

    p[np.isnan(p)] = 0
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
        flat_list.extend(range(1, len(snps) + 1))
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

    df_allele = df_allele.sort_values(['CHROM', 'POS'], key=natsort.natsort_keygen())
    # df_allele['CHROM'] = df_allele['CHROM'].astype('string')

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
    
    # Fix observed counts, collapsing multiple SNPs per gene
    gene_collect = {}
    Y_obs_fix = []
    for chrom in bulk['CHROM'].unique():
        chrom_bulk = bulk[bulk['CHROM'] == chrom]
        for idx, row in chrom_bulk.iterrows():
            if pd.notna(row.gene) and row.gene in gene_collect:
                gene_collect[row.gene][row.snp_id] = row.Y_obs
                Y_obs_fix.append(np.nan)
            else:
                gene_collect[row.gene] = {row.snp_id: row.Y_obs}
                Y_obs_fix.append(row.Y_obs)
    bulk['Y_obs'] = Y_obs_fix
    
    # Calculate fold changes and normalize lambda_obs
    fc = np.exp(np.log(bulk['lambda_obs']) - np.log(bulk['lambda_ref']))
    bulk['lambda_obs'] = bulk['Y_obs'] / bulk['d_obs']
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
    bulk_ranges = pr.PyRanges(df=bulk) 
    
    # segs_consensus_ranges
    segs_consensus = segs_consensus.rename(columns={'CHROM':'Chromosome', 'seg_start':'Start', 'seg_end':'End'})
    segs_consensus_ranges = pr.PyRanges(df=segs_consensus) 
    
    # Find overlaps between bulk and segs_consensus
    overlaps = segs_consensus_ranges.join(bulk_ranges, how='left', slack=1)
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
    
    bulk = bulk.rename(columns={'Chromosome':'CHROM', 'Start':'POS'})
    
    # Drop unnecessary columns
    columns_to_exclude = ['sample']
    
    overlaps_df = overlaps_df.drop(columns=[col for col in columns_to_exclude if col in overlaps_df.columns])
    overlaps_df = overlaps_df.loc[:,['snp_id'] + [col for col in segs_consensus if col not in columns_to_exclude]]
    # Exclude overlapping columns from bulk except 'snp_id' and 'CHROM'
    exclude_from_bulk = [col for col in overlaps_df.columns if col not in ['snp_id', 'CHROM']]
    bulk = bulk.drop(columns=[col for col in exclude_from_bulk if col in bulk.columns])
    
    # # Merge bulk and overlaps_df
    bulk = bulk.merge(overlaps_df, on=['snp_id', 'CHROM'], how=how)    
    # # Assign 'seg' from 'seg_cons'
    bulk.loc[:,'seg'] = bulk.loc[:,'seg_cons']

    return bulk


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
    exp_bulk = exp_bulk[(exp_bulk.loc[:,'logFC'] > -5) & (exp_bulk.loc[:,'logFC'] < 5) | (exp_bulk.loc[:,'Y_obs'] == 0)]
    exp_bulk.loc[:,'mse'] = fit['mse']
    allele_bulk = get_allele_bulk(df_allele, nu=nu, min_depth=min_depth)
    bulk = combine_bulk(allele_bulk, exp_bulk, filter_hla=filter_hla, filter_segments=filter_segments)
    if np.unique(bulk.loc[:,'snp_id']).shape[0] != bulk.loc[:,'snp_id'].shape[0]:
        raise ValueError('Duplicated SNPs found, please check genotypes')
    
    # Filter out rows where lambda_ref is zero or gene is not NaN
    bulk = bulk[(bulk.loc[:, 'lambda_ref'] != 0) | (bulk.loc[:,'gene'].isna())]

    bulk.loc[:,'CHROM'] = np.where(bulk.loc[:, 'CHROM'] == 'X', 23, bulk.loc[:,'CHROM'])
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


def annot_segs(bulk, var = 'cnv_state'):
    # you need to reset index so you can pass portion of list (groups)
    bulk = bulk.copy().reset_index(drop=True) 
    boundary = []
    postfix = []
    for chrom in bulk.CHROM.unique():
        temp_sorted = bulk[bulk.loc[:, 'CHROM'] == chrom]
        boundary += [0]+[1 if temp_sorted.loc[:,var].iloc[i] != temp_sorted.loc[:,var].iloc[i - 1] else 0 for i in range(1,temp_sorted.shape[0])]
        current_postfix = generate_postfix(np.cumsum(boundary[temp_sorted.index[0]:temp_sorted.index[-1]+1]))
        postfix += [str(chrom)+i for i in current_postfix]
    # Natural sorting and cast to Categorical to avoid warnings
    postfix = pd.Series(postfix)
    unique_segs = natsorted(postfix.unique())
    bulk['seg'] = pd.Categorical(postfix, categories=unique_segs)
    
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

    bulk.loc[:, 'seg_start'] = seg_start
    bulk.loc[:, 'seg_end'] = seg_end
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
    _, pvalue = ttest_ind(x, y, equal_var=True, nan_policy='omit')
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
    min_depth: int = 0
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
    for chrom in tqdm.tqdm(chrom_unique):
        gene_unique = bulk[bulk.loc[:, 'CHROM'] == chrom].gene.unique()
        for gene in gene_unique:
            tmp_bulk = bulk[(bulk.loc[:,'CHROM'] == chrom) &
                            (bulk.loc[:,'gene'] == gene)]
            gene_snps = tmp_bulk[(~tmp_bulk.loc[:,'AD'].isna()) &
                                (tmp_bulk.loc[:,'DP'] > min_depth)].shape[0]
            Y_obs = tmp_bulk.loc[:,'Y_obs'].dropna().unique().astype(np.int32).sum()
            lambda_ref = tmp_bulk.loc[:,'lambda_ref'].dropna().unique().item()
            logFC = tmp_bulk.loc[:,'logFC'].dropna().unique().item()
            d_obs = tmp_bulk.loc[:,'d_obs'].dropna().unique().item()
            bulk_snps['gene_snps'].append(gene_snps)
            bulk_snps['Y_obs'].append(Y_obs)
            bulk_snps['lambda_ref'].append(lambda_ref)
            bulk_snps['logFC'].append(logFC)
            bulk_snps['d_obs'].append(np.int32(d_obs))
            bulk_snps['gene'].append(tmp_bulk.gene.values[0])
            bulk_snps['gene_start'].append(tmp_bulk.gene_start.astype(np.int32).values[0])
            bulk_snps['gene_end'].append(tmp_bulk.gene_end.astype(np.int32).values[0])
            bulk_snps['gene_length'].append(np.array(tmp_bulk.gene_end.values[0] - tmp_bulk.gene_start.values[0]).astype(np.int32))
            bulk_snps['CHROM'].append(chrom)
    
    bulk_snps_df = pd.DataFrame(bulk_snps)
    bulk_snps_df = bulk_snps_df[(bulk_snps_df.logFC < 8) & (bulk_snps_df.logFC > -8)]
    bulk_snps_df = bulk_snps_df.sort_values(['CHROM','gene_start']).reset_index(drop=True)
    
    fit = dist_prob.fit_lnpois(bulk_snps_df.Y_obs.values,
                     bulk_snps_df.lambda_ref.values,
                     bulk_snps_df.d_obs.unique())
    
    mu, sig = fit
    bulk_snps_df.gene_length = bulk_snps_df.gene_length.values.astype(np.int32)
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
    
    vtb = hmm.viterbi_loh(HMM)
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
    
    segs_loh = segs_loh.groupby(['CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state'], observed=True, sort=False).sum()
    segs_loh = segs_loh.reset_index().loc[:,['CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state', 'gene_snps', 'gene_length']]
    
    
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

    # Use PyRanges to reduce intervals
    gr = pr.PyRanges(chromosomes=segs_neu['CHROM'].astype("string"),
                     starts=segs_neu['seg_start'],
                     ends=segs_neu['seg_end'])
    gr_reduced = gr.merge()  # equivalent to reduce in GenomicRanges
    segs_neu_reduced = gr_reduced.as_df()

    segs_neu_reduced = segs_neu_reduced.rename(columns={'Chromosome':'CHROM','Start':'seg_start','End':'seg_end'})
    segs_neu_reduced['seg_length'] = segs_neu_reduced['seg_end'] - segs_neu_reduced['seg_start']

    return segs_neu_reduced


def fill_neu_segs(segs_consensus: pd.DataFrame, segs_neu: pd.DataFrame) -> pd.DataFrame:
    """
    Insert neutral intervals into a consensus set of segments and re-label.

    Computes gaps as "neutral - consensus" via PyRanges subtraction, appends
    these gaps to the consensus, sets missing "cnv_state" to "neu", and assigns
    a per-chromosome consensus segment ID 'seg_cons' using an external
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

    Notes
    -----
    - Sorting uses ``natsort.natsort_keygen()`` to produce human-friendly ordering.
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
    # combined.loc[:,'CHROM'] = combined.CHROM.astype('string')
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


