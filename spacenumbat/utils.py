#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 11:35:09 2024

@author: carlino.calogero
"""
import string

import numpy as np
import pandas as pd

import anndata as ad


import pyranges as pr
from pyranges import PyRanges

import scipy
from scipy.stats import ttest_ind

import natsort
from natsort import natsorted

from typing import Dict, List, Union, Sequence, Any, Optional
from numpy.typing import NDArray

from collections import Counter

import warnings


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
        genes_exclude = gtf_df[(gtf_df['CHROM'] == 6) &
                               (gtf_df['gene_start'] < 33480577) &
                               (gtf_df['gene_end'] > 28510120)]['gene'].tolist()
        genes_keep = [gene for gene in genes_keep if gene not in genes_exclude]

    if filter_segments is not None and not filter_segments.empty:
        genes_exclude = []
        for _, row in filter_segments.iterrows():
            overlapping = gtf_df[(gtf_df['CHROM'].astype(str) == str(row.CHROM)) &
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
    mean_lambdas_obs = lambdas_obs[lambdas_obs > 0].mean(dtype=np.float64)

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
        print(f'number of genes left: {len(retained)}')

    return retained


def get_exp_bulk(count_mat:ad.AnnData, 
                 lambdas_bar:pd.Series,
                 gtf:pd.DataFrame,
                 verbose:bool=False,
                 filter_hla:bool=False) -> pd.DataFrame:
    depth_obs_before_filt = count_mat.X.sum()
    mut_expressed = filter_genes(count_mat, lambdas_bar, gtf, filter_hla=filter_hla)
    count_mat = count_mat[:,mut_expressed]
    # depth_obs_after_filt = count_mat.sum().sum()
    lambdas_bar = lambdas_bar.loc[mut_expressed]
    
    # dataframe set-up
    bulk_obs = pd.DataFrame({'Y_obs':count_mat.X.sum(0).T.A.ravel()}, index=count_mat.var_names)
    bulk_obs = bulk_obs.rename_axis('gene').reset_index()
    # library depth before gene filtering is used to normalize counts.
    bulk_obs.loc[:,'d_obs'] = depth_obs_before_filt
    bulk_obs.loc[:,'lambda_obs'] = np.array(bulk_obs.loc[:,'Y_obs'] / bulk_obs.loc[:,'d_obs'], dtype=np.float64)
    bulk_obs.loc[:,'lambda_ref'] = lambdas_bar[bulk_obs.loc[:,'gene']].values.astype(np.float64)
    gtf.loc[:,'gene_index'] = gtf.index
    bulk_obs = bulk_obs.merge(gtf, on='gene', how='left', sort=False)
    bulk_obs.CHROM = bulk_obs.CHROM.astype('category')
    bulk_obs.gene = bulk_obs.gene.astype('category')
    
    bulk_obs.loc[:, 'logFC'] = np.log2(bulk_obs.loc[:,'lambda_obs'].values) - np.log2(bulk_obs.loc[:, 'lambda_ref'].values)
    bulk_obs.loc[:, 'lnFC'] = np.log(bulk_obs.loc[:,'lambda_obs'].values) - np.log(bulk_obs.loc[:,'lambda_ref'].values)
    # add control for infinity
    bulk_obs = bulk_obs[(~np.isinf(bulk_obs.logFC)) & (~np.isinf(bulk_obs.lnFC))] # here we filter infinity. Original implementation assign NA

    return bulk_obs


def get_inter_cm(cM:pd.Series) -> NDArray:
    if len(cM) <= 1:
        return np.nan
    else:
        return np.hstack([np.nan, np.array(cM[1:].values - cM[:-1].values)])

def switch_prob(distance:NDArray,
                nu:float=1, 
                min_p:float=1e-10) -> NDArray:
    if nu == 0:
        p = np.zeros(len(distance))
    else:
        p = np.exp(np.log(1 - np.exp(-2*nu*distance)) - np.log(2))
        p = np.maximum(p, min_p)

    p[np.isnan(p)] = 0
    return p


def get_allele_bulk(df_allele:pd.DataFrame, 
                    nu:float=1, 
                    min_depth:int=0) -> pd.DataFrame:
    
    df_allele = df_allele.loc[:, ['snp_id', 'CHROM', 'POS', 'cM', 'REF', 'ALT', 'AD', 'DP', 'GT', 'gene']]
    df_allele = df_allele[[i in {'1|0', '0|1'} for i in df_allele.GT]]
    df_allele = df_allele[~np.isnan(df_allele.cM)]    
    df_allele = df_allele.groupby(['snp_id', 'CHROM', 'POS', 'cM', 'REF', 'ALT', 'GT', 'gene'], sort=False, as_index=False, dropna=False).sum(['AD', 'DP'])
    df_allele.loc[:,'AR'] = df_allele.AD / df_allele.DP
    df_allele = df_allele.sort_values(['CHROM', 'POS'])
    
    flat_list = []
    for chrom in df_allele.CHROM.unique():
        for idx, snp in enumerate(df_allele[df_allele.CHROM == chrom].snp_id, start=1):
            flat_list.append(idx)
            
    df_allele.loc[:,'snp_index'] = flat_list
    df_allele = df_allele[df_allele.DP >= min_depth]
    
    pBAF = []
    pAD = []
    for row, data in df_allele.iterrows():
        if data.GT == '1|0':
            pBAF.append(data.AR)
            pAD.append(data.AD)
        else:
            pBAF.append(1-data.AR)
            pAD.append(data.DP - data.AD)
    df_allele.loc[:,'pBAF'] = pBAF
    df_allele.loc[:,'pAD'] = pAD
    #
    df_allele = df_allele.sort_values(['CHROM', 'POS'])
    df_allele.CHROM = df_allele.CHROM.astype('category')
    
    inter_snp_cm = np.zeros(df_allele.shape[0])
    start_idx = 0
    for chrom in df_allele.CHROM.unique():
        df_allele_chrom = df_allele.loc[df_allele.CHROM == chrom]
        end_idx = start_idx + df_allele_chrom.shape[0]
        inter_snp_cm[start_idx:end_idx] = get_inter_cm(df_allele_chrom.cM)
        start_idx += df_allele_chrom.shape[0]
    
    df_allele.loc[:,'inter_snp_cm'] = inter_snp_cm
    
    df_allele.loc[:,'p_s'] = switch_prob(df_allele.inter_snp_cm)
    df_allele.loc[:,'gene'] = [i if isinstance(i, str) else np.nan for i in df_allele.gene]

    return df_allele


def combine_bulk(allele_bulk, exp_bulk, filter_hla=True):
    bulk = pd.merge(allele_bulk, exp_bulk, how='outer', on=['CHROM','gene'])
    bulk.loc[:,'snp_id'] = np.where(bulk['snp_id'].isna(), bulk['gene'], bulk['snp_id'])
    bulk.loc[:,'gene'] = pd.Categorical(bulk.loc[:,'gene'], categories=exp_bulk.loc[:,'gene'])
    bulk.loc[:,'POS'] = np.where(bulk.loc[:,'POS'].isna(), bulk.loc[:,'gene_start'], bulk.loc[:,'POS'])
    bulk.loc[:,'p_s'] = np.where(bulk.loc[:,'p_s'].isna(), 0, bulk.loc[:,'p_s'])
    bulk = bulk.sort_values(by=['CHROM','POS'], key=natsort.natsort_keygen())
    # filter HLA
    if filter_hla:
        to_filter = bulk[(bulk.loc[:,'CHROM'] == 6) & (bulk.loc[:,'POS'] > 28510120) & (bulk.loc[:,'POS'] < 33480577)].index
        bulk = bulk.loc[bulk.index.difference(to_filter),:]
        
    gene_collect = {}
    Y_obs_fix = []
    for chrom in bulk.CHROM.unique():
        running_chrom_bulk = bulk[bulk.loc[:,'CHROM'] == chrom]
        for idx, row, in running_chrom_bulk.iterrows():
            if row.gene in gene_collect and (row.gene is not np.nan):
                gene_collect[row.gene][row.snp_id] = row.Y_obs
                Y_obs_fix.append(np.nan)
            else:
                gene_collect[row.gene] = {}
                gene_collect[row.gene][row.snp_id] = row.Y_obs
                Y_obs_fix.append(row.Y_obs)
    bulk.loc[:, 'Y_obs'] = Y_obs_fix
    fc = np.exp(np.log(bulk.loc[:,'lambda_obs']) - np.log(bulk.loc[:,'lambda_ref']))
    bulk.loc[:,'lambda_obs'] = bulk.loc[:,'Y_obs'] / bulk.loc[:,'d_obs']
    bulk.loc[:,'logFC'] = np.log2(fc)
    bulk.loc[:,'logFC'] = np.where(np.isinf(bulk.loc[:,'logFC']), np.nan, bulk.loc[:,'logFC'])
    bulk.loc[:,'lnFC'] = np.log(fc)
    bulk.loc[:,'lnFC'] = np.where(np.isinf(bulk.loc[:,'lnFC']), np.nan, bulk.loc[:,'lnFC'])
    
    bulk = bulk.sort_values(by=['CHROM','POS'], key=natsort.natsort_keygen()).reset_index(drop=True)
    # assign index to snps
    snp_index = []
    for chrom in bulk.CHROM.unique():
        current_snp_num = bulk[bulk.CHROM == chrom].shape[0]
        snp_index += [i for i in range(current_snp_num)]
    bulk.loc[:,'snp_index'] = snp_index
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
    # overlaps = segs_consensus_ranges.join(bulk_ranges, how='right', slack=1)
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
    cat_bulk = pd.CategoricalDtype(categories=bulk.CHROM.unique())
    bulk.CHROM = bulk.CHROM.astype('int')
    bulk.CHROM = bulk.CHROM.astype(cat_bulk)
    
    overlaps_df.CHROM = overlaps_df.CHROM.astype('int')
    overlaps_df.CHROM = overlaps_df.CHROM.astype('category')
    
    # # # Drop unnecessary columns
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
    # Factor 'seg' using natsorted order
    unique_segs = natsorted(bulk['seg'].dropna().unique())
    bulk['seg'] = pd.Categorical(bulk['seg'], categories=unique_segs)

    return bulk


def get_bulk(count_mat,
             lambdas_ref,
             df_allele,
             gtf,
             subset = None,
             min_depth = 0,
             nu = 1,
             segs_loh = None,
             verbose = True,
             disp = False,
             filter_hla = True):

    # YOU NEED TO DO ***EXPLICIT*** COPY OF THE ANNDATA WHEN YOU WRITE ON IT!
    count_mat = check_anndata(count_mat.copy())
    if subset is not None: 
        if not set(subset).issubset(set(count_mat.obs_names)):
            raise KeyError('All the requested cell barcodes must be present in count_mat')
        else:
            count_mat = count_mat[subset]
            df_allele_subset_mask = [i in subset for i in df_allele.cell]
            df_allele = df_allele[df_allele_subset_mask]
    fit = fit_ref_sse_ad(count_mat, lambdas_ref, gtf, verbose=disp)
    exp_bulk = get_exp_bulk(count_mat, fit['lambdas_bar'], gtf, verbose=verbose, filter_hla=filter_hla)
    exp_bulk = exp_bulk[(exp_bulk.loc[:,'logFC'] > -5) & (exp_bulk.loc[:,'logFC'] < 5) | (exp_bulk.loc[:,'Y_obs'] == 0)]
    exp_bulk.loc[:,'mse'] = fit['mse']
    allele_bulk = get_allele_bulk(df_allele, nu=nu, min_depth=min_depth)
    bulk = combine_bulk(allele_bulk, exp_bulk, filter_hla=filter_hla)
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
        # bulk.loc[:,'loh'] = bulk.loc[:,'loh'].fillna(False).astype(bool)
        bulk.loc[:,'loh'] = bulk.loc[:,'loh'].fillna(0).astype(bool)
    
    return bulk


## Fit snp rate on loh calling

def fit_snp_rate(gene_snps, gene_length):

    # Define the objective function to minimize
    def objective(params):
        v = params[0]
        sig = params[1]
        mu = v * gene_length / 1e6
        log_likelihood = np.sum(scipy.stats.nbinom.logpmf(gene_snps, sig, sig / (mu + sig)))
        return -log_likelihood

    # Initial parameters
    initial_params = [10, 1]

    # Constraints on the parameters (equivalent to lower bounds in R optim)
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


def t_test_pval(x, y):
    """
    T-test wrapper, handles error for insufficient observations.
    Returns 1 if either x or y doesn't have more than one element.
    Otherwise returns the p-value of a two-sample t-test.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Check length conditions
    if x.size <= 1 or y.size <= 1:
        return 1.0
    # Perform two-sample t-test (assuming equal var)
    _, pvalue = ttest_ind(x, y, equal_var=True, nan_policy='omit')
    return pvalue


def simes_p(p_vals, n_dim):
    """
    Calculate simes' p.
    p.vals: array-like of p-values
    n_dim: scalar integer
    """
    p_vals = np.asarray(p_vals)
    sorted_p = np.sort(p_vals)
    indices = np.arange(len(sorted_p))
    return n_dim * np.min(sorted_p / indices)


def Modes(x):
    """
    Get the modes of a vector.
    Returns a list of the most frequent values.
    """
    x = np.asarray(x)
    # Count occurrences using Counter
    c = Counter(x)
    # Find max frequency
    max_freq = max(c.values())
    # Return all elements that have this frequency
    return [k for k, v in c.items() if v == max_freq]