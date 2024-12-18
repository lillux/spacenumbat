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

from natsort import natsorted

from typing import Dict, List
from numpy.typing import NDArray

import warnings



## Prepare bulk data

def annotate_genes(df, gtf):
    # Rename some gtf columns. Needed for PyRanges.
    gtf = gtf.rename(columns={'gene_start':'Start', 'gene_end':'End', 'CHROM':'Chromosome'})
    # Take unique SNPs and rename columns for PyRanges
    snps = df[['snp_id', 'CHROM', 'POS']].drop_duplicates()
    snps = snps.rename(columns={'CHROM':'Chromosome', 'POS':'Start'})
    snps.loc[:,'End'] = snps.loc[:,'Start']
    # Create PyRanges for SNPs and GTF
    snps_pr = PyRanges(df=snps)
    gtf_pr = PyRanges(df=gtf)
    # Find overlaps between SNPs and genes, remove duplicates
    hits = snps_pr.join(gtf_pr).df.drop_duplicates('snp_id')
    # Add genes name to snps
    snps = snps.merge(hits.loc[:,['snp_id', 'gene']], on='snp_id', how='left')
    # Left join with SNPs
    df = df.loc[:, df.columns.difference(['gene', 'gene_start', 'gene_end'], sort=False)].merge(snps.loc[:,['snp_id','gene']], on='snp_id', how='left')

    return df
    

def filter_coverage(df_obs, df_allele, cell_coverage: int = 0, gene_coverage: int = 1):
    
    n_starting_genes = df_obs.shape[0]
    n_starting_cells = df_obs.shape[1]
    print(f'Starting with {n_starting_genes} genes and {n_starting_cells} cells.')

    # remove cells with less than min_coverage
    under_cov = df_obs.loc[:,df_obs.sum() < cell_coverage].columns
    if len(under_cov) > cell_coverage:
        df_obs = df_obs.loc[:, df_obs.columns.difference(list(under_cov))]
        print(f'Filtered {len(under_cov)} cells with less than {cell_coverage} coverage.')
        cov_mask = [i not in set(list(under_cov)) for i in df_allele.loc[:,'cell']]
        df_allele = df_allele[cov_mask]
        # Check if df_allele is empty?
    # remove genes with less than gene_coverage
    df_obs = df_obs.loc[df_obs.sum(1) > gene_coverage,:]

    
    print(f'Finishing with {df_obs.shape[0]} genes and {df_obs.shape[1]} cells.')
    return df_obs, df_allele
    

def check_anndata(count_ad:ad.AnnData, count_to_int=True, fix_names=True):
    # check type
    if isinstance(count_ad.X, np.ndarray):
        count_ad.X = scipy.sparse.csc_matrix(count_ad.X)
    elif not scipy.sparse.issparse(count_ad.X):
        msg = f'You passed an object of type {type(count_ad.X)}. count_mat should be a numpy array or a scipy sparse csc_matrix.'
        raise ValueError(msg)

    # check if data are integer counts
    if not np.issubdtype(count_ad.X, np.integer):
        if count_to_int:
            msg = f'The count matrix in the supplied count_ad is of type {count_ad.X.dtype}. Converting to {np.int32}.'
            warnings.warn(msg)
            count_ad.X = count_ad.X.astype(np.int32)
            warnings.warn(f'Conversion to {np.int32} completed.')
        else:
            msg = f'Supplied matrix is not of dtype integer, but it is {count_ad.X.dtype}. Please supply an integer count matrix.'
            raise ValueError(msg)
    # check if genes name are duplicated, and fix
    if count_ad.var_names.shape[0] != np.unique(count_ad.var_names).shape[0]:
        if fix_names:
            count_ad.var_names_make_unique()
        else:
            msg = f'Some genes name in var_names are not unique. Please make them unique or set the argument fix_names=True instead of {fix_names}.'
            raise ValueError(msg)
    
    return count_ad


def fit_ref_sse_ad(count_mat:ad.AnnData, 
                   lambdas_ref:pd.DataFrame, 
                   gtf:pd.DataFrame, 
                   min_lambda = 2e-6, 
                   verbose = False) -> Dict:
    '''
    Fit reference expression profile, using AnnData.

    Parameters
    ----------
    count_mat : ad.AnnData
        AnnData containing the sample counts.
    lambdas_ref : pd.DataFrame
        Reference expression profile.
    gtf : pd.DataFrame
        Reference genome annotation.
    min_lambda : TYPE, optional
        Minimal gene expression frequency.
        The default is 2e-6.
    verbose : TYPE, optional
        Verbosity of the optimization algorithm.
        The default is False.

    Returns
    -------
    Dict
        DESCRIPTION.

    '''
    count_mat = count_mat[:,np.array(count_mat.X.sum(0) > 0).flatten()]
    common_genes = set(gtf.loc[:,'gene']).intersection(set(count_mat.var_names)).intersection(set(lambdas_ref[lambdas_ref.mean(1) > min_lambda].index))

    common_genes = [i for i in gtf.loc[:,'gene'] if i in common_genes]
    count_mat = count_mat[:,common_genes]
    lambdas_obs = np.exp(np.log(np.array(count_mat.X.sum(0)).flatten()) - np.log(np.array(count_mat.X.sum(0)).flatten().sum()))
    lambdas_ref = lambdas_ref.loc[common_genes,:] # fix the case in which there is only 1 columns (remove : in [common_genes,:])
    
    n_ref = lambdas_ref.shape[1]
    
    def kl_to_min(x):
        return np.sum(np.power(np.log(lambdas_obs) - (np.log(np.matmul(lambdas_ref, x/np.sum(x)))),2))

    
    par = np.array([1/n_ref for i in range(n_ref)])
    fit = scipy.optimize.minimize(fun=kl_to_min,
                                  x0=par,
                                  method='L-BFGS-B',
                                  tol=1e-6,
                                  options={'disp': verbose})
    x = fit.x
    x = x/np.sum(x)
    lambdas_bar = np.matmul(lambdas_ref, x)
    lambdas_mse = fit.fun / len(lambdas_obs)
    return {'w':x, 'lambdas_bar':lambdas_bar, 'mse':lambdas_mse}


def filter_genes(count_mat:ad.AnnData, 
                 lambdas_bar:pd.Series,
                 gtf:pd.DataFrame,
                 filter_hla:bool=False, 
                 verbose:bool=False) -> List:
    gtf_df = pd.DataFrame(gtf)
    
    # Get genes to keep
    genes_keep = set(gtf_df['gene']).intersection(set(count_mat.var_names)).intersection(set(lambdas_bar.keys()))
    # Sort genes name following gtf ordering 
    genes_keep = [gene for gene in gtf_df.loc[:,'gene'] if gene in genes_keep]
    
    if filter_hla:
        # Gene in HLA region in human, based on chromosome and position.
        # This works on hg19 and hg38.
        genes_exclude = gtf_df[(gtf_df['CHROM'] == 6) & 
                               (gtf_df['gene_start'] < 33480577) & 
                               (gtf_df['gene_end'] > 28510120)]['gene'].tolist()
        
        # Exclude genes to keep
        genes_keep = [gene for gene in genes_keep if gene not in genes_exclude]
    
    # Filter count matrix, lambdas_ref, and compute lambdas_obs
    count_mat_filtered = count_mat[:,genes_keep]
    lambdas_bar_filtered = lambdas_bar[genes_keep]
    lambdas_obs = pd.Series(np.array(count_mat_filtered.X.sum(0) / count_mat_filtered.X.sum()).ravel(), index=count_mat_filtered.var.index)

    # Thresholds
    min_both = 2
    mean_lambdas_bar = lambdas_bar_filtered[lambdas_bar_filtered > 0].values.mean()
    mean_lambdas_obs = lambdas_obs[lambdas_obs > 0].values.mean(dtype=np.float64)
    # print(mean_lambdas_bar, mean_lambdas_obs)
    
    # # Genes to retain
    mut_expressed = pd.DataFrame(((lambdas_bar_filtered.values.flatten() * 1e6 > min_both) & (lambdas_obs.values * 1e6 > min_both) | 
                    (lambdas_bar_filtered.values.flatten() > mean_lambdas_bar) | 
                    (lambdas_obs.values > mean_lambdas_obs)) & (lambdas_bar_filtered.values.flatten() > 0))
    mut_expressed.index = lambdas_bar_filtered.index
    
    retained = [gene for gene, expressed in zip(genes_keep, mut_expressed.values) if expressed]

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

def switch_prob(distance:np.Array,
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


def combine_bulk(allele_bulk, exp_bulk, filter_hla=False):
    bulk = pd.merge(allele_bulk, exp_bulk, how='outer', on=['CHROM','gene'])
    bulk.loc[:,'snp_id'] = np.where(bulk['snp_id'].isna(), bulk['gene'], bulk['snp_id'])
    bulk.loc[:,'gene'] = pd.Categorical(bulk.loc[:,'gene'], categories=exp_bulk.loc[:,'gene'])
    bulk.loc[:,'POS'] = np.where(bulk.loc[:,'POS'].isna(), bulk.loc[:,'gene_start'], bulk.loc[:,'POS'])
    bulk.loc[:,'p_s'] = np.where(bulk.loc[:,'p_s'].isna(), 0, bulk.loc[:,'p_s'])
    bulk = bulk.sort_values(by=['CHROM','POS'])
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
    
    bulk = bulk.sort_values(by=['CHROM','POS']).reset_index(drop=True)
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
    # Reset index to get 'marker_index' and 'seg_index'
    bulk = bulk.reset_index(drop=True)
    bulk['marker_index'] = bulk.index
    
    # # Create seg_index in segs_consensus
    segs_consensus = segs_consensus.reset_index(drop=True)
    segs_consensus['seg_index'] = segs_consensus.index
    
    # Create PyRanges objects for bulk and segs_consensus
    bulk_ranges = pr.PyRanges(pd.DataFrame({
        'Chromosome': bulk['CHROM'],
        'Start': bulk['POS'],
        'End': bulk['POS'],
        'snp_id': bulk['snp_id']
    }))
    
    segs_consensus_ranges = pr.PyRanges(pd.DataFrame({
        'Chromosome': segs_consensus['CHROM'],
        'Start': segs_consensus['seg_start'],
        'End': segs_consensus['seg_end'],
        'seg_index': segs_consensus['seg_index']
    }))
    
    
    # Find overlaps between bulk and segs_consensus
    overlaps = bulk_ranges.join(segs_consensus_ranges, how='right', slack=1)
    overlaps_df = overlaps.df
    
    # Merge overlaps with bulk to get marker_index and snp_id
    overlaps_df = overlaps_df.merge(
        bulk[['marker_index', 'snp_id']], on='snp_id', how='left'
    )
    
    # Merge overlaps with segs_consensus to get seg_cons and other info
    overlaps_df = overlaps_df.merge(
        segs_consensus[['seg_index', 'seg_cons']], on='seg_index', how='left'
    )
    
    # Remove duplicates of snp_id, keeping the first occurrence
    overlaps_df = overlaps_df.drop_duplicates(subset='snp_id')
    overlaps_df = overlaps_df.rename(columns={'Chromosome':'CHROM','Start_b':'seg_start', 'End_b':'seg_end'})
    overlaps_df = overlaps_df.drop(['Start', 'End'], axis=1)
    
    cat_bulk = pd.CategoricalDtype(categories=bulk.CHROM.unique())
    
    bulk.CHROM = bulk.CHROM.astype('int')
    bulk.CHROM = bulk.CHROM.astype(cat_bulk)
    
    overlaps_df.CHROM = overlaps_df.CHROM.astype('int')
    overlaps_df.CHROM = overlaps_df.CHROM.astype(cat_bulk)
    overlaps_df.loc[:,'loh'] = True
    
    # Drop unnecessary columns
    columns_to_exclude = ['sample', 'marker_index', 'seg_index']
    overlaps_df = overlaps_df.drop(columns=[col for col in columns_to_exclude if col in overlaps_df.columns])
    
    # Exclude overlapping columns from bulk except 'snp_id' and 'CHROM'
    columns_to_exclude_from_bulk = [col for col in overlaps_df.columns if col not in ['snp_id', 'CHROM']]
    bulk = bulk.drop(columns=[col for col in columns_to_exclude_from_bulk if col in bulk.columns])
    
    # Merge bulk and overlaps_df
    bulk = bulk.merge(overlaps_df, on=['snp_id', 'CHROM'], how=how)
    
    # Assign 'seg' from 'seg_cons'
    bulk.loc[:,'seg'] = bulk.loc[:,'seg_cons']
    
    # Factor 'seg' using natsorted order
    unique_segs = natsorted(bulk['seg'].dropna().unique())
    bulk['seg'] = pd.Categorical(bulk['seg'], categories=unique_segs, ordered=True)
    
    bulk.loc[:,'loh'] = bulk.loc[:,'loh'].fillna(False).astype(bool)
    
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
    fit = fit_ref_sse_ad(count_mat, lambdas_ref, gtf, verbose=verbose)
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
    bulk = bulk.sort_values(by=['CHROM','POS'])
    bulk = bulk.reset_index(drop=True)

    # Annotate clonal LOH regions
    if segs_loh is None:
        bulk.loc[:,'loh'] = False
    else:
        # Annotate consensus segments
        bulk = annot_consensus(bulk, segs_loh, join_mode='left')
        # Set 'loh' to False where it's NaN
        bulk.loc[:,'loh'] = bulk.loc[:,'loh'].fillna(False).astype(bool)
    
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
    
    bulk.loc[:,'boundary'] = boundary
    bulk.loc[:,'seg'] = postfix
    bulk.seg = bulk.seg.astype('category')
    
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