#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 00:33:04 2025

@author: lillux
"""

from typing import Any, Dict, Union

import pandas as pd
import numpy as np
from joblib import cpu_count, Parallel, delayed

import networkx as nx
import pyranges as pr
import natsort

from . import utils


from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("Test operations")

def run_group_hmms(
    bulks, t=1e-4, gamma=20, alpha=1e-4, min_genes=10, nu=1,
    common_diploid=True, diploid_chroms=None, allele_only=False, retest=True, run_hmm=True,
    exclude_neu=True, ncores=1, verbose=False, debug=False
    ):
    """
    Run multiple HMMs.

    Parameters:
    ----------
    bulks : (pd.DataFrame): Pseudobulk profiles.
    t (float): Transition probability.
    gamma (float): Dispersion parameter for the Beta-Binomial allele model.
    alpha (float): P-value cutoff to determine segment clusters in find_diploid.
    min_genes (int): Minimum number of genes.
    nu (float): Parameter nu.
    common_diploid (bool): Whether to find common diploid regions between pseudobulks.
    diploid_chroms (list or None): Known diploid chromosomes to use as baseline.
    allele_only (bool): Whether to use only allele data to run HMM.
    retest (bool): Whether to retest CNVs.
    run_hmm (bool): Whether to run HMM segments or just retest.
    exclude_neu (bool): Whether to exclude neutral segments.
    ncores (int): Number of cores.
    verbose (bool): Verbosity.
    debug (bool): Debug mode.

    Returns:
    ----------
        pd.DataFrame: Resulting data after running HMMs.
    """
    
    # Drop samples with no allele data
    bulks = bulks.groupby('sample', observed=True, sort=False).filter(lambda x: x['DP'].notna().sum() > 0).copy()

    if bulks.shape[0] == 0:
        return pd.DataFrame()

    n_groups = bulks['sample'].nunique()

    if verbose:
        print(f'Running HMMs on {n_groups} cell groups...')

    # Determine whether to find diploid regions
    if not run_hmm:
        find_diploid = False
    elif common_diploid and diploid_chroms is None:
        bulks = utils.find_common_diploid(bulks, gamma=gamma, alpha=alpha, ncores=ncores)
        find_diploid = False
    else:
        find_diploid = True

    # Parallel calls to 'analyze_bulk' on each sample group
    def analyze(bulk: pd.DataFrame):
        # try-except to catch errors
        try:
            return utils.analyze_bulk(
                bulk,
                t=t,
                gamma=gamma,
                nu=nu,
                find_diploid=find_diploid,
                run_hmm=run_hmm,
                allele_only=allele_only,
                diploid_chroms=diploid_chroms,
                min_genes=min_genes,
                retest=retest,
                verbose=verbose,
                exclude_neu=exclude_neu
            )
        except Exception as e:
            return e  # pass the exception back
    bulk_groups = bulks.groupby('sample', observed=True, sort=False)
    ncores = np.max([1,np.min((len(bulk_groups), cpu_count(), ncores))])
    print(f'Running bulk analysis on {ncores} core')

    results = Parallel(n_jobs=ncores)(
        delayed(analyze)(df_group) for sample_val, df_group in bulk_groups
    )

    # Check for errors
    for r in results:
        if isinstance(r, Exception):
            log.error(str(r))
            raise r 

    bulks = pd.concat(results, axis=0).reset_index(drop=True)
    bulks_groups = bulks.groupby(['seg', 'sample'], observed=True, sort=False) 
    for k, group in bulks_groups:
        bulks.loc[group.index, 'seg_start_index'] = group.snp_index.min()
        bulks.loc[group.index, 'seg_end_index'] = group.snp_index.max()

    return bulks


def resolve_cnvs(segs_all: pd.DataFrame, min_overlap: float = 0.5, debug: bool = False) -> Union[pd.DataFrame, Dict[str, Any]]:
    """
    Resolve consensus CNV segments across samples by:
    1) building an undirected overlap graph of segments,
    2) finding connected components, and
    3) selecting, per component, the sample with the strongest evidence.

    The function assigns a unique vertex id to each row, computes all pairwise
    overlaps via PyRanges self-join, filters edges by an overlap fraction
    threshold, builds a NetworkX graph, finds connected components, and for each
    component selects the row from the sample with the largest LLR_sample
    (defined as the per-(component, sample) maximum of LLR_x + LLR_y).

    Parameters
    ----------
    segs_all
        Input segments table. Expected columns (no explicit validation here):
        - CHROM
        - seg_start_index, seg_end_index
        - LLR_x, LLR_y
        - sample
        Optional columns are passed through (for example seg_start, used for final sorting).
        The function adds:
        - vertex (unique integer id)
        - component (connected component id)
        - sum_LLR and LLR_sample (intermediate scores)
    min_overlap
        Minimum overlap fraction to keep an edge. An edge is kept if the overlap
        fraction is at least min_overlap for either of the two intervals in the pair.
        Default is 0.5.
    debug
        If True, return a dictionary with both the graph and the consensus DataFrame.
        If False, return only the consensus DataFrame. Default is False.

    Returns
    -------
    pandas.DataFrame or dict
        If debug is False: a DataFrame with one consensus row per connected component,
        sorted by CHROM and then seg_start when available.
        If debug is True: a dict with keys:
        - 'G': the NetworkX graph
        - 'segs_consensus': the consensus DataFrame

    Notes
    -----
    - Overlap rule: the current filter keeps an edge if either interval meets the
      threshold (more permissive). If you intended both intervals to pass, adjust
      the filter accordingly.
    - Zero-length segments (end == start) can cause division by zero when computing
      overlap fractions; ensure non-zero lengths upstream or guard for them here.
    - Natural sorting is applied using natsort for multi-column sorts; be aware it
      may coerce values to strings when used across mixed dtypes.

    Examples
    --------
    >>> out = resolve_cnvs(segs_all_df, min_overlap=0.6, debug=True)
    >>> G = out['G']
    >>> segs_consensus = out['segs_consensus']
    >>> segs_consensus.head()
    """
    if segs_all.shape[0] == 0:
        return segs_all
    
    # Create 'vertex' column
    segs_all = segs_all.copy().reset_index(drop=True)
    segs_all.loc[:,'vertex'] = np.arange(0, len(segs_all))
    
    # Build PyRanges object
    # store the 'vertex' in the 'Name' field
    pr_input = pr.PyRanges(
        pd.DataFrame({
            'Chromosome': segs_all['CHROM'],
            'Start': segs_all['seg_start_index'],
            'End': segs_all['seg_end_index'],
            'Name': segs_all['vertex']
        }))
    
    # find all self-overlaps
    overlaps = pr_input.join(pr_input, report_overlap=True) 
    
    # Rename cols
    df_ov = overlaps.as_df()
    df_ov = df_ov.rename(columns={'Name': 'from','Name_b': 'to','Overlap':'len_overlap'})
    df_ov = df_ov.loc[:,['from', 'to', 'len_overlap']]
    # filter 'from != to'. FILTER SNPs
    df_ov = df_ov[df_ov['from'] != df_ov['to']].copy()
    df_ov['vp'] = df_ov.apply(lambda row: f"{min(row['from'], row['to'])},{max(row['from'], row['to'])}", axis=1)
    df_ov = df_ov.drop_duplicates(subset='vp')
    
    segs_all_for_merge = segs_all[['vertex','seg_start_index','seg_end_index']].rename(
        columns={'vertex':'from','seg_start_index':'start_x','seg_end_index':'end_x'})
    df_ov = df_ov.merge(segs_all_for_merge, on='from', how='left')
    
    segs_all_for_merge2 = segs_all[['vertex','seg_start_index','seg_end_index']].rename(
        columns={'vertex':'to','seg_start_index':'start_y','seg_end_index':'end_y'})
    df_ov = df_ov.merge(segs_all_for_merge2, on='to', how='left')
    
    df_ov['len_x'] = df_ov['end_x'] - df_ov['start_x']
    df_ov['len_y'] = df_ov['end_y'] - df_ov['start_y']
    df_ov['frac_overlap_x'] = df_ov['len_overlap'] / df_ov['len_x']
    df_ov['frac_overlap_y'] = df_ov['len_overlap'] / df_ov['len_y']
    # keep edges above min_overlap
    df_ov = df_ov[~((df_ov['frac_overlap_x']<min_overlap) & (df_ov['frac_overlap_y']<min_overlap))].copy()
    
    # Build an undirected graph using networkx
    G = nx.Graph()
    for idx, row in segs_all.iterrows():
        G.add_node(row['vertex'])
    for idx, row in df_ov.iterrows():
        G.add_edge(row['from'], row['to'])
    
    # Find connected components
    comps = list(nx.connected_components(G))
    # build dict
    vertex_to_comp = {}
    for i, comp_set in enumerate(comps):
        for v in comp_set:
            vertex_to_comp[v] = i
    
    segs_all.loc[:,'component'] = segs_all['vertex'].map(vertex_to_comp)
    segs_all = segs_all.copy()
    segs_all.loc[:,'sum_LLR'] = segs_all['LLR_x'] + segs_all['LLR_y']
    
    # compute max sum_LLR per (component, sample)
    grp = segs_all.groupby(['component','sample'],observed=True, sort=False)['sum_LLR'].transform('max')
    segs_all['LLR_sample'] = grp
    # for each component, find the sample with largest LLR_sample, and keep subset.
    segs_all = segs_all.sort_values(by=['CHROM','component','LLR_sample'],
                                    ascending=[True, True, False],
                                    key=natsort.natsort_keygen())
    
    segs_consensus_group = segs_all.groupby('component', as_index=False, group_keys=False, sort=False, observed=True)
    max_llr_idx = []
    for k, group in segs_consensus_group:
        max_llr_idx.append(group.LLR_sample.idxmax())
    segs_consensus = segs_all.loc[max_llr_idx,:]
    segs_consensus = segs_consensus.sort_values(by=['CHROM','seg_start'], key=natsort.natsort_keygen())
    segs_consensus = segs_consensus.drop(['vertex', 'sum_LLR'], axis=1)

    if debug:
        return {'G': G, 'segs_consensus': segs_consensus}
    else:
        return segs_consensus
    
    
def get_segs_consensus(
    bulks: pd.DataFrame,
    min_LLR: float = 5,
    min_overlap: float = 0.45,
    retest: bool = True
    ) -> pd.DataFrame:
    """
    Build consensus CNV segments across samples.

    This function merges per-sample CNV segments, resolves overlapping
    aberrant segments across samples using a graph-based approach, optionally
    constructs additional candidate intervals to retest between aberrations,
    merges neutral segments, and returns a final consensus set.

    The core steps are:
    1) Ensure a sample column exists; compute per-segment genomic start and end.
    2) Force segments with LLR < min_LLR to neutral.
    3) Resolve non-neutral segments across samples with resolve_cnvs, using an
       overlap threshold of min_overlap.
    4) If retest is True, derive inter-aberration intervals to re-evaluate.
    5) Union and reduce all neutral segments across samples.
    6) If all segments are neutral, return the neutral set with segment IDs.
       Otherwise, combine resolved aberrant segments with retest intervals and
       fill in neutral segments via fill_neu_segs.

    Parameters
    ----------
    bulks : pandas.DataFrame
        Input table of per-marker or per-bin annotations that can be aggregated
        into segments. Expected to contain at least:
        - sample
        - CHROM
        - seg
        - POS
        - seg_start_index, seg_end_index
        - cnv_state
        - LLR, LLR_x, LLR_y
        Additional columns are preserved if present and used in downstream logic
        (for example theta_mle, phi_mle).
    min_LLR : float, default 5
        Segments with LLR below this threshold are treated as neutral.
    min_overlap : float, default 0.45
        Minimum fractional overlap used by resolve_cnvs when building the
        overlap graph among non-neutral segments.
    retest : bool, default True
        If True, generates candidate intervals between non-neutral regions
        for potential retesting.

    Returns
    -------
    pandas.DataFrame
        Final consensus set of segments. If all segments are neutral, returns
        only the neutral segments with assigned seg labels; otherwise returns
        the union of resolved aberrant segments, retest intervals (if any),
        and neutral segments filled via fill_neu_segs.

    Examples
    --------
    >>> consensus = get_segs_consensus(bulks_df, min_LLR=6, min_overlap=0.5, retest=True)
    >>> consensus.head()
    """
    
    bulks = bulks.copy()
    
    if 'sample' not in bulks.columns:
        bulks['sample'] = 0
    
    info_cols = [
        'sample','CHROM','seg','cnv_state','cnv_state_post','seg_start','seg_end',
        'seg_start_index','seg_end_index','theta_mle','theta_sigma','phi_mle','phi_sigma',
        'p_loh','p_del','p_amp','p_bamp','p_bdel','LLR','LLR_y','LLR_x','n_genes','n_snps'
    ]
    
    # Build segs_all
    groupcols = ['sample','seg','CHROM']
    def seg_start_end_aggregator(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['seg_start'] = df['POS'].min()
        df['seg_end']   = df['POS'].max()
        return df
    bulks = bulks.groupby(groupcols, group_keys=False, observed=True, sort=False)[bulks.columns].apply(seg_start_end_aggregator).reset_index(drop=True)
    bulks = bulks[bulks['seg_start'] != bulks['seg_end']]
    
    segs_all = bulks[info_cols].drop_duplicates().copy()
    segs_all.loc[(segs_all['LLR'].isna()) | (segs_all['LLR']<min_LLR), 'cnv_state'] = 'neu'
    segs_star = segs_all[segs_all['cnv_state']!='neu'].copy()
    segs_star = resolve_cnvs(segs_star, min_overlap=min_overlap, debug=False)
    
    if retest:
        segs_cnv = segs_all[segs_all['cnv_state']!='neu'].copy()
        # build PyRanges from segs_cnv
        pr_cnv = pr.PyRanges(
            pd.DataFrame({
                'Chromosome': segs_cnv['CHROM'],
                'Start': segs_cnv['seg_start'],
                'End': segs_cnv['seg_end']
            })
        ).merge()
    
        # build PyRanges from segs_star
        pr_star = pr.PyRanges(
            pd.DataFrame({
                'Chromosome': segs_star['CHROM'],
                'Start': segs_star['seg_start'],
                'End': segs_star['seg_end']
            })
        ).merge()
    
        # find segments in between CNVs regions
        pr_retest = pr_cnv.subtract(pr_star)
        df_retest = pr_retest.as_df()
        df_retest = df_retest[df_retest['End']> df_retest['Start']]
        # add cnv_state 'retest'
        df_retest['cnv_state'] = 'retest'
        df_retest['cnv_state_post'] = 'retest'
        df_retest = df_retest.rename(columns={'Chromosome':'CHROM','Start':'seg_start','End':'seg_end'})
    else:
        df_retest = pd.DataFrame()
    
    # union of neutral segments
    segs_neu_input = segs_all[segs_all['cnv_state']=='neu']
    pr_neu = pr.PyRanges(
        pd.DataFrame({
            'Chromosome': segs_neu_input['CHROM'],
            'Start': segs_neu_input['seg_start'],
            'End': segs_neu_input['seg_end']
        })
    ).merge()
    df_neu = pr_neu.as_df()
    df_neu = df_neu.rename(columns={'Chromosome':'CHROM','Start':'seg_start','End':'seg_end'})
    df_neu['seg_length'] = df_neu['seg_end']-df_neu['seg_start']
    
    # if all segs_all['cnv_state'] == 'neu
    if (segs_all['cnv_state']!='neu').sum() == 0:
        df_neu = df_neu.sort_values(by='CHROM', key=natsort.natsort_keygen())
        def assign_seg(rows):
            rows = rows.copy()
            n_ = len(rows)
            postfix = utils.generate_postfix(range(n_))
            rows['seg'] = [f"{row['CHROM']}{pfx}" for row, pfx in zip(rows.to_dict('records'), postfix)]
            rows['cnv_state'] = 'neu'
            rows['cnv_state_post'] = 'neu'
            return rows
    
        df_neu = df_neu.groupby('CHROM', group_keys=False, sort=False, observed=True)[df_neu.columns].apply(assign_seg).reset_index(drop=True)
        return df_neu
    # else
    segs_consensus = pd.concat([segs_star, df_retest], axis=0, ignore_index=True)
    segs_consensus = utils.fill_neu_segs(segs_consensus, df_neu)
    segs_consensus['cnv_state_post'] = np.where(
        segs_consensus['cnv_state']=='neu',
        segs_consensus['cnv_state'],
        segs_consensus['cnv_state_post']
    )

    return segs_consensus


def retest_bulks(
    bulks: pd.DataFrame,
    segs_consensus: pd.DataFrame = None,
    t: float = 1e-5,
    min_genes: int = 10,
    gamma: float = 20,
    nu: float = 1,
    use_loh: bool = False,
    diploid_chroms=None,
    ncores: int = 1,
    exclude_neu: bool = True,
    min_LLR: float = 5
    ) -> pd.DataFrame:
    """
    This function:
      1) Optionally builds a consensus set of CNV segments (segs_consensus).
      2) Decides whether to use LOH as baseline if no 'use_loh' input given.
      3) Annotates the 'bulk' data with the consensus segments,
         marks certain chromosomes or states as 'diploid'.
      4) Retests CNVs by calling run_group_hmms(...) with run_hmm=False,
         then sets any segments with LLR < min_LLR to 'neu'.

    Parameters
    ----------
    bulks : pd.DataFrame
        Pseudobulk profiles, must contain columns used by 'annot_consensus' and 'run_group_hmms'.
        Also 'CHROM', 'cnv_state', 'LLR', 'POS' (for partial references).
    segs_consensus : pd.DataFrame, optional
        If None, calls get_segs_consensus(bulks).
    t : float, optional
        Transition probability for run_group_hmms (default=1e-5).
    min_genes : int, optional
        For run_group_hmms. (default=10)
    gamma : float, optional
        Dispersion parameter for Beta-Binomial allele model in run_group_hmms (default=20).
    nu : float, optional
        Phase switch rate for run_group_hmms (default=1).
    use_loh : bool, optional
        If True, includes 'loh' as baseline state. If None or not provided in the R code,
        we decide based on the total neutral segment length in 'segs_consensus'.
    diploid_chroms : list of str or None, optional
        If provided, mark these chromosomes as 'diploid' in bulks. 
    ncores : int, optional
        Number of cores for parallel processing in run_group_hmms (default=1).
    exclude_neu : bool, optional
        Whether to exclude neutral states from retesting in run_group_hmms (default=True).
    min_LLR : float, optional
        LLR threshold to set any segment with LLR < min_LLR => 'neu' (default=5).

    Returns
    -------
    pd.DataFrame
        Updated pseudobulk DataFrame with retested CNVs.

    Notes
    -----
    - The logic for "deciding use_loh" if it was None is 
      a direct check of segs_consensus's neutral region length <1.5e8.
      If so, we log a message and set use_loh=True.
    """

    # If segs_consensus is None, build it
    if segs_consensus is None:
        segs_consensus = get_segs_consensus(bulks)

    # use_loh can be decide automatically if the total neutral
    # region < 1.5e8 ## OR THIS MAY BE TUNABLE
    if use_loh is None:
        segs_neu = segs_consensus[segs_consensus['cnv_state'] == 'neu']
        length_neu = segs_neu['seg_length'].sum()
        if length_neu < 1.5e8:
            use_loh = True
            print('less than 5% of genome is in neutral region - including LOH in baseline')
        else:
            use_loh = False

    if use_loh:
        ref_states = ['neu','loh']
    else:
        ref_states = ['neu']

    # annotate bulks
    bulks = utils.annot_consensus(bulks, segs_consensus)

    # If diploid_chroms is not None
    if diploid_chroms is not None:
        bulks['diploid'] = bulks['CHROM'].isin(diploid_chroms)
    else:
        bulks['diploid'] = bulks['cnv_state'].isin(ref_states)

    # retest CNVs
    bulks = run_group_hmms(bulks,
                           t=t,
                           gamma=gamma, 
                           nu=nu, 
                           min_genes=min_genes, 
                           run_hmm=False, 
                           exclude_neu=exclude_neu, 
                           ncores=ncores)

    bulks['LLR'] = bulks['LLR'].fillna(0)
    bulks.loc[ bulks['LLR']< min_LLR, 'cnv_state_post'] = 'neu'
    bulks['cnv_state'] = bulks['cnv_state_post']

    return bulks