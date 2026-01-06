#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 00:33:04 2025

@author: lillux
"""

from typing import Any, Dict, Union, Optional, List, Tuple, Literal
import math

import pandas as pd
import numpy as np
import scipy
from scipy.stats import binom

from joblib import cpu_count, Parallel, delayed
from numba import njit, prange

import networkx as nx
import pyranges as pr
import natsort

import anndata as ad

import tqdm
from . import utils, dist_prob, clustering, _progressbar, spatial_utils
import warnings

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
        log.info(f'Running HMMs on {n_groups} cell groups...')

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
    log.info(f'Running bulk analysis on {ncores} core')

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
    if segs_star.shape[0] == 0:
        msg = "All segments have been predicted to be neutral. Try to decrease min_LLR."
        log.info(msg)
        return
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
    
    # if all segs_all['cnv_state'] == 'neu'
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
            log.info('less than 5% of genome is in neutral region - including LOH in baseline')
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


def test_multi_allelic(
    bulks: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    min_LLR: float = 5,
    p_min: float = 0.999
    ) -> pd.DataFrame:
    """
    Detect multi-allelic CNV states per consensus segment and update probabilities.

    This function scans per-sample CNV evidence to find consensus segments
    that appear with more than one non-neutral CNV state across samples
    (for example, both deletion and amplification observed for the same
    consensus segment). For such multi-allelic segments, it sets a uniform
    probability of 0.5 for each implicated state among {del, amp, loh, bamp, bdel}
    and records the list of observed states. For segments that are not
    multi-allelic, it preserves the existing probabilities and records a single
    state (or zero if neutral).

    Steps
    -----
    1. Extract distinct rows from bulks with the required columns and compute
       p_max as the rowwise maximum among the CNV state probabilities.
    2. Keep rows with LLR > min_LLR and p_max > p_min.
    3. Group by seg_cons to collect unique cnv_state_post values and flag
       segments with more than one distinct non-neutral state.
    4. For multi-allelic segments, merge back into segs_consensus and update
       per-state probabilities and n_states.
    5. For non-multi-allelic segments, set n_states to 0 if neutral else 1,
       and record cnv_states accordingly.
    6. Convert cnv_states from list to comma-separated string.

    Parameters
    ----------
    bulks : pandas.DataFrame
        Input table with at least these columns:
        sample, CHROM, seg_cons, LLR, p_amp, p_del, p_bdel, p_loh, p_bamp, cnv_state_post.
        Rows are per-sample measurements summarized at consensus segments.
    segs_consensus : pandas.DataFrame
        Consensus segments to be annotated. Expected to contain columns such as:
        seg_cons, cnv_state, cnv_state_post, and probability columns
        p_del, p_amp, p_loh, p_bamp, p_bdel.
    min_LLR : float, default 5
        LLR threshold above which an event is considered supported.
    p_min : float, default 0.999
        Probability threshold above which an event is considered strongly
        supported when determining multi-allelic status.

    Returns
    -------
    pandas.DataFrame
        segs_consensus with added or updated columns:
        - n_states: number of non-neutral states observed for the segment
                    across samples (0, 1, or >1).
        - cnv_states: comma-separated string of observed states. For multi-allelic
                      segments this includes multiple states; otherwise a single
                      state or empty for neutral.
        - For multi-allelic segments, probability columns p_del, p_amp, p_loh,
          p_bamp, p_bdel are set to 0.5 for states that appear and 0.0 otherwise.

    Notes
    -----
    - No columns are created or dropped beyond those assigned in the body.
      Ensure segs_consensus already includes the probability columns that may
      be updated.
    - The function logs a summary with the number and IDs of multi-allelic
      segments found.
    - If bulks does not contain any rows passing the thresholds, no segments
      are marked as multi-allelic.

    Examples
    --------
    >>> segs_consensus_out = test_multi_allelic(
    ...     bulks=bulks_df,
    ...     segs_consensus=segs_df,
    ...     min_LLR=6,
    ...     p_min=0.995
    ... )
    """

    log.info('Testing for multi-allelic CNVs ..')

    cols_needed = ['sample','CHROM','seg_cons','LLR','p_amp','p_del','p_bdel','p_loh','p_bamp','cnv_state_post']
    bulks_dist = bulks[cols_needed].drop_duplicates().copy()
    bulks_dist['p_max'] = bulks_dist[['p_amp','p_del','p_bdel','p_loh','p_bamp']].max(axis=1)
    bulks_dist = bulks_dist[(bulks_dist['LLR']>min_LLR) & (bulks_dist['p_max']>p_min)]

    if bulks_dist.empty:
        segs_multi = pd.DataFrame(columns=['seg_cons','cnv_states','n_states'])
    else:
        grouped = bulks_dist.groupby('seg_cons', as_index=False, observed = True, sort = False)[bulks_dist.columns]
        def aggregator(df_g: pd.DataFrame):
            states = sorted(df_g['cnv_state_post'].unique())
            return pd.Series({
                'cnv_states': states,
                'n_states': len(states)
            })
        segs_multi = grouped.apply(aggregator)
        segs_multi = segs_multi[ segs_multi['n_states']>1 ].copy()

    segs = segs_multi['seg_cons'].values.tolist()

    count_segs = len(segs)
    seg_str = ", ".join(str(s) for s in segs)
    log.info(f"{count_segs} multi-allelic CNVs found: {seg_str}")

    segs_consensus = segs_consensus.copy()
    if count_segs>0:
        segs_consensus = segs_consensus.merge(segs_multi, on='seg_cons', how='left')

        def update_p(row):
            cnv_states = row['cnv_states']
            if np.any(pd.isna(cnv_states)):
                cnv_states = [row['cnv_state_post']]
            
            n_states = sum([1 for x in cnv_states if x!='neu'])
            row['n_states'] = n_states

            if n_states>1:
                row['p_del']  = 0.5 if ('del'  in cnv_states) else 0.0
                row['p_amp']  = 0.5 if ('amp'  in cnv_states) else 0.0
                row['p_loh']  = 0.5 if ('loh'  in cnv_states) else 0.0
                row['p_bamp'] = 0.5 if ('bamp' in cnv_states) else 0.0
                row['p_bdel'] = 0.5 if ('bdel' in cnv_states) else 0.0
            else:
                # keep existing p
                pass

            row['cnv_states'] = cnv_states
            return row

        segs_consensus = segs_consensus.apply(update_p, axis=1)
    else:
        def simple_mutate(row):
            if row['cnv_state'] == 'neu':
                row['n_states'] = 0
            else:
                row['n_states'] = 1
            row['cnv_states'] = row['cnv_state']
            return row
        segs_consensus = segs_consensus.apply(simple_mutate, axis=1)

    def list_to_str(val):
        if isinstance(val, list):
            return ",".join(val)
        return val if pd.notna(val) else ""

    segs_consensus['cnv_states'] = segs_consensus['cnv_states'].apply(list_to_str)

    return segs_consensus


def get_exp_sc(
    segs_consensus: pd.DataFrame,
    count_mat:ad.AnnData,
    gtf: pd.DataFrame,
    segs_loh: Optional[pd.DataFrame] = None) -> ad.AnnData:
    """
    Build a per-gene, per-segment single-cell expression object and annotate LOH.

    This function maps genes to consensus CNV segments using genomic overlap
    (via PyRanges), reorders and subsets the AnnData matrix columns to match
    genomic order, adds segment index metadata for each gene, and optionally
    flags genes that fall within clonal LOH intervals.

    Parameters
    ----------
    segs_consensus : pandas.DataFrame
        Consensus segments with at least the following columns:
        CHROM, seg_start, seg_end, seg_cons or seg (the function renames seg_cons to seg).
    count_mat : anndata.AnnData
        Single-cell count matrix. The following must be present in `count_mat.var`:
        - Index: gene symbols used in the GTF table column `gene`.
        - CHROM, gene_start, gene_end (genomic coordinates).
    gtf : pandas.DataFrame
        Gene annotation with columns:
        CHROM, gene_start, gene_end, gene.
    segs_loh : pandas.DataFrame, optional
        LOH segments with columns CHROM, seg_start, seg_end.
        If provided, overlapping genes are flagged as loh=True in the output.

    Returns
    -------
    anndata.AnnData
        returns an AnnData object (the input `count_mat` after being
        subset and annotated). The returned object includes in `var`:
        CHROM, gene_start, gene_end, seg, seg_start, seg_end, cnv_state,
        gene_index, seg_start_index, seg_end_index, n_genes, and loh (if `segs_loh` is given).

    Notes
    -----
    - Segment index metadata are computed within each segment:
      seg_start_index, seg_end_index, and n_genes count in the genomic order.
    - If `segs_loh` is provided, LOH is assigned using a point overlap at
      gene_start.
    """
    
    # Build genome ranges
    gtf_temp = gtf.copy().reset_index(drop=True)
    gtf_temp['gene_index'] = np.arange(len(gtf_temp))
    
    pr_genes = pr.PyRanges(
        pd.DataFrame({
            'Chromosome': gtf_temp['CHROM'],
            'Start': gtf_temp['gene_start'],
            'End': gtf_temp['gene_end'],
            'gene_index': gtf_temp['gene_index']
        })
    )
    # Build seg_consensus ranges
    segs_temp = segs_consensus.copy().reset_index(drop=True)
    segs_temp['seg_index'] = np.arange(len(segs_temp))
    
    pr_segs = pr.PyRanges(
        pd.DataFrame({
            'Chromosome': segs_temp['CHROM'],
            'Start': segs_temp['seg_start'],
            'End': segs_temp['seg_end'],
            'seg_index': segs_temp['seg_index']
        })
    )
    # Put seg index on 
    ov = pr_genes.join(pr_segs).as_df()
    
    df_ov = ov.rename(columns={'Chromosome':'CHROM','Start':'gene_start', 'End':'gene_end','Start_b':'seg_start','End_b':'seg_end'})
    df_ov = df_ov.merge(gtf_temp, on='gene_index', how='left')
    df_ov = df_ov.drop(['CHROM_y','gene_start_y','gene_end_y'], axis=1)
    df_ov = df_ov.rename(columns={'CHROM_x':'CHROM', 'gene_start_x':'gene_start', 'gene_end_x':'gene_end'})
    df_ov = df_ov.merge(segs_temp, on='seg_index', how='left')
    df_ov = df_ov.drop(['CHROM_y', 'seg', 'seg_start_y','seg_end_y'], axis=1)
    df_ov = df_ov.rename(columns={'CHROM_x':'CHROM', 'seg_start_x':'seg_start', 'seg_end_x':'seg_end'})
    df_ov = df_ov.drop_duplicates(subset=['gene']) # THIS MAY LOSE SOME GENES. MAYBE CHECK FOR ENSEMBL ID
    df_ov = df_ov.rename(columns={'seg_cons':'seg'})
    
    df_ov_filt = df_ov.loc[:,['CHROM','gene','seg','seg_start','seg_end','gene_start','gene_end','cnv_state']].copy()
    var_merged = count_mat.var.merge(df_ov_filt, left_index=True, right_on='gene', how='inner').copy()
    var_merged.index = var_merged.gene
    var_merged_sort = var_merged.sort_values(['CHROM', 'gene_start'], key=natsort.natsort_keygen()).copy()
    count_mat_selected = count_mat[:,var_merged_sort.index].copy()
    count_mat_selected.var = var_merged_sort.copy()
    count_mat_selected.var.loc[:,'gene_index'] = np.arange(count_mat_selected.shape[1])
    
    var_group = count_mat_selected.var.groupby('seg', sort=False, observed=True)
    seg_index_df = pd.DataFrame({'seg_start_index': np.repeat(np.nan, count_mat_selected.shape[1]),
                                 'seg_end_index': np.repeat(np.nan, count_mat_selected.shape[1]),
                                 'n_genes': np.repeat(np.nan, count_mat_selected.shape[1])})
    seg_index_df.index = count_mat_selected.var.index.copy()
    for k, group in var_group:
        seg_index_df.loc[group.index,'seg_start_index'] = group.gene_index.min()
        seg_index_df.loc[group.index,'seg_end_index'] = group.gene_index.max()
        seg_index_df.loc[group.index,'n_genes'] = group.shape[0]
    
    count_mat_selected.var = count_mat_selected.var.merge(seg_index_df, left_index=True, right_index=True)
    
    # exclude_loh
    exp_sc = exclude_loh(count_mat_selected, segs_loh)
    return exp_sc


def exclude_loh(exp_sc: ad.AnnData, segs_loh: Optional[pd.DataFrame] = None) -> ad.AnnData:
    """
    Flag genes that overlap clonal LOH regions in an AnnData object.

    If LOH segments are provided, genes whose genomic start coordinate falls
    within any LOH interval are marked with var['loh'] = True; otherwise False.
    If `segs_loh` is None, all genes are marked as loh=False.

    Parameters
    ----------
    exp_sc : anndata.AnnData
        AnnData object with gene metadata in `var`. Required columns:
        CHROM, gene_start, gene_index.
    segs_loh : pandas.DataFrame, optional
        Table of LOH segments with columns CHROM, seg_start, seg_end.

    Returns
    -------
    anndata.AnnData
        returns the modified AnnData object `exp_sc` with a boolean
        column var['loh'].
    """
    if segs_loh is None:
        exp_sc.var.loc[:,'loh'] = False
        return exp_sc
    
    print('Excluding clonal LOH regions ..')
    
    pr_genes = pr.PyRanges(
        pd.DataFrame({
            'Chromosome': exp_sc.var['CHROM'],
            'Start': exp_sc.var['gene_start'],
            'End': exp_sc.var['gene_end'], # 'gene_start'
    'gene_index': exp_sc.var['gene_index']
        })
    )
    
    pr_loh = pr.PyRanges(
        pd.DataFrame({
            'Chromosome': segs_loh['CHROM'],
            'Start': segs_loh['seg_start'],
            'End': segs_loh['seg_end'],
            'loh_index': np.arange(segs_loh.shape[0])
        })
    )
    
    ov = pr_genes.join(pr_loh).as_df()
    gene_idx_loh = ov.gene_index.unique()
    
    exp_sc.var.loc[:,'loh'] = False
    gene_idx = exp_sc.var[[i in set(gene_idx_loh) for i in exp_sc.var.gene_index]].index
    exp_sc.var.loc[gene_idx, 'loh'] = True

    return exp_sc


def get_exp_likelihoods(
    exp_counts: pd.DataFrame,
    diploid_chroms: Optional[List[str]] = None,
    use_loh: bool = False,
    depth_obs: Optional[float] = None,
    mu: Optional[float] = None,
    sigma: Optional[float] = None,
    disp: bool = False,
    n_points: int = 256
    ) -> pd.DataFrame:
    """
    Compute expression-model likelihood summaries per segment.

    This function filters and summarizes per-segment expression counts under a
    lognormal–Poisson (LN-Poisson) model. If baseline parameters (mu, sigma)
    are not provided, they are estimated from putatively diploid data using
    either the chromosomes listed in diploid_chroms or rows whose cnv_state is
    in the reference set (neu, and optionally loh). For each combination of
    CHROM, seg, and cnv_state, it returns the number of rows, the maximum
    likelihood estimate of phi (scale), several fixed-phi log-likelihood
    evaluations, and the baseline parameters used.

    Parameters
    ----------
    exp_counts : pd.DataFrame
        Input table with at least the following columns:
        Y_obs (observed counts), lambda_ref (reference rate), CHROM, seg,
        cnv_state, and loh (boolean) if diploid_chroms is not provided.
        Additional columns are allowed and ignored.
    diploid_chroms : list of str or None, optional
        If provided, baseline fitting (mu, sigma) uses rows on these chromosomes
        with loh != True. If None, fitting uses rows with cnv_state in the
        reference set (neu, and loh if use_loh is True) and loh != True.
    use_loh : bool, optional
        If True, the LOH state is included in the reference set for baseline
        fitting; otherwise only neu is used. Default is False.
    depth_obs : float or None, optional
        Library depth to pass to the likelihood functions. If None, it is set
        to exp_counts['Y_obs'].sum() after filtering. Default is None.
    mu : float or None, optional
        Lognormal mean parameter. If None, it is estimated as described above.
        Default is None.
    sigma : float or None, optional
        Lognormal standard deviation parameter. If None, it is estimated as
        described above. Default is None.
    disp : bool, optional
        If True, passed through to the fitter to enable verbose output.
        Default is False.
    n_points : int, optional
        Number of evaluation points used inside l_lnpois when computing fixed-phi
        log-likelihoods. Default is 256.

    Returns
    -------
    pd.DataFrame
        One row per (CHROM, seg, cnv_state) present in non-neutral data, with
        columns:
        n, phi_mle, l11, l20, l10, l21, l31, l22, l32, l00, mu, sigma,
        along with the group keys (CHROM, seg, cnv_state).

    Notes
    -----
    - Rows with missing Y_obs or nonpositive lambda_ref are removed.
    - By design, l20 equals l11 and l22 equals l31 in the output.
    - The function drops rows where cnv_state == 'neu' before grouping.
    """

    exp_counts_filtered = exp_counts.dropna(subset=['Y_obs']).copy()
    exp_counts_filtered = exp_counts_filtered[exp_counts_filtered['lambda_ref']>0].copy()

    if depth_obs is None:
        depth_obs = exp_counts_filtered['Y_obs'].sum()

    # define reference states
    if use_loh:
        ref_states = ['neu','loh']
    else:
        ref_states = ['neu']

    # fit mu and sigma if not there already
    if mu is None or sigma is None:
        if diploid_chroms is not None:
            df_dip = exp_counts_filtered[(exp_counts_filtered['loh']!=True) 
                                        & (exp_counts_filtered['CHROM'].isin(diploid_chroms))].copy()
        else:
            df_dip = exp_counts_filtered[(exp_counts_filtered['loh']!=True) 
                                        & (exp_counts_filtered['cnv_state'].isin(ref_states))].copy()

        fit = dist_prob.fit_lnpois(df_dip['Y_obs'].values,
                         df_dip['lambda_ref'].values,
                         depth_obs,
                         disp=disp)
        mu = fit[0]
        sigma = fit[1]

    # Summarize for each (CHROM, seg, cnv_state)
    group_cols = ['CHROM','seg','cnv_state']
    def aggregator(df: pd.DataFrame) -> pd.Series:
        n = len(df)
        phi_mle_val = calc_phi_mle_lnpois(df['Y_obs'].values,
                                          df['lambda_ref'].values,
                                          depth_obs,
                                          mu, sigma,
                                          lower=0.1,
                                          upper=10)

        l11_val = dist_prob.l_lnpois(df['Y_obs'].values, df['lambda_ref'].values, depth_obs, mu, sigma, phi=1.0, n_points=n_points)
        l20_val = l11_val
        l10_val = dist_prob.l_lnpois(df['Y_obs'].values, df['lambda_ref'].values, depth_obs, mu, sigma, phi=0.5, n_points=n_points)
        l21_val = dist_prob.l_lnpois(df['Y_obs'].values, df['lambda_ref'].values, depth_obs, mu, sigma, phi=1.5, n_points=n_points)
        l31_val = dist_prob.l_lnpois(df['Y_obs'].values, df['lambda_ref'].values, depth_obs, mu, sigma, phi=2.0, n_points=n_points)
        l22_val = l31_val
        l32_val = dist_prob.l_lnpois(df['Y_obs'].values, df['lambda_ref'].values, depth_obs, mu, sigma, phi=2.5, n_points=n_points)
        l00_val = dist_prob.l_lnpois(df['Y_obs'].values, df['lambda_ref'].values, depth_obs, mu, sigma, phi=0.25, n_points=n_points)

        return pd.Series({
            'n': n,
            'phi_mle': phi_mle_val,
            'l11': l11_val,
            'l20': l20_val,
            'l10': l10_val,
            'l21': l21_val,
            'l31': l31_val,
            'l22': l22_val,
            'l32': l32_val,
            'l00': l00_val,
            'mu': mu,
            'sigma': sigma
        })
    exp_counts_filtered = exp_counts_filtered[exp_counts_filtered.cnv_state != 'neu']
    results = exp_counts_filtered.groupby(group_cols, as_index=False, observed=True, sort=False)[exp_counts_filtered.columns].apply(aggregator)

    return results.reset_index(drop=True)


def calc_phi_mle_lnpois(
    Y_obs: np.ndarray,
    lambda_ref: np.ndarray,
    d: float,
    mu: float,
    sig: float,
    lower: float = 0.1,
    upper: float = 10.0,
    disp: bool = False
) -> float:
    """
    Maximum-likelihood estimate of phi under an LN-Poisson model.

    This function finds the value of the multiplicative scale parameter phi that
    maximizes the LN-Poisson log-likelihood l_lnpois for the given counts and
    reference rates. The optimization uses L-BFGS-B with bounds [lower, upper]
    and starts at 1.0 clipped to the bounds.

    Parameters
    ----------
    Y_obs : np.ndarray
        Observed counts for the segment.
    lambda_ref : np.ndarray
        Reference rates for the same rows as Y_obs.
    d : float
        Library depth or exposure term passed to the likelihood.
    mu : float
        Lognormal mean parameter.
    sig : float
        Lognormal standard deviation parameter.
    lower : float, optional
        Lower bound for phi. Default is 0.1.
    upper : float, optional
        Upper bound for phi. Default is 10.0.
    disp : bool, optional
        If True, enable optimizer verbosity. Default is False.

    Returns
    -------
    float
        The maximizing value of phi. If Y_obs is empty, returns 1.0.

    Notes
    -----
    - The objective minimized is the negative of l_lnpois evaluated at phi.
    """
    if len(Y_obs)==0:
        return 1.0

    start = max(min(1.0, upper), lower)

    def objective(phi):
        # negative log-likelihood
        return -dist_prob.l_lnpois(Y_obs, lambda_ref, d, mu, sig, phi=phi)

    res = scipy.optimize.minimize(
        objective,
        x0=[start],
        method='L-BFGS-B',
        bounds=[(lower, upper)],
        options={'disp': disp},
        tol = 1e-6 # added later
    )
    
    return res.x[0]


@njit
def _log_sum_exp(vals: np.ndarray) -> float:
    """
    Compute log(sum(exp(vals))) in a numerically stable way.

    This routine expects a 1D array of log-values and returns the
    log-sum-exp. It handles empty inputs and all-negative-infinity inputs.

    Parameters
    ----------
    vals : np.ndarray
        One-dimensional array of log-values (shape: (n,)).

    Returns
    -------
    float
        The logarithm of the sum of exponentials of the input values.
        Returns -inf if the input is empty. If all entries are -inf,
        returns -inf as well; if the maximum is non-finite, that
        maximum is returned.
    """
    if vals.shape[0] == 0:
        return -np.inf
    max_val = np.max(vals)
    if not np.isfinite(max_val):
        return max_val  # e.g. if all entries are -inf
    cumsum = np.sum(np.exp(vals - max_val))
    return max_val + math.log(cumsum)


@njit(parallel=True)
def _compute_posterior_numba(
    l21: np.ndarray,
    l31: np.ndarray,
    l20: np.ndarray,
    l10: np.ndarray,
    l22: np.ndarray,
    l00: np.ndarray,
    l11: np.ndarray,
    prior_amp: np.ndarray,
    prior_loh: np.ndarray,
    prior_del: np.ndarray,
    prior_bamp: np.ndarray,
    prior_bdel: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized posterior calculations for CNV states (Numba-accelerated).

    For each row i, the function combines log-likelihood terms l..[i] with
    prior probabilities to produce intermediate log-evidence quantities Z_*,
    a total evidence Z, CNV-only evidence Z_cnv, and posterior probabilities
    for each state. Computations are carried out in log space using
    log-sum-exp for numerical stability.

    The states and their log-evidence are:
      - amp: log-sum-exp of l21 + log(prior_amp/4) and l31 + log(prior_amp/4)
      - loh: l20 + log(prior_loh/2)
      - del: l10 + log(prior_del/2)
      - bamp: l22 + log(prior_bamp/2)
      - bdel: l00 + log(prior_bdel/2)
      - neu: l11 + log(1/2)

    The totals are:
      - Z = log-sum-exp([Z_n, Z_loh, Z_del, Z_amp, Z_bamp, Z_bdel])
      - Z_cnv = log-sum-exp([Z_loh, Z_del, Z_amp, Z_bamp, Z_bdel])

    Posterior probabilities are exp(Z_state - Z). The log Bayes factor
    is logBF = Z_cnv - Z_n. p_cnv = exp(Z_cnv - Z), p_n = p_neu.

    Parameters
    ----------
    l21, l31, l20, l10, l22, l00, l11 : np.ndarray
        One-dimensional arrays of log-likelihood components for each state
        (shape: (n,)). All must be float64-compatible.
    prior_amp, prior_loh, prior_del, prior_bamp, prior_bdel : np.ndarray
        One-dimensional arrays of prior probabilities for the corresponding
        states (shape: (n,)). Values must be strictly greater than zero to
        avoid log(0).

    Returns
    -------
    Tuple of np.ndarray
        Seventeen 1D arrays of length n, in the following order:
        Z_amp, Z_loh, Z_del, Z_bamp, Z_bdel, Z_n, Z, Z_cnv,
        p_amp, p_neu, p_del, p_loh, p_bamp, p_bdel, logBF, p_cnv, p_n.

    Notes
    -----
    - Inputs must be contiguous float64 arrays for best Numba performance.
    - Priors are used inside log operations; zeros will cause -inf. If needed,
      clip priors to a small epsilon > 0 before calling.
    - Parallelization uses prange over rows.
    """
    n = l21.shape[0]
    
    Z_amp = np.empty(n, dtype=np.float64)
    Z_loh = np.empty(n, dtype=np.float64)
    Z_del = np.empty(n, dtype=np.float64)
    Z_bamp = np.empty(n, dtype=np.float64)
    Z_bdel = np.empty(n, dtype=np.float64)
    Z_n = np.empty(n, dtype=np.float64)
    Z = np.empty(n, dtype=np.float64)
    Z_cnv = np.empty(n, dtype=np.float64)
    
    p_amp = np.empty(n, dtype=np.float64)
    p_neu = np.empty(n, dtype=np.float64)
    p_del = np.empty(n, dtype=np.float64)
    p_loh = np.empty(n, dtype=np.float64)
    p_bamp = np.empty(n, dtype=np.float64)
    p_bdel = np.empty(n, dtype=np.float64)
    
    logBF = np.empty(n, dtype=np.float64)
    p_cnv = np.empty(n, dtype=np.float64)
    p_n = np.empty(n, dtype=np.float64)
    
    for i in prange(n):
        a = l21[i] + math.log(prior_amp[i] / 4.0)
        b = l31[i] + math.log(prior_amp[i] / 4.0)
        Z_amp[i] = _log_sum_exp(np.array([a, b]))
        
        Z_loh[i] = l20[i] + math.log(prior_loh[i] / 2.0)
        Z_del[i] = l10[i] + math.log(prior_del[i] / 2.0)
        Z_bamp[i] = l22[i] + math.log(prior_bamp[i] / 2.0)
        Z_bdel[i] = l00[i] + math.log(prior_bdel[i] / 2.0)
        Z_n[i] = l11[i] + math.log(1.0 / 2.0)
        
        # Compute overall Z = log_sum_exp([Z_n, Z_loh, Z_del, Z_amp, Z_bamp, Z_bdel])
        arr = np.empty(6, dtype=np.float64)
        arr[0] = Z_n[i]
        arr[1] = Z_loh[i]
        arr[2] = Z_del[i]
        arr[3] = Z_amp[i]
        arr[4] = Z_bamp[i]
        arr[5] = Z_bdel[i]
        Z[i] = _log_sum_exp(arr)
        
        # Compute Z_cnv = log_sum_exp([Z_loh, Z_del, Z_amp, Z_bamp, Z_bdel])
        arr2 = np.empty(5, dtype=np.float64)
        arr2[0] = Z_loh[i]
        arr2[1] = Z_del[i]
        arr2[2] = Z_amp[i]
        arr2[3] = Z_bamp[i]
        arr2[4] = Z_bdel[i]
        Z_cnv[i] = _log_sum_exp(arr2)
        
        # Compute posterior probabilities
        p_amp[i] = math.exp(Z_amp[i] - Z[i])
        p_neu[i] = math.exp(Z_n[i] - Z[i])
        p_del[i] = math.exp(Z_del[i] - Z[i])
        p_loh[i] = math.exp(Z_loh[i] - Z[i])
        p_bamp[i] = math.exp(Z_bamp[i] - Z[i])
        p_bdel[i] = math.exp(Z_bdel[i] - Z[i])
        logBF[i] = Z_cnv[i] - Z_n[i]
        p_cnv[i] = math.exp(Z_cnv[i] - Z[i])
        p_n[i] = p_neu[i]
        
    return (Z_amp, Z_loh, Z_del, Z_bamp, Z_bdel, Z_n, Z, Z_cnv, 
            p_amp, p_neu, p_del, p_loh, p_bamp, p_bdel, logBF, p_cnv, p_n)


def compute_posterior(PL: pd.DataFrame) -> pd.DataFrame:
    """
    Compute posterior probabilities and related statistics for an HMM model.
    
    This function takes a DataFrame `PL` containing various log-likelihoods and prior 
    probabilities associated with different CNV states. For each row, it computes 
    several derived quantities including the log-sum-exp of combinations of log-likelihoods,
    posterior probabilities for each state, and the log Bayes Factor.
    
    The computed values are:
      - Z_amp, Z_loh, Z_del, Z_bamp, Z_bdel, Z_n: Intermediate log-probabilities.
      - Z: The overall log-sum-exp of all states.
      - Z_cnv: The log-sum-exp of CNV-related states.
      - p_amp, p_neu, p_del, p_loh, p_bamp, p_bdel: Posterior probabilities for each state.
      - logBF: Log Bayes Factor computed as Z_cnv - Z_n.
      - p_cnv: Posterior probability for CNV.
      - p_n: Posterior probability for the neutral state.
    
    Parameters
    ----------
    PL : pd.DataFrame
        A DataFrame with columns:
          - 'l21', 'l31', 'l20', 'l10', 'l22', 'l00', 'l11'
          - 'prior_amp', 'prior_loh', 'prior_del', 'prior_bamp', 'prior_bdel'
    
    Returns
    -------
    pd.DataFrame
        The input DataFrame augmented with the computed columns:
        'Z_amp', 'Z_loh', 'Z_del', 'Z_bamp', 'Z_bdel', 'Z_n', 'Z', 'Z_cnv',
        'p_amp', 'p_neu', 'p_del', 'p_loh', 'p_bamp', 'p_bdel', 'logBF', 'p_cnv', 'p_n'.
    """
    # Extract required columns from the DataFrame as NumPy arrays.
    l21 = PL['l21'].values.astype(np.float64)
    l31 = PL['l31'].values.astype(np.float64)
    l20 = PL['l20'].values.astype(np.float64)
    l10 = PL['l10'].values.astype(np.float64)
    l22 = PL['l22'].values.astype(np.float64)
    l00 = PL['l00'].values.astype(np.float64)
    l11 = PL['l11'].values.astype(np.float64)
    
    prior_amp = PL['prior_amp'].values.astype(np.float64)
    prior_loh = PL['prior_loh'].values.astype(np.float64)
    prior_del = PL['prior_del'].values.astype(np.float64)
    prior_bamp = PL['prior_bamp'].values.astype(np.float64)
    prior_bdel = PL['prior_bdel'].values.astype(np.float64)
    
    # Compute all posterior values using a parallelized Numba function.
    (Z_amp, Z_loh, Z_del, Z_bamp, Z_bdel, Z_n, Z, Z_cnv, 
     p_amp, p_neu, p_del, p_loh, p_bamp, p_bdel, logBF, p_cnv, p_n) = _compute_posterior_numba(l21, l31, l20, l10, l22, l00, l11,
                                                                            prior_amp, prior_loh, prior_del, prior_bamp, prior_bdel)
    
    # Create a copy of the input DataFrame to hold the results.
    PL_out = PL.copy()
    # Add the computed columns to the DataFrame.
    PL_out['Z_amp'] = Z_amp
    PL_out['Z_loh'] = Z_loh
    PL_out['Z_del'] = Z_del
    PL_out['Z_bamp'] = Z_bamp
    PL_out['Z_bdel'] = Z_bdel
    PL_out['Z_n'] = Z_n

    PL_out['Z'] = Z
    PL_out['Z_cnv'] = Z_cnv
    PL_out['p_amp'] = p_amp
    PL_out['p_neu'] = p_neu
    PL_out['p_del'] = p_del
    PL_out['p_loh'] = p_loh
    PL_out['p_bamp'] = p_bamp
    PL_out['p_bdel'] = p_bdel
    PL_out['logBF'] = logBF
    PL_out['p_cnv'] = p_cnv
    PL_out['p_n'] = p_n
    
    return PL_out


def get_exp_post(
    segs_consensus: pd.DataFrame,
    count_mat: ad.AnnData,
    gtf: pd.DataFrame,
    lambdas_ref: pd.DataFrame,
    sc_refs: Optional[pd.Series] = None,
    diploid_chroms: Optional[List[str]] = None,
    use_loh: Optional[bool] = None,
    segs_loh: Optional[pd.DataFrame] = None,
    ncores: int = 30,
    verbose: bool = True,
    debug: bool = False,
    n_points: int = 200
    ) -> pd.DataFrame:
    """
    Compute per-cell expression-based posteriors for CNV states and merge them with segment priors.

    This function builds gene-to-segment mappings per cell, decides whether to include LOH
    in the baseline, selects a reference profile for each cell, computes expression
    likelihoods per segment, merges those likelihoods with segment-level priors, and then
    computes posterior probabilities for CNV states. Results from all cells are concatenated
    into a single DataFrame.

    Workflow
    --------
    1. Build a per-gene mapping to consensus segments using get_exp_sc.
    2. Decide whether to include LOH in the baseline if use_loh is not provided.
    3. If sc_refs is not provided, choose per-cell reference columns using choose_ref_cor.
    4. In parallel across cells:
       - Extract counts and reference rates for the cell.
       - Compute expression likelihoods via get_exp_likelihoods.
    5. Merge the per-cell likelihoods with segment-level priors from segs_consensus.
    6. Optionally set very small priors to zero (see Notes).
    7. Run compute_posterior to obtain posteriors and Bayes factors.
    8. Add a convenience label column seg_label like "seg(state)".

    Parameters
    ----------
    segs_consensus : pd.DataFrame
        Consensus segments with at least the columns:
        CHROM, seg_cons, seg_start, seg_end, p_loh, p_amp, p_del, p_bamp, p_bdel.
    count_mat : ad.AnnData
        Single-cell count matrix. The variable dimension (var) must contain gene
        annotations used by get_exp_sc, including CHROM and gene_start. The .X
        matrix is used to extract per-cell gene counts.
    gtf : pd.DataFrame
        Gene annotation table with columns such as CHROM, gene, gene_start, gene_end.
        Used by get_exp_sc to map genes to segments.
    lambdas_ref : pd.DataFrame
        Reference expression rates (e.g., per-gene lambda_ref). Index must match gene
        identifiers used in count_mat.var.index. Columns represent reference profiles.
    sc_refs : Optional[pd.Series], default None
        Mapping from cell ID to a column name in lambdas_ref to use as that cell's
        reference (index is cell IDs, values are column labels). If None, it is
        computed by choose_ref_cor.
    diploid_chroms : Optional[List[str]], default None
        If provided, restricts the baseline fitting in get_exp_likelihoods to these
        chromosomes (excluding LOH when building the baseline).
    use_loh : Optional[bool], default None
        If None, decided automatically based on the fraction of genes in neutral
        non-LOH regions. If True, LOH segments are included in the baseline.
    segs_loh : Optional[pd.DataFrame], default None
        Intervals of clonal LOH to be excluded at the gene level by get_exp_sc.
        Expected columns include CHROM, seg_start, seg_end.
    ncores : int, default 30
        Number of parallel workers used when computing per-cell likelihoods.
    verbose : bool, default True
        If True, prints progress and summaries.
    debug : bool, default False
        Reserved for future use. Not used in the current implementation.
    n_points : int, default 200
        Integration grid size or evaluation resolution passed to get_exp_likelihoods.

    Returns
    -------
    pd.DataFrame
        Long-format table with one row per (cell, segment, state). Includes likelihood
        summaries from get_exp_likelihoods, merged priors, posterior probabilities
        from compute_posterior, and a seg_label column.

    """
    
    exp_sc = get_exp_sc(segs_consensus, count_mat, gtf, segs_loh)
    
    # Decide if use_loh
    if use_loh is None:
        fraction_neu_notloh = np.mean((exp_sc.var['cnv_state']=='neu') & (~exp_sc.var['loh']))
        if fraction_neu_notloh<0.05:
            use_loh = True
            log.info('less than 5% genes are in neutral region - including LOH in baseline')
        else:
            use_loh = False
    else:
        if use_loh:
            log.info('Including LOH in baseline as specified')
    
    if sc_refs is None:
        sc_refs = clustering.choose_ref_cor(count_mat, lambdas_ref, gtf)
    cells = list(sc_refs.index)
    
    def process_cell(cell):
        try:
            ref = sc_refs[cell]
            sc_exp_data = exp_sc[cell, :].var.loc[:,['seg', 'CHROM', 'cnv_state', 'loh', 'seg_start', 'seg_end']].copy()
            sc_exp_data.loc[:,'Y_obs'] = exp_sc[cell, sc_exp_data.index].X.toarray().ravel()
            sc_exp_data.loc[:,'lambda_ref'] = lambdas_ref.loc[sc_exp_data.index,ref]
            sc_exp_data.loc[:,'lambda_obs'] = sc_exp_data.Y_obs / sc_exp_data.Y_obs.sum()
            sc_exp_data.loc[:,'logFC'] = np.log2(sc_exp_data.lambda_obs) - np.log2(sc_exp_data.lambda_ref)
            cell_lik = get_exp_likelihoods(exp_counts=sc_exp_data,
                                           use_loh=use_loh,
                                           diploid_chroms=diploid_chroms,
                                           n_points=n_points)
            cell_lik.loc[:,'cell'] = cell
            cell_lik.loc[:,'ref'] = ref
            return cell_lik
        except Exception as e:
            return e  # pass back the exception
    
    if verbose:
        log.info('Computing expression likelihoods for each cell...')
        
    #with _progressbar.tqdm_joblib(tqdm.tqdm(desc="Processing cells", total=len(cells))) as progress_bar:
        # Use the top-level function for parallel execution:
        #results = Parallel(n_jobs=ncores)(delayed(process_cell)(cell) for cell in cells)
    with _progressbar.tqdm_joblib(total=len(cells), desc="Processing cells", disable=not verbose) as pbar:
        results = Parallel(n_jobs=ncores)(delayed(process_cell)(cell) for cell in cells)

    # check for errors
    bad = [isinstance(r, Exception) for r in results]

    if any(bad):
        if verbose:
            log.warning(f"{sum(bad)} cell(s) failed")
        first_error = [r for r in results if isinstance(r, Exception)][0]
        bad_cell = np.array(cells)[np.where(np.array(bad) == True)]
        log.warning(str(first_error))
        nl = '\n'
        log.warning(f"Bad cells are:\n{nl.join(bad_cell)}")
    else:
        log.info('All cells succeeded')
    
    # gather good result
    good_results = [r for r in results if not isinstance(r, Exception)]
    exp_post = pd.concat(good_results).reset_index(drop=True)
    
    segs_cons_temp = segs_consensus.loc[:,['CHROM',
                                           'seg_cons',
                                           'seg_start',
                                           'seg_end',
                                           'p_loh',
                                           'p_amp',
                                           'p_del',
                                           'p_bamp',
                                           'p_bdel']].copy()
    segs_cons_temp = segs_cons_temp.rename(columns={'seg_cons':'seg',
                                                    'p_loh':'prior_loh',
                                                    'p_amp':'prior_amp',
                                                    'p_del':'prior_del',
                                                    'p_bamp':'prior_bamp',
                                                    'p_bdel':'prior_bdel'})

    exp_post_merged = exp_post.merge(segs_cons_temp, on=['seg','CHROM'])
    prior_cols = ['prior_loh','prior_amp','prior_del','prior_bamp','prior_bdel']
    for c in prior_cols:
        exp_post_merged.loc[exp_post_merged[c]<0.05, c] = 1e-12
    log.info('Disabling system warnings...')
    # warnings.filterwarnings('ignore')
    exp_posterior = compute_posterior(exp_post_merged)
    # warnings.filterwarnings('always')
    log.info('System warnings enabled.')
    exp_posterior['seg_label'] = exp_posterior.apply(lambda r: f"{r['seg']}({r['cnv_state']})", axis=1)

    return exp_posterior


def get_haplotype_post(
    bulks: pd.DataFrame, 
    segs_consensus: pd.DataFrame, 
    naive: bool = False
    ) -> pd.DataFrame:
    """
    Get phased haplotypes from pseudobulk profiles and consensus CNV segments.
    
    This function processes two DataFrames:
      - `bulks`: A DataFrame of subtree pseudobulk profiles containing columns such as
                 'CHROM', 'seg', 'snp_id', 'pAD', 'AR', and optionally 'sample'.
      - `segs_consensus`: A DataFrame of consensus CNV segments containing columns such as
                          'cnv_state_post' and 'seg_cons', and optionally 'sample'.
    
    The function ensures that both DataFrames have a 'sample' column. If not present, a 
    default value ('0') is assigned. Then, it checks that there is at least one CNV (i.e.,
    not all consensus segments are marked as 'neu'). If the `naive` flag is True, the 
    function assigns a naive haplotype classification based on the 'AR' (allelic ratio)
    column in `bulks` (i.e., 'major' if AR >= 0.5, otherwise 'minor').
    
    Next, it filters the `bulks` DataFrame to include only rows with non-missing 'pAD'
    values, selects the columns of interest, and performs an inner join with `segs_consensus`
    on the keys 'sample', 'CHROM', and 'seg'. Finally, it selects and renames columns to 
    produce the final DataFrame of posterior haplotypes.
    
    Parameters
    ----------
    bulks : pd.DataFrame
        DataFrame containing pseudobulk profiles. Expected to have at least the columns:
        'CHROM', 'seg', 'snp_id', 'pAD', 'AR'. Optionally, a 'sample' column.
    segs_consensus : pd.DataFrame
        DataFrame containing consensus CNV segments. Expected to have at least the columns:
        'cnv_state_post' and 'seg_cons'. Optionally, a 'sample' column.
    naive : bool, default False
        Whether to use naive haplotype classification. If True, the haplotype posterior 
        ('haplo_post') in `bulks` is set to 'major' if the allelic ratio (AR) is at least 0.5,
        and 'minor' otherwise.
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the posterior haplotypes with the following columns:
            - 'CHROM': Chromosome identifier.
            - 'seg': Consensus segment identifier (renamed from 'seg_cons').
            - 'cnv_state': CNV state.
            - 'snp_id': SNP identifier.
            - 'haplo_post': Haplotype posterior classification.
    
    Raises
    ------
    ValueError
        If all entries in segs_consensus['cnv_state_post'] are 'neu', indicating that no CNVs 
        are present.
    """

    # Ensure both DataFrames have a 'sample' column. If not, assign a default value '0'.
    if 'sample' not in bulks.columns or 'sample' not in segs_consensus.columns:
        bulks = bulks.copy()
        segs_consensus = segs_consensus.copy()
        bulks['sample'] = '0'
        segs_consensus['sample'] = '0'
    
    # If all consensus segments are neutral, there is nothing to test.
    if (segs_consensus['cnv_state_post'] == 'neu').all():
        raise ValueError("No CNVs")
    
    # If using naive haplotype classification, set haplo_post based on AR.
    if naive:
        bulks = bulks.copy()
        # Naively classify haplotypes
        bulks['haplo_post'] = np.where(bulks['AR'] >= 0.5, 'major', 'minor')
    
    # Filter the bulks DataFrame to include only rows where pAD is not missing.
    bulks_filtered = bulks[~bulks['pAD'].isna()]
    # Select the relevant columns from bulks.
    bulks_sel = bulks_filtered.loc[:,['CHROM', 'seg', 'snp_id', 'sample', 'haplo_post']].copy()
    
    merged = bulks_sel.merge(segs_consensus, on=['sample','CHROM','seg'])
    haplotypes = merged.loc[:,['CHROM', 'seg_cons', 'cnv_state', 'snp_id', 'haplo_post']].rename(columns={'seg_cons': 'seg'})

    return haplotypes


def get_allele_post(
    df_allele: pd.DataFrame,
    haplotypes: pd.DataFrame,
    segs_consensus: pd.DataFrame
    ) -> pd.DataFrame:
    """
    Compute allele-based posterior probabilities per cell and segment.

    This function aggregates SNP-level allele counts into segment-level totals
    and computes binomial log-likelihood terms and posterior probabilities for
    CNV states using priors from segs_consensus.

    Parameters
    ----------
    df_allele : pandas.DataFrame
        SNP-level counts per cell. Must include:
        - cell: cell identifier
        - CHROM: chromosome name
        - snp_id: SNP identifier matching haplotypes.snp_id
        - POS: genomic SNP position (integer)
        - GT: phased genotype string (e.g., '1|0')
        - AD: alt allele read count (integer)
        - DP: total read depth (integer)
    haplotypes : pandas.DataFrame
        Per-SNP haplotype annotations. Must include:
        - CHROM, seg, cnv_state, snp_id
        - haplo_post: 'major' or 'minor' indicating the haplotype’s phase
    segs_consensus : pandas.DataFrame
        Segment-level priors. Must include:
        - seg_cons, seg_start, seg_end
        - p_loh, p_amp, p_del, p_bamp, p_bdel

    Returns
    -------
    pandas.DataFrame
        A DataFrame indexed by (cell, CHROM, seg, cnv_state) with:
        - major, minor, total, MAF
        - prior_loh, prior_amp, prior_del, prior_bamp, prior_bdel
        - binomial log-likelihood terms (l11, l10, l01, l20, l02, l21, l12, l31, l13, l32, l22, l00)
        - posterior outputs from compute_posterior (e.g., p_amp, p_del, p_loh, p_bamp, p_bdel, p_neu, p_cnv, logBF)
        - seg_label of the form "{seg}({cnv_state})"

    Notes
    -----
    - Rows with cnv_state == 'neu' are removed before aggregation.
    - SNPs are thinned within (cell, CHROM) by keeping only rows where inter_snp_dist > 250
      or the distance is missing (first SNP in a run).
    - MAF is computed as major / total. If DP or total are zero for a group, MAF may be NaN/inf.
    - Requires scipy.stats.binom and compute_posterior to be available in scope.

    Raises
    ------
    KeyError
        If any of the required columns listed above are missing.
    """

    # Compute pAD based on genotype: if GT == '1|0' then pAD = AD, else pAD = DP - AD.
    allele_counts = df_allele.copy()
    allele_counts['pAD'] = np.where(allele_counts['GT'] == '1|0',
                                    allele_counts['AD'],
                                    allele_counts['DP'] - allele_counts['AD'])
    # Inner join with haplotypes (only relevant columns)
    haplo_sel = haplotypes.loc[:,['CHROM', 'seg', 'cnv_state', 'snp_id', 'haplo_post']]
    allele_counts = allele_counts.merge(haplo_sel, on=['CHROM', 'snp_id'], how='inner')
    # Filter rows where cnv_state is 'neu'
    allele_counts = allele_counts[allele_counts['cnv_state'] != 'neu']
    # Compute major and minor allele counts and MAF.
    allele_counts['major_count'] = np.where(allele_counts['haplo_post'] == 'major',
                                            allele_counts['AD'],
                                            allele_counts['DP'] - allele_counts['AD'])
    
    #warnings.filterwarnings('ignore')
    allele_counts['minor_count'] = allele_counts['DP'] - allele_counts['major_count']
    allele_counts['MAF'] = allele_counts['major_count'] / allele_counts['DP']
    #warnings.filterwarnings('always')

    allele_counts['n_chrom_snp'] = allele_counts.groupby(['cell', 'CHROM'], sort=False, observed=True)['POS'].transform('count')
    allele_counts['inter_snp_dist'] = allele_counts.groupby(['cell', 'CHROM'], sort=False, observed=True)['POS'].diff()
    # Filter rows where inter_snp_dist > 250 or is NA.
    allele_counts = allele_counts[(allele_counts['inter_snp_dist'] > 250) | (allele_counts['inter_snp_dist'].isna())]
    # Summarise by grouping over 'cell', 'CHROM', 'seg', and 'cnv_state'
    allele_post = allele_counts.groupby(['cell', 'CHROM', 'seg', 'cnv_state'],
                                         observed=True,
                                         sort=False,
                                         as_index=False).agg(major=('major_count', 'sum'),
                                                             minor=('minor_count', 'sum'))
    allele_post['total'] = allele_post['major'] + allele_post['minor']
    allele_post['MAF'] = allele_post['major'] / allele_post['total']

    # Merge join with segs_consensus.
    segs_cons_temp = segs_consensus.loc[:,['seg_cons', 'seg_start', 'seg_end', 'p_loh', 'p_amp', 'p_del', 'p_bamp', 'p_bdel']].copy()
    segs_cons_temp = segs_cons_temp.rename(columns={'seg_cons':'seg',
                                                    'p_loh':'prior_loh',
                                                    'p_amp':'prior_amp',
                                                    'p_del':'prior_del',
                                                    'p_bamp':'prior_bamp',
                                                    'p_bdel':'prior_bdel'})
    allele_post = allele_post.merge(segs_cons_temp, on='seg')
    
    # Rowwise compute log-likelihood values using the binomial log-PMF.
    def compute_ll(row):
        major = row['major']
        total = row['total']
        row['l11'] = binom.logpmf(major, total, 0.5)
        row['l10'] = binom.logpmf(major, total, 0.9)
        row['l01'] = binom.logpmf(major, total, 0.1)
        row['l20'] = binom.logpmf(major, total, 0.9)
        row['l02'] = binom.logpmf(major, total, 0.1)
        row['l21'] = binom.logpmf(major, total, 0.66)
        row['l12'] = binom.logpmf(major, total, 0.33)
        row['l31'] = binom.logpmf(major, total, 0.75)
        row['l13'] = binom.logpmf(major, total, 0.25)
        row['l32'] = binom.logpmf(major, total, 0.6)
        row['l22'] = row['l11']
        row['l00'] = row['l11']
        return row
    
    allele_post = allele_post.apply(compute_ll, axis=1)
    # Compute the overall posterior probabilities.
    # warnings.filterwarnings('ignore')
    allele_post = compute_posterior(allele_post)
    # warnings.filterwarnings('always')
    # Create a seg_label by concatenating seg and cnv_state.
    allele_post['seg_label'] = allele_post['seg'].astype("string") + "(" + allele_post['cnv_state'].astype("string") + ")"
    
    return allele_post


def get_joint_post(
    exp_post: pd.DataFrame,
    allele_post: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    count_mat: Optional[ad.AnnData] = None,
    spatial: bool = False,
    method: Literal["degree", "weighted", "diffuse", "cpr"] = "cpr",
    connectivity_key: str = "spatial_connectivities",
    distance_key: str = "weighted_adjacency",
    method_kwargs: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Build a joint CNV posterior by combining expression- and allele-level posteriors,
    with optional spatial smoothing on a per-segment basis.

    Workflow
    --------
    1) Select and copy relevant columns from `exp_post` (excluding 'neu' states)
       and from `allele_post`.
    2) (Optional) Perform neighborhood smoothing within each segment (`by=['seg']`)
       using `neighbors_average`, operating on the columns needed to recompute
       posteriors.
    3) Outer-merge the (smoothed) expression and allele tables on
       ['cell','CHROM','seg','cnv_state'] and fill NaNs for *_x/*_y with zeros.
    4) Left-join per-segment priors and metadata from `segs_consensus`.
    5) Sum *_x and *_y into joint log-likelihoods (l11, l20, l10, l21, l31, l22, l00),
       recompute posterior probabilities via `compute_posterior`, and derive
       MLE and MAP CNV states.
    6) Add a state label 'seg_label' as "{seg}({cnv_state})".

    Parameters
    ----------
    exp_post
        Expression-level posterior DataFrame. Expected to contain at least:
        {'cell','CHROM','seg','cnv_state','l11','l20','l10','l21','l31','l22','l00',
         'Z','Z_cnv','Z_n','logBF'}. Rows with 'cnv_state' == 'neu' are removed. CHECK THIS BY NOT REMOVING NEUTRALS
    allele_post
        Allele-level posterior DataFrame. Expected columns include the above plus
        {'MAF','major','total'} (where 'total' is renamed to 'n_snp' later).
    segs_consensus
        Per-segment consensus/prior information. Expected columns include:
        {'seg_cons','seg_start','seg_end'} and optionally
        {'n_genes','n_snps','p_loh','p_amp','p_del','p_bamp','p_bdel','LLR','LLR_x','LLR_y'}.
        The probabilities are renamed to {'prior_loh','prior_amp','prior_del',
        'prior_bamp','prior_bdel'}.
    count_mat
        AnnData holding spatial neighbor graphs in `obsp`. Required if `spatial=True`
        and the chosen `method` in `neighbors_average` needs access to those graphs.
    spatial
        If True, apply neighborhood smoothing within each segment before merging.
    method
        Smoothing method passed to `neighbors_average` when `spatial=True`.
        One of {"degree","weighted","diffuse","cpr"}.
    distance_key
        Key in `count_mat.obsp` for the distance matrix (used by certain methods).
    method_kwargs
        Extra keyword arguments forwarded to `neighbors_average`. For example,
        for diffuse/cpr you might pass {'alpha': 0.75, 'steps': 15, ...}.
        (Note: this function keeps the provided default unchanged.)

    Returns
    -------
    joint_post_sp
        DataFrame with joint posteriors and derived states. Includes:
        - Joint log-likelihoods: {l11,l20,l10,l21,l31,l22,l00}
        - Posterior probabilities from `compute_posterior`: e.g., p_neu, p_loh, ...
        - Logistic transforms of logBF_x/logBF_y: {p_cnv_x, p_cnv_y}
        - State calls: {'cnv_state_mle','cnv_state_map'}
        - Seg labels: 'seg_label' as "{seg}({cnv_state})"
        - Segment metadata/priors from `segs_consensus`.

    Caveats / Potential Improvements
    --------------------------------

    - Key column presence is not validated explicitly; adding schema checks could
      yield clearer errors when inputs are missing required fields.
    - The per-row `apply(compute_states, axis=1)` can be a bottleneck; vectorizing
      the MLE/MAP computation would speed up large datasets.
    """

    # Process expression posteriors
    exp_sel = exp_post[exp_post['cnv_state'] != 'neu'].copy()
    exp_col_select = {'cell', 'CHROM', 'seg', 'cnv_state', 'l11', 'l20', 'l10', 'l21', 
                      'l31', 'l22', 'l00', 'Z', 'Z_cnv', 'Z_n', 'logBF'}
    exp_sel = exp_sel.loc[:,[col for col in exp_sel.columns if col in exp_col_select]].copy()
    
    # Process allele posteriors
    allele_col_select = {'cell', 'CHROM', 'seg', 'cnv_state', 'l11', 'l20', 'l10', 'l21', 'l31',
                        'l22', 'l00', 'Z', 'Z_cnv', 'Z_n', 'logBF', 'MAF', 'major', 'total'}
    allele_sel = allele_post.loc[:, [col for col in allele_post.columns if col in allele_col_select]].copy()
    allele_sel = allele_sel.rename({'total' : 'n_snp'})
    
    if spatial:

        exp_sel = spatial_utils.neighbors_average(
            df=exp_sel,
            adata=count_mat,
            columns=['l11', 'l20', 'l10', 'l21', 'l31', 'l22', 'l00','Z', 'Z_cnv', 'Z_n', 'logBF'],
            by=['seg'],
            method=method,
            method_kwargs=method_kwargs,
            connectivity_key=connectivity_key,
            distance_key=distance_key)
    
        allele_sel = spatial_utils.neighbors_average(
            df=allele_sel,
            adata=count_mat,
            columns=['l11', 'l20', 'l10', 'l21', 'l31', 'l22', 'l00',
                     'Z', 'Z_cnv', 'Z_n', 'logBF', 'MAF', 'major', 'total'],
            by=['seg'],
            method=method,
            method_kwargs=method_kwargs,
            connectivity_key=connectivity_key,
            distance_key=distance_key)
    
    # join exp_sel and allele_sel on keys: cell, CHROM, seg, cnv_state.
    joint_post_sp = pd.merge(exp_sel, allele_sel, on=['cell', 'CHROM', 'seg', 'cnv_state'], how='outer')
    # Replace NA values in all columns ending with _x or _y with 0. These are {'l*', 'Z*', 'logBF' }
    for col in joint_post_sp.columns:
       if col.endswith('_x') or col.endswith('_y'):
           joint_post_sp[col] = joint_post_sp[col].fillna(0)
    
    # Left join with segs_consensus
    segs_sel = segs_consensus.loc[:,['seg_cons', 'seg_start', 'seg_end'] +
                                                 [col for col in segs_consensus if col in {'n_genes',
                                                                                           'n_snps',
                                                                                           'p_loh',
                                                                                           'p_amp',
                                                                                           'p_del',
                                                                                           'p_bamp',
                                                                                           'p_bdel',
                                                                                           'LLR', 
                                                                                           'LLR_x', 
                                                                                           'LLR_y'}]].copy()
    
    segs_sel = segs_sel.rename(columns={'seg_cons':'seg',
                                                       'p_loh':'prior_loh',
                                                       'p_amp':'prior_amp',
                                                       'p_del':'prior_del',
                                                       'p_bamp':'prior_bamp',
                                                       'p_bdel':'prior_bdel'})
    
    joint_post_sp = pd.merge(joint_post_sp, segs_sel, on='seg', how='left')
    
    ## ADD SPATIAL CONTEXT    
    # Compute new joint log-likelihood columns by summing the _x and _y columns
    for col in ['l11', 'l20', 'l10', 'l21', 'l31', 'l22', 'l00']:
       joint_post_sp[col] = joint_post_sp[f'{col}_x'] + joint_post_sp[f'{col}_y']
    
    # Compute the joint posterior # TODO: fix warnings
    warnings.filterwarnings('ignore')
    joint_post_sp = compute_posterior(joint_post_sp)
    warnings.filterwarnings('always')
    
    # Compute logistic transformations for logBF values.
    joint_post_sp['p_cnv_x'] = 1 / (1 + np.exp(-joint_post_sp['logBF_x']))
    joint_post_sp['p_cnv_y'] = 1 / (1 + np.exp(-joint_post_sp['logBF_y']))
    
    # For each row, compute maximum likelihood CNV state and MAP CNV state.
    # Define state labels for MLE and MAP.
    state_labels_mle = ['neu', 'loh', 'del', 'amp', 'amp', 'bamp']
    state_labels_map = ['neu', 'loh', 'del', 'amp', 'bamp']
    
    def compute_states(row):
        # MLE
        values_mle = [row['l11'], row['l20'], row['l10'], row['l21'], row['l31'], row['l22']]
        idx_mle = np.argmax(values_mle)
        row['cnv_state_mle'] = state_labels_mle[idx_mle]
        # MAP
        values_map = [row.get('p_neu', 0), row.get('p_loh', 0), row.get('p_del', 0), row.get('p_amp', 0), row.get('p_bamp', 0)]
        idx_map = np.argmax(values_map)
        row['cnv_state_map'] = state_labels_map[idx_map]
        return row
    
    joint_post_sp = joint_post_sp.apply(compute_states, axis=1) # bottleneck
    # Create seg_label from seg and cnv_state.
    joint_post_sp['seg_label'] = joint_post_sp['seg'].astype(str) + "(" + joint_post_sp['cnv_state'].astype(str) + ")"

    return joint_post_sp


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Compute the element-wise binary entropy H(p) = -p log2 p - (1 - p) log2(1 - p).

    Parameters
    ----------
    p : np.ndarray
        Array of probabilities in the range [0, 1]. Values outside this range
        are not checked and may produce nonsensical results.

    Returns
    -------
    np.ndarray
        Array of the same shape as `p` with the corresponding binary entropy
        values. NaN entries produced by 0 * log2(0) or similar expressions are
        replaced by 0 in the output.
    """
    H = -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    H[np.isnan(H)] = 0
    return H


def joint_post_entropy(joint_post: pd.DataFrame) -> pd.Series:
    """
    Compute per-segment mean binary entropy of the CNV posterior probability.

    For each segment (seg), this function computes the binary entropy of the
    `p_cnv` posterior probabilities across rows in that segment and assigns
    the segment-wise mean entropy to all rows belonging to that segment.

    Parameters
    ----------
    joint_post : pd.DataFrame
        DataFrame containing at least the following columns:
        - 'seg' : segment identifier used for grouping.
        - 'p_cnv' : posterior probability of CNV (float), possibly with NaNs.

    Returns
    -------
    pd.Series
        A Series of dtype float64 indexed like `joint_post`, where each entry
        is the mean binary entropy of `p_cnv` for the corresponding segment.
    """
    binary_entropy_series = pd.Series(
        np.repeat(0.0, joint_post.shape[0]),
        index=joint_post.index,
        dtype=np.float64,
    )
    seg_group = joint_post.groupby(by="seg", observed=True, sort=False)
    for _, group in seg_group:
        binary_entropy_series[group.index] = np.mean(
            binary_entropy(group.p_cnv[group.p_cnv.notna()])
        )
    return binary_entropy_series


def expand_states(
    sc_post: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    ) -> pd.DataFrame:
    """
    Expand multi-allelic CNV states into separate rows in a single-cell posterior table.

    This function takes a per-cell CNV posterior table (`sc_post`) and a consensus
    segment table (`segs_consensus`) that may contain multi-allelic CNV calls.
    For segments with more than one possible CNV state (`n_states > 1`), the
    function generates one row per CNV state and attaches the corresponding
    posterior values from `sc_post` (for example, columns named "p_amp",
    "Z_amp", etc.). Segments that are not multi-allelic are passed through
    unchanged.

    Parameters
    ----------
    sc_post : pandas.DataFrame
        Single-cell posterior table. Expected to contain at least:
          - "cell": cell identifier.
          - "CHROM": chromosome identifier.
          - "seg": segment identifier matching the consensus segments.
        For multi-allelic expansion, it should also contain:
          - One or more probability columns named "p_<state>" for each CNV
            state listed in `segs_consensus["cnv_states"]`.
          - One or more score/latent columns named "Z_<state>" for each such
            state.
        It may optionally contain a pre-existing "cnv_state" column, which
        will be dropped before the multi-allelic expansion.

    segs_consensus : pandas.DataFrame
        Consensus CNV segment table. Expected to contain at least:
          - "seg_cons": consensus segment identifier (to be renamed to "seg").
          - "n_states": integer number of possible CNV states for that segment.
          - "cnv_states": string encoding one or more states separated by
            commas, for example "amp,del".

    Returns
    -------
    pandas.DataFrame
        A DataFrame of single-cell posteriors with multi-allelic segments
        expanded so that each CNV state has its own row. The result includes:
          - All columns from the input `sc_post` (except any dropped
            "cnv_state" before expansion).
          - For expanded rows, new columns:
              - "cnv_state": the CNV state label for that row.
              - "p_cnv": posterior probability for the CNV state.
              - "p_n": posterior probability for the normal state
                (1 - p_cnv when available).
              - "Z_cnv": state-specific score/latent value.
              - "n_states": the number of states for the underlying segment.
              - "seg_label": ordered categorical label combining segment and
                state for plotting or grouping.
    """
    # Expand segs_consensus for multi-allelic CNVs.
    segs_multi = (segs_consensus[segs_consensus['n_states'] > 1]
                  .loc[:, ['seg_cons', 'cnv_states', 'n_states']]
                  .rename(columns={'seg_cons': 'seg'}))
    
    # Split 'cnv_states' on commas and explode into separate rows.
    segs_multi = segs_multi.assign(cnv_states=segs_multi['cnv_states'].str.split(',')).explode('cnv_states')
    segs_multi = segs_multi.rename(columns={'cnv_states': 'cnv_state'})
    
    # If there are any multi-allelic segments, process them.
    if (segs_consensus['n_states'] > 1).any():
        # Create sc_post_multi by dropping the 'cnv_state' column and inner joining with segs_multi on 'seg'.
        sc_post_multi = sc_post.drop(columns=['cnv_state'], errors='ignore').merge(
            segs_multi,
            on='seg',
            how='inner'
        )
        # Append the cnv_state to the seg identifier.
        sc_post_multi['seg'] = sc_post_multi['seg'].astype(str) + '_' + sc_post_multi['cnv_state'].astype(str)
        
        # For each row, dynamically select the posterior values based on cnv_state.
        # This assumes that sc_post contains columns named like 'p_amp', 'Z_amp', etc.
        def select_posteriors(row):
            state = row['cnv_state']
            # Retrieve the posterior value from the column 'p_{state}' if it exists; default to NaN otherwise.
            p_col = f"p_{state}"
            z_col = f"Z_{state}"
            row['p_cnv'] = row.get(p_col, np.nan)
            row['p_n'] = 1 - row['p_cnv'] if pd.notna(row['p_cnv']) else np.nan
            row['Z_cnv'] = row.get(z_col, np.nan)
            return row

        sc_post_multi = sc_post_multi.apply(select_posteriors, axis=1)
        
        # Filter out rows from sc_post whose seg is present in segs_multi (unexpanded version).
        sc_post_filtered = sc_post[~sc_post['seg'].isin(segs_multi['seg'])]
        sc_post_filtered = sc_post_filtered.copy()
        sc_post_filtered['n_states'] = 1
        
        # Concatenate the filtered sc_post with sc_post_multi.
        sc_post = pd.concat([sc_post_filtered, sc_post_multi], ignore_index=True)
        
        # Sort by 'cell', 'CHROM', and 'seg'.
        sc_post = sc_post.sort_values(by=['cell', 'CHROM', 'seg'])
        # Create seg_label by concatenating seg and cnv_state.
        sc_post['seg_label'] = sc_post['seg'].astype(str) + "(" + sc_post['cnv_state'].astype(str) + ")"
        # Convert seg_label to a categorical preserving order of appearance.
        unique_labels = sc_post['seg_label'].drop_duplicates().tolist()
        sc_post['seg_label'] = pd.Categorical(sc_post['seg_label'], categories=unique_labels, ordered=True)
    else:
        log.info("No multi-allelic CNVs, skipping expansion.")
    
    return sc_post


def get_joint_post_matrix(joint_post_filtered: pd.DataFrame, p_min: float) -> np.ndarray:
    """
    Build a cell-by-segment posterior probability table from joint posterior data.

    This function takes a long-format joint posterior DataFrame and produces a
    matrix-like table of posterior CNV probabilities.
    
    
    Parameters
    ----------
    joint_post_filtered : pd.DataFrame
        A DataFrame containing joint posterior data with at least the following columns:
          - 'cell': Identifier for each cell.
          - 'seg': Segment identifier.
          - 'p_cnv': Posterior probability for CNV state.
    p_min : float
        The minimum threshold for p_cnv. p_cnv values will be clamped to the interval 
        [p_min, 1 - p_min].
    
    Returns
    -------
    pandas.DataFrame
        A 2D DataFrame where:
            - Rows correspond to cells (index is "cell").
            - Columns correspond to segments (column labels are values of "seg").
            - Entries are clamped CNV posterior probabilities in
              the range [p_min, 1 - p_min].
        Missing cell–segment combinations are filled with 0.5.
    """
    
    df = joint_post_filtered.copy()
    
    # Clamp 'p_cnv' values.
    df['p_cnv'] = df['p_cnv'].clip(lower=p_min, upper=1 - p_min)
    
    # Reshape DataFrame.
    pivot_df = df.pivot(index='cell', columns='seg', values='p_cnv').fillna(0.5)
    
    return pivot_df


def _lse(vals: np.ndarray) -> float:
    """Log-sum-exp"""
    vals = np.asarray(vals, dtype=float)
    return _log_sum_exp(vals)


def get_clone_post(gtree: nx.DiGraph,
                   exp_post: pd.DataFrame,
                   allele_post: pd.DataFrame) -> pd.DataFrame:
    """
    Map cells to clones (tree genotypes) using expression and allele posteriors.

    Parameters
    ----------
    gtree : nx.DiGraph
        Single-cell lineage tree (nodes need attrs: GT, clone, compartment, leaf).
    exp_post, allele_post : pd.DataFrame
        Must contain columns: 'cell', 'seg', 'cnv_state', 'Z_cnv', 'Z_n'.

    Returns
    -------
    pd.DataFrame
        Wide table with per-clone posteriors per cell and tumor/normal summary.
    """

    # Collect clones from the tree
    nodes_df = pd.DataFrame([{**{"_id": n}, **d} for n, d in gtree.nodes(data=True)])
    # Make sure the required columns exist (fill missing with defaults)
    for col, default in (("GT", ""), ("clone", np.nan), ("compartment", "normal"), ("leaf", False)):
        if col not in nodes_df.columns:
            nodes_df[col] = default

    # Group by (GT, clone, compartment); clone_size = sum(leaf)
    clones = (nodes_df
              .groupby(["GT", "clone", "compartment"], dropna=False, as_index=False, sort=False)
              .agg(clone_size=("leaf", lambda x: int(np.asarray(x, dtype=bool).sum()))))

    # If normal GT ('') missing, add it with clone id 0 and size 0
    if not ((clones["GT"] == "")).any():
        clones = pd.concat([
            pd.DataFrame([{"GT": "", "clone": 0, "compartment": "normal", "clone_size": 0}]),
            clones
        ], ignore_index=True)

    # Priors per clone (depends on unique GTs)
    unique_GTs = sorted(pd.unique(clones["GT"]))
    k_non_normal = sum(gt != "" for gt in unique_GTs)
    if k_non_normal == 0:
        # Only normal present
        prior_map = { "": 1.0 }
    else:
        prior_map = { "": 0.5, **{gt: 0.5 / k_non_normal for gt in unique_GTs if gt != ""} }

    clones["prior_clone"] = clones["GT"].map(prior_map).astype(float)

    # Build clone-segment table with indicator I
    # Universe of segments = those appearing in any clone GT (split by commas)
    def _split_gt(gt: str) -> List[str]:
        gt = gt or ""
        return [s for s in (i.strip() for i in gt.split(",")) if s]

    # For each clone row, get its set of segments
    clones["seg_list"] = clones["GT"].apply(_split_gt)
    # All segs seen across clones (if none, this stays empty)
    seg_universe = sorted(set(sum(clones["seg_list"].tolist(), [])))

    # Expand to all (clone × seg) with I = 1 if seg in clone GT, else 0
    # Start with rows that have seg in seg_list (I=1)
    rows_ones = []
    for _, r in clones.iterrows():
        for seg in r["seg_list"]:
            rows_ones.append({
                "GT": r["GT"],
                "clone": r["clone"],
                "compartment": r["compartment"],
                "prior_clone": r["prior_clone"],
                "clone_size": r["clone_size"],
                "seg": seg,
                "I": 1
            })
    clone_segs = pd.DataFrame(rows_ones, columns=["GT","clone","compartment","prior_clone","clone_size","seg","I"])

    # Now complete to all segs for each clone tuple, filling I=0 (and drop seg == '')
    if seg_universe:
        clone_keys = clones[["GT","clone","compartment","prior_clone","clone_size"]].drop_duplicates()
        # Cartesian product
        full = (clone_keys.assign(_tmp=1)
                .merge(pd.DataFrame({"seg": seg_universe, "_tmp": 1}), on="_tmp")
                .drop(columns="_tmp"))
        clone_segs = (full
                      .merge(clone_segs, how="left",
                             on=["GT","clone","compartment","prior_clone","clone_size","seg"]))
        clone_segs["I"] = clone_segs["I"].fillna(0).astype(int)
    # Filter seg != '' (should already be true)
    clone_segs = clone_segs[clone_segs["seg"] != ""].copy()

    # Join exp/allele posteriors and sum per (cell, clone, GT, prior)
    def _collapse_side(df_side: pd.DataFrame, side: str) -> pd.DataFrame:
        # Filter non-neutral and inner-join by seg with clone_segs
        df = df_side[df_side["cnv_state"] != "neu"].copy()
        if df.empty or clone_segs.empty:
            # return empty with correct columns
            return pd.DataFrame(columns=["cell","clone","GT","prior_clone",f"l_clone_{side}"])
        df = (df.merge(clone_segs, how="inner", left_on="seg", right_on="seg"))
        # l_clone = Z_cnv if I==1 else Z_n
        z = np.where(df["I"] == 1, df["Z_cnv"].to_numpy(), df["Z_n"].to_numpy())
        df[f"l_clone_{side}"] = z
        return (df.groupby(["cell","clone","GT","prior_clone"], as_index=False, sort=False)
                  .agg(**{f"l_clone_{side}": (f"l_clone_{side}", "sum")}))

    l_x = _collapse_side(exp_post, "x")
    l_y = _collapse_side(allele_post, "y")

    # Full outer join on the keys, fill NA with 0
    join_keys = ["cell","clone","GT","prior_clone"]
    clone_post = (pd.merge(l_x, l_y, how="outer", on=join_keys)
                    .fillna({ "l_clone_x": 0.0, "l_clone_y": 0.0 }))

    # If either side is entirely empty, we still want all combinations
    # of (cell × clones) present in the data.
    if clone_post.empty:
        # No informative CNV segments matched; return empty DataFrame with expected columns
        out_cols = ["cell","clone_opt","GT_opt","p_opt","p_cnv","p_cnv_x","p_cnv_y","compartment_opt"]
        return pd.DataFrame(columns=out_cols)

    # Per-cell normalization to probabilities (softmax in log-space)
    def _per_cell_probs(g: pd.DataFrame) -> pd.DataFrame:
        # Z's (with prior)
        z_clone   = np.log(g["prior_clone"].to_numpy()) + g["l_clone_x"].to_numpy() + g["l_clone_y"].to_numpy()
        z_clone_x = np.log(g["prior_clone"].to_numpy()) + g["l_clone_x"].to_numpy()
        z_clone_y = np.log(g["prior_clone"].to_numpy()) + g["l_clone_y"].to_numpy()

        lse   = _lse(z_clone)
        lse_x = _lse(z_clone_x)
        lse_y = _lse(z_clone_y)

        p   = np.exp(z_clone   - lse)
        p_x = np.exp(z_clone_x - lse_x)
        p_y = np.exp(z_clone_y - lse_y)

        g = g.copy()
        g["Z_clone"]   = z_clone
        g["Z_clone_x"] = z_clone_x
        g["Z_clone_y"] = z_clone_y
        g["p"]   = p
        g["p_x"] = p_x
        g["p_y"] = p_y

        # argmax over p
        i = int(np.argmax(p))
        g["clone_opt"] = g["clone"].iloc[i]
        g["GT_opt"]    = g["GT"].iloc[i]
        g["p_opt"]     = p[i]
        return g

    clone_post = (clone_post
                  .groupby("cell", group_keys=False, sort=False)
                  .apply(_per_cell_probs))

    # p_* columns per clone
    # Build tidy array for pivot: value columns p, p_x, p_y
    tidy = clone_post[["cell","clone","p","p_x","p_y","clone_opt","GT_opt","p_opt"]].copy()
    wide_p   = tidy.pivot_table(index=["cell","clone_opt","GT_opt","p_opt"],
                                columns="clone", values="p",   fill_value=0.0)
    wide_px  = tidy.pivot_table(index=["cell","clone_opt","GT_opt","p_opt"],
                                columns="clone", values="p_x", fill_value=0.0)
    wide_py  = tidy.pivot_table(index=["cell","clone_opt","GT_opt","p_opt"],
                                columns="clone", values="p_y", fill_value=0.0)

    # Flatten MultiIndex columns with prefixes p_, p_x_, p_y_
    wide_p.columns  = [f"p_{c}"   for c in wide_p.columns]
    wide_px.columns = [f"p_x_{c}" for c in wide_px.columns]
    wide_py.columns = [f"p_y_{c}" for c in wide_py.columns]

    clone_post_wide = (wide_p
                       .join(wide_px, how="outer")
                       .join(wide_py, how="outer")
                       .reset_index())

    # Tumor-vs-normal summary
    # Tumor clone IDs come from the clones table (compartment == 'tumor')
    tumor_clones = clones.loc[clones["compartment"] == "tumor", "clone"].dropna().astype(int).unique().tolist()

    def _sum_cols(df: pd.DataFrame, stem: str, ids: List[int]) -> np.ndarray:
        cols = [f"{stem}{i}" for i in ids]
        cols = [c for c in cols if c in df.columns]   # only existing columns
        if not cols:
            return np.zeros(len(df), dtype=float)
        return df[cols].sum(axis=1).to_numpy()

    clone_post_wide["p_cnv"]   = _sum_cols(clone_post_wide, "p_",   tumor_clones)
    clone_post_wide["p_cnv_x"] = _sum_cols(clone_post_wide, "p_x_", tumor_clones)
    clone_post_wide["p_cnv_y"] = _sum_cols(clone_post_wide, "p_y_", tumor_clones)

    clone_post_wide["compartment_opt"] = np.where(clone_post_wide["p_cnv"] > 0.5, "tumor", "normal")

    return clone_post_wide



