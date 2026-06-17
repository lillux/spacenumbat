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
from scipy.special import expit

from joblib import cpu_count, Parallel, delayed
from numba import njit, prange

import networkx as nx
import pyranges as pr
import natsort

import anndata as ad

from . import utils, dist_prob, clustering, _progressbar, spatial_utils, hmrf, numeric

#import warnings

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
    
    if bulks is None:
        return pd.DataFrame()

    bulks = bulks.copy()
    if bulks.shape[0] == 0:
        if 'sample' not in bulks.columns:
            bulks['sample'] = pd.Series(dtype=object)
        return bulks

    if 'sample' not in bulks.columns:
        bulks['sample'] = '0'

    # Drop samples with no allele data
    bulks = bulks.groupby('sample', observed=True, sort=False).filter(lambda x: x['DP'].notna().sum() > 0).copy()

    if bulks.shape[0] == 0:
        return bulks

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
        Input segments table. Expected columns (no explicit validation):
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
      threshold (permissive).

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
            'End': segs_all['seg_end_index'] + 1,
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
    
    df_ov['len_x'] = df_ov['end_x'] - df_ov['start_x'] + 1
    df_ov['len_y'] = df_ov['end_y'] - df_ov['start_y'] + 1
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
    bulks = (bulks.groupby(groupcols, group_keys=False, observed=True, sort=False)[bulks.columns]
             .apply(seg_start_end_aggregator)
             .reset_index(drop=True))
    bulks = bulks[bulks['seg_start'] != bulks['seg_end']]
    
    segs_all = bulks[info_cols].drop_duplicates().copy()
    segs_all.cnv_state = np.where(
        (segs_all['LLR'].isna() | (segs_all['LLR']<min_LLR)), 
        "neu",
        segs_all.cnv_state)
    segs_star = segs_all[segs_all['cnv_state']!='neu'].copy()

    segs_star = resolve_cnvs(segs_star, min_overlap=min_overlap, debug=False)
    
    if retest:
        segs_cnv = segs_all[segs_all['cnv_state']!='neu'].copy()
        
        if segs_cnv.shape[0] == 0:
            df_retest = pd.DataFrame(columns=["CHROM", "seg_start", "seg_end", "cnv_state", "cnv_state_post"])
        else:
            segs_cnv = segs_cnv.sort_values("CHROM", key=natsort.natsort_keygen())
            # build PyRanges from segs_cnv
            pr_cnv = pr.PyRanges(
                pd.DataFrame({'Chromosome': segs_cnv['CHROM'],
                              'Start': segs_cnv['seg_start'],
                              'End': segs_cnv['seg_end'] + 1
                              })
                ).merge()
            
            # if segs_star empty -> return segs_cnv
            if segs_star.shape[0] == 0:
                pr_retest = pr_cnv
            else:
                # build PyRanges from segs_star
                pr_star = pr.PyRanges(
                    pd.DataFrame({'Chromosome': segs_star['CHROM'],
                                  'Start': segs_star['seg_start'],
                                  'End': segs_star['seg_end'] + 1
                                  })
                    ).merge()
            
                # find segments in between CNVs regions
                pr_retest = pr_cnv.subtract(pr_star)
                
            df_retest = pr_retest.as_df()
            if df_retest.shape[0] == 0:
                    df_retest = pd.DataFrame(columns=["CHROM", "seg_start", "seg_end", "cnv_state", "cnv_state_post"])
            else:
                df_retest['End'] = df_retest['End'] - 1
                df_retest = df_retest[(df_retest['End'] - df_retest['Start']) > 0]
                # add cnv_state 'retest'
                df_retest['cnv_state'] = 'retest'
                df_retest['cnv_state_post'] = 'retest'
                df_retest = df_retest.rename(columns={'Chromosome':'CHROM','Start':'seg_start','End':'seg_end'})
                df_retest.CHROM = df_retest.CHROM.astype("string") # TODO: just added 19/02/2026
        
    else:
        df_retest = pd.DataFrame()
    
    # union of neutral segments
    segs_neu_input = segs_all[segs_all['cnv_state']=='neu'].sort_values("CHROM", key=natsort.natsort_keygen())
    
    if segs_neu_input.shape[0] == 0:
        df_neu = pd.DataFrame(columns=["CHROM", "seg_start", "seg_end", "seg_length"])
    else:
    
        pr_neu = pr.PyRanges(
            pd.DataFrame({'Chromosome': segs_neu_input['CHROM'],
                          'Start': segs_neu_input['seg_start'],
                          'End': segs_neu_input['seg_end'] + 1
                          })
            ).merge()
        
        df_neu = pr_neu.as_df()
        df_neu['End'] = df_neu['End'] - 1
        df_neu = df_neu.rename(columns={'Chromosome':'CHROM','Start':'seg_start','End':'seg_end'})
        df_neu['seg_length'] = df_neu['seg_end'] - df_neu['seg_start'] + 1
        df_neu.CHROM = df_neu.CHROM.astype("string") # TODO: just added 19/02/2026
    
    # if all segs_all['cnv_state'] == 'neu'
    if (segs_all['cnv_state']!='neu').sum() == 0:
        df_neu = df_neu.sort_values(by='CHROM', key=natsort.natsort_keygen())
        def assign_seg(rows):
            rows = rows.copy()
            n_ = len(rows)
            postfix = utils.generate_postfix(range(n_))
            rows["seg"] = [f"{chrom}{pfx}" for chrom, pfx in zip(rows["CHROM"].astype(str), postfix)]
            rows['cnv_state'] = 'neu'
            rows['cnv_state_post'] = 'neu'
            return rows
        
        if df_neu.shape[0] == 0:
            return df_neu.assign(seg=pd.Series(dtype="object"), cnv_state="neu", cnv_state_post="neu")
    
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
            'End': gtf_temp['gene_end'] + 1,
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
            'End': segs_temp['seg_end'] + 1,
            'seg_index': segs_temp['seg_index']
        })
    )
    # Put seg index on 
    ov = pr_genes.join(pr_segs).as_df()
    
    df_ov = ov.rename(columns={'Chromosome':'CHROM','Start':'gene_start', 'End':'gene_end','Start_b':'seg_start','End_b':'seg_end'})
    df_ov['gene_end'] = df_ov['gene_end'] - 1
    df_ov['seg_end'] = df_ov['seg_end'] - 1
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
    
    log.info('Excluding clonal LOH regions ..')
    
    pr_genes = pr.PyRanges(pd.DataFrame({'Chromosome': exp_sc.var['CHROM'],
                                         'Start': exp_sc.var['gene_start'],
                                         'End': exp_sc.var['gene_end'] + 1,
                                         'gene_index': exp_sc.var['gene_index']}))
    
    pr_loh = pr.PyRanges(pd.DataFrame({'Chromosome': segs_loh['CHROM'],
                                       'Start': segs_loh['seg_start'],
                                       'End': segs_loh['seg_end'] + 1,
                                       'loh_index': np.arange(segs_loh.shape[0])}))
    
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
    n_points: int = 256,
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
                         #disp=disp,
                         n_points=n_points)
        mu = fit[0]
        sigma = fit[1]

    # Summarize for each (CHROM, seg, cnv_state)
    group_cols = ['CHROM','seg','cnv_state']
    def aggregator(df: pd.DataFrame) -> pd.Series:
        n = len(df)
        phi_mle_val = calc_phi_mle_lnpois(df['Y_obs'].values,
                                          df['lambda_ref'].values,
                                          depth_obs,
                                          mu,
                                          sigma,
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
        #options={'disp': disp},
        tol = 1e-5,
    )
    
    return res.x[0]


@njit
def _safe_log_scaled_prior(
    prior: float,
    divisor: float,
    ) -> float:
    """
    Return log(prior / divisor) without invalid logarithm operations.
    
    """
    if np.isnan(prior):
        return np.nan

    if prior < 0.0:
        return np.nan

    if prior == 0.0:
        return -np.inf

    if prior == np.inf:
        return np.inf

    return math.log(prior / divisor)


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
    prior_bdel: np.ndarray,
    ) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Compute posterior probabilities for CNV states safely in log space.

    The function preserves the existing input/output contract and output order.
    Missing or mathematically undefined inputs propagate as NaN. They are never
    replaced with zero.

    Zero priors remain valid and produce -inf log evidence and zero posterior
    probability when the total evidence is otherwise defined.
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

    log_half = math.log(0.5)

    for i in prange(n):
        # Amplification combines two underlying likelihood configurations.
        log_prior_amp = _safe_log_scaled_prior(
            prior_amp[i],
            4.0,
        )

        amp_21 = numeric.safe_add(
            l21[i],
            log_prior_amp,
        )
        amp_31 = numeric.safe_add(
            l31[i],
            log_prior_amp,
        )

        amp_values = np.empty(2, dtype=np.float64)
        amp_values[0] = amp_21
        amp_values[1] = amp_31

        z_amp = numeric.log_sum_exp(amp_values)

        z_loh = numeric.safe_add(
            l20[i],
            _safe_log_scaled_prior(prior_loh[i], 2.0),
        )

        z_del = numeric.safe_add(
            l10[i],
            _safe_log_scaled_prior(prior_del[i], 2.0),
        )

        z_bamp = numeric.safe_add(
            l22[i],
            _safe_log_scaled_prior(prior_bamp[i], 2.0),
        )

        z_bdel = numeric.safe_add(
            l00[i],
            _safe_log_scaled_prior(prior_bdel[i], 2.0),
        )

        z_neu = numeric.safe_add(
            l11[i],
            log_half,
        )

        Z_amp[i] = z_amp
        Z_loh[i] = z_loh
        Z_del[i] = z_del
        Z_bamp[i] = z_bamp
        Z_bdel[i] = z_bdel
        Z_n[i] = z_neu

        all_values = np.empty(6, dtype=np.float64)
        all_values[0] = z_neu
        all_values[1] = z_loh
        all_values[2] = z_del
        all_values[3] = z_amp
        all_values[4] = z_bamp
        all_values[5] = z_bdel

        z_total = numeric.log_sum_exp(all_values)
        Z[i] = z_total

        cnv_values = np.empty(5, dtype=np.float64)
        cnv_values[0] = z_loh
        cnv_values[1] = z_del
        cnv_values[2] = z_amp
        cnv_values[3] = z_bamp
        cnv_values[4] = z_bdel

        z_cnv = numeric.log_sum_exp(cnv_values)
        Z_cnv[i] = z_cnv

        # Posterior probabilities. The helper handles NaN and infinite
        # normalization constants without invalid subtraction.
        p_amp[i] = numeric.safe_exp_difference(z_amp, z_total)
        p_neu[i] = numeric.safe_exp_difference(z_neu, z_total)
        p_del[i] = numeric.safe_exp_difference(z_del, z_total)
        p_loh[i] = numeric.safe_exp_difference(z_loh, z_total)
        p_bamp[i] = numeric.safe_exp_difference(z_bamp, z_total)
        p_bdel[i] = numeric.safe_exp_difference(z_bdel, z_total)

        logBF[i] = numeric.safe_subtract(
            z_cnv,
            z_neu,
        )

        p_cnv[i] = numeric.safe_exp_difference(
            z_cnv,
            z_total,
        )

        p_n[i] = p_neu[i]

    return (
        Z_amp,
        Z_loh,
        Z_del,
        Z_bamp,
        Z_bdel,
        Z_n,
        Z,
        Z_cnv,
        p_amp,
        p_neu,
        p_del,
        p_loh,
        p_bamp,
        p_bdel,
        logBF,
        p_cnv,
        p_n,
    )


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
    ncores: int = 1,
    verbose: bool = True,
    use_pbar: bool = False,
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
    ncores : int, default 1
        Number of parallel workers used when computing per-cell likelihoods.
    verbose : bool, default True
        If True, prints progress and summaries.
    use_pbar : bool, default False
        If True, show an asynchronous joblib/tqdm progress bar while processing
        cells. If False, run in parallel without rendering a progress bar. This
        is intended to be controlled by the pipeline-level progress-bar switch.
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
            #sc_exp_data.loc[:,'logFC'] = np.log2(sc_exp_data.lambda_obs) - np.log2(sc_exp_data.lambda_ref)
            safe_lambda_obs = pd.Series(np.where(sc_exp_data.lambda_obs > 0, 
                                                 sc_exp_data.lambda_obs, 
                                                 np.nan), index=sc_exp_data.index)
            safe_lambda_ref = pd.Series(np.where(sc_exp_data.lambda_ref > 0,
                                                 sc_exp_data.lambda_ref,
                                                 np.nan), index=sc_exp_data.index)
            sc_exp_data.loc[:,'logFC'] = np.log2(safe_lambda_obs) - np.log2(safe_lambda_ref)
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
        
    n_jobs = int(np.max([1, np.min((len(cells), cpu_count(), ncores))]))
    parallel_kwargs = {
        "n_jobs": n_jobs,
        # `process_cell` closes over large AnnData/DataFrame objects. Using threads here
        # avoids repeatedly serializing those objects into child processes, which can
        # trigger worker crashes and excessive memory usage on large samples.
        "backend": "threading",
        # Keep dispatch bounded to avoid building up too many pending tasks/results
        # at once when cell count is large.
        "pre_dispatch": n_jobs,
    }
    
    if verbose:
        log.info(f'Running expression likelihood jobs on {n_jobs} core')

    if use_pbar:
        with _progressbar.tqdm_joblib(total=len(cells), desc="Processing cells", disable=not verbose):
            results = Parallel(**parallel_kwargs)(delayed(process_cell)(cell) for cell in cells)
    else:
        results = Parallel(**parallel_kwargs)(delayed(process_cell)(cell) for cell in cells)

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
    
    exp_post.CHROM = exp_post.CHROM.astype("string")
    exp_post.seg = exp_post.seg.astype("string")
    segs_consensus.CHROM = segs_consensus.CHROM.astype("string")
    segs_consensus.seg = segs_consensus.seg.astype("string")
    segs_consensus.seg_cons = segs_consensus.seg_cons.astype("string")
    
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
    #log.info('Disabling system warnings...')
    #warnings.filterwarnings('ignore')
    exp_posterior = compute_posterior(exp_post_merged)
    #warnings.filterwarnings('always')
    #log.info('System warnings enabled.')
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
    
    allele_counts['minor_count'] = allele_counts['DP'] - allele_counts['major_count']
    allele_counts['MAF'] = allele_counts['major_count'] / allele_counts['DP']
    
    allele_counts = allele_counts.sort_values(["cell", "CHROM", "POS"], key=natsort.natsort_keygen())
    allele_counts['n_chrom_snp'] = allele_counts.groupby(['cell', 'CHROM'], sort=False, observed=True)['POS'].transform('count')
    allele_counts['inter_snp_dist'] = allele_counts.groupby(['cell', 'CHROM'], sort=False, observed=True)['POS'].diff()
    # Filter rows where inter_snp_dist > 250 or is NA.
    allele_counts = allele_counts[(allele_counts['inter_snp_dist'] > 250) | (allele_counts['inter_snp_dist'].isna())]  # TODO: check skipping interval
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
    #warnings.filterwarnings('ignore')
    allele_post = compute_posterior(allele_post)
    #warnings.filterwarnings('always')
    # Create a seg_label by concatenating seg and cnv_state.
    allele_post['seg_label'] = allele_post['seg'].astype("string") + "(" + allele_post['cnv_state'].astype("string") + ")"
    
    return allele_post



def get_joint_post(
    exp_post: pd.DataFrame,
    allele_post: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    count_mat: Optional[ad.AnnData] = None,
    spatial: bool = False,
    method: Literal["hmrf", "degree", "diffuse", "cpr"] = "hmrf",
    connectivity_key: str = "spatial_connectivities",
    distance_key: str = "weighted_adjacency",
    method_kwargs: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
    """
    Combine expression and allele likelihoods and optionally apply spatial
    regularization.

    ``method="hmrf"`` preserves the local likelihoods and places a Potts prior
    on per-spot CNV states. The other methods retain the legacy likelihood
    smoothing behavior.
    """
    method_kwargs = {} if method_kwargs is None else dict(method_kwargs)
    method = method.lower()

    accepted_methods = {"hmrf", "degree", "diffuse", "cpr"}
    if method not in accepted_methods:
        raise ValueError(
            f"Unknown spatial method {method!r}. "
            f"Accepted methods are {sorted(accepted_methods)}."
        )

    if spatial and count_mat is None:
        raise ValueError("count_mat is required when spatial=True.")

    key_columns = ["cell", "CHROM", "seg", "cnv_state"]
    likelihood_columns = [
        "l11",
        "l20",
        "l10",
        "l21",
        "l31",
        "l22",
        "l00",
    ]

    # Expression likelihoods.
    exp_columns = {
        *key_columns,
        *likelihood_columns,
        "Z",
        "Z_cnv",
        "Z_n",
        "logBF",
    }

    exp_sel = exp_post.loc[
        exp_post["cnv_state"] != "neu",
        [column for column in exp_post.columns if column in exp_columns],
    ].copy()

    # Allele likelihoods.
    allele_columns = {
        *key_columns,
        *likelihood_columns,
        "Z",
        "Z_cnv",
        "Z_n",
        "logBF",
        "MAF",
        "major",
        "minor",
        "total",
    }

    allele_sel = allele_post.loc[
        :,
        [column for column in allele_post.columns if column in allele_columns],
    ].copy()

    # spatial smoothing.
    if spatial and method != "hmrf":
        
        exp_sel = spatial_utils.neighbors_average(
            df=exp_sel,
            adata=count_mat,
            columns=[
                *likelihood_columns,
                "Z",
                "Z_cnv",
                "Z_n",
                "logBF",
            ],
            by=["seg"],
            method=method,
            method_kwargs=method_kwargs,
            connectivity_key=connectivity_key,
            distance_key=distance_key,
        )

        allele_smoothing_columns = [
            *likelihood_columns,
            "Z",
            "Z_cnv",
            "Z_n",
            "logBF",
            #"MAF",
            #"major",
            #"minor",
            #"total",
        ]
        
        allele_smoothing_columns = [column for column in allele_smoothing_columns if column in allele_sel.columns]

        allele_sel = spatial_utils.neighbors_average(
            df=allele_sel,
            adata=count_mat,
            columns=allele_smoothing_columns,
            by=["seg"],
            method=method,
            method_kwargs=method_kwargs,
            connectivity_key=connectivity_key,
            distance_key=distance_key,
        )

    joint_post = pd.merge(exp_sel, allele_sel, on=key_columns, how="outer")

    # missing modality contributes zero additive log-likelihood.
    for suffix in ("x", "y"):
        for likelihood in likelihood_columns:
            column = f"{likelihood}_{suffix}"
            if column in joint_post.columns:
                joint_post[column] = joint_post[column].fillna(0.0)

    segment_columns = [
        "seg_cons",
        "seg_start",
        "seg_end",
        *[column for column in segs_consensus.columns if column in {
            "n_genes",
            "n_snps",
            "p_loh",
            "p_amp",
            "p_del",
            "p_bamp",
            "p_bdel",
            "LLR",
            "LLR_x",
            "LLR_y",
            }],
        ]

    segs_sel = segs_consensus.loc[:, segment_columns].copy()

    segs_sel = segs_sel.rename(
        columns={
            "seg_cons": "seg",
            "p_loh": "prior_loh",
            "p_amp": "prior_amp",
            "p_del": "prior_del",
            "p_bamp": "prior_bamp",
            "p_bdel": "prior_bdel",
            })

    joint_post = pd.merge(joint_post, segs_sel, on="seg", how="left")

    # Combine expression and allele log-likelihoods.
    for likelihood in likelihood_columns:
        joint_post[likelihood] = (joint_post[f"{likelihood}_x"]
                                  + joint_post[f"{likelihood}_y"]
                                  )

    # Local, non-spatial posterior.
    joint_post = compute_posterior(joint_post)

    # Modality diagnostic probabilities.
    for suffix in ("x", "y"):
        logbf_column = f"logBF_{suffix}"
        probability_column = f"p_cnv_{suffix}"

        if logbf_column in joint_post.columns:
            #joint_post[probability_column] = (1.0 / (1.0 + np.exp(-joint_post[logbf_column])))
            joint_post[probability_column] = expit(joint_post[logbf_column].to_numpy(dtype=np.float64))

    # MLE from the local joint likelihoods.
    mle_columns = [
        "l11",
        "l20",
        "l10",
        "l21",
        "l31",
        "l22",
        "l00",
    ]
    mle_states = np.asarray(["neu", "loh", "del", "amp", "amp", "bamp", "bdel"])

    joint_post["cnv_state_mle"] = mle_states[
        np.argmax(joint_post.loc[:, mle_columns].to_numpy(dtype=float), axis=1,)]

    if spatial and method == "hmrf":
        joint_post = hmrf.hmrf_regularize_joint_post(
            joint_post=joint_post,
            adata=count_mat,
            connectivity_key=connectivity_key,
            **method_kwargs,
        )
    else:
        map_columns = [
            "p_neu",
            "p_loh",
            "p_del",
            "p_amp",
            "p_bamp",
            "p_bdel",
        ]
        
        map_states = np.asarray(["neu", "loh", "del", "amp", "bamp", "bdel"])
        probabilities = joint_post[map_columns].to_numpy(dtype=float)
        valid = np.isfinite(probabilities).all(axis=1)
        
        joint_post["cnv_state_map"] = pd.Series(pd.NA,
                                                index=joint_post.index,
                                                dtype="string",
                                                )
        
        joint_post.loc[valid, "cnv_state_map"] = map_states[np.argmax(probabilities[valid], axis=1)]
        
        
    joint_post["seg_label"] = (
        joint_post["seg"].astype("string")
        + "("
        + joint_post["cnv_state"].astype("string")
        + ")"
    )

    return joint_post


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """
    Compute binary entropy without evaluating log2(0).

    For valid probabilities:
      - H(0) = H(1) = 0
      - values in (0, 1) use the standard binary-entropy formula

    NaN and out-of-range inputs return 0.
    """
    p = np.asarray(p, dtype=np.float64)
    entropy = np.zeros_like(p)

    interior = np.isfinite(p) & (p > 0.0) & (p < 1.0)
    values = p[interior]

    entropy[interior] = (
        -values * np.log2(values)
        - (1.0 - values) * np.log2(1.0 - values)
    )

    return entropy


def joint_post_entropy(joint_post: pd.DataFrame) -> pd.Series:
    """
    Compute the mean binary entropy of p_cnv within each segment.
    """
    entropy_by_row = pd.Series(
        np.nan,
        index=joint_post.index,
        dtype=np.float64,
    )

    for _, group in joint_post.groupby(
        "seg",
        observed=True,
        sort=False,
    ):
        values = group["p_cnv"].to_numpy(dtype=np.float64)
        values = values[~np.isnan(values)]

        # Preserve the old result for an all-missing group: mean entropy is NaN.
        mean_entropy = (
            np.nan
            if values.size == 0
            else float(binary_entropy(values).mean())
        )

        entropy_by_row.loc[group.index] = mean_entropy

    return entropy_by_row


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
        sc_post_multi['seg'] = sc_post_multi['seg'].astype("string") + '_' + sc_post_multi['cnv_state'].astype("string")
        
        # For each row, dynamically select the posterior values based on cnv_state.
        def select_posteriors(row):
            state = row["cnv_state"]
            p_col = f"p_{state}"
            z_col = f"Z_{state}"

            p_cnv = row.get(p_col, np.nan)

            row["p_cnv"] = p_cnv
            row["p_n"] = (
                1.0 - p_cnv
                if pd.notna(p_cnv)
                else np.nan
            )
            row["Z_cnv"] = row.get(z_col, np.nan)

            # logBF must describe the posterior values stored in this row.
            if pd.notna(row["p_cnv"]) and pd.notna(row["p_n"]):
                eps = 1e-12
                p_alt = np.clip(float(row["p_cnv"]), eps, 1.0)
                p_ref = np.clip(float(row["p_n"]), eps, 1.0)

                row["logBF"] = np.log(p_alt) - np.log(p_ref)
            else:
                row["logBF"] = np.nan

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
        sc_post['seg_label'] = sc_post['seg'].astype("string") + "(" + sc_post['cnv_state'].astype("string") + ")"

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


# Last part of iteration

def _nx_dfs_nodes(G: nx.DiGraph, root: int) -> List[int]:
    """Reachable-only DFS preorder."""
    if root not in G:
        return []
    return list(nx.dfs_preorder_nodes(G, source=root))


def _unique_preserve_order(xs: List[Any]) -> List[Any]:
    seen = set()
    out = []
    for x in xs:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def segs_equal(segs_1: pd.DataFrame, segs_2: pd.DataFrame) -> bool:
    """
   
    """
    cols = ["CHROM", "seg", "seg_start", "seg_end", "cnv_state_post"]

    a = segs_1.loc[:, [c for c in cols if c in segs_1.columns]]
    b = segs_2.loc[:, [c for c in cols if c in segs_2.columns]]

    return a.equals(b)


def clone_to_node_from_Gm(G_m: nx.DiGraph) -> Dict[int, int]:
    """
    
    """
    out: Dict[int, int] = {}
    for node_id, attrs in G_m.nodes(data=True):
        c = attrs.get("clone", None)
        if c is None or (isinstance(c, float) and np.isnan(c)):
            continue
        out[int(c)] = int(node_id)
    return out


def build_subtrees_from_Gm(
    Gm: nx.DiGraph,
    clone_post: pd.DataFrame,
    gt_vertex_col: str = "GT",
    clone_vertex_col: str = "clone",
    gt_opt_col: str = "GT_opt",
    cell_col: str = "cell",
    ) -> Dict[int, Dict[str, Any]]:
    """

    """
    # if gt_opt_col not in clone_post.columns:
    #     raise ValueError(f"clone_post must contain column '{gt_opt_col}'.")
    # if cell_col not in clone_post.columns:
    #     raise ValueError(f"clone_post must contain column '{cell_col}'.")

    # Keep node id exactly as in Gm
    v_rows = []
    node_labels = sorted(Gm.nodes())
    for vid in node_labels:
        attrs = Gm.nodes[vid]

        gt = attrs.get(gt_vertex_col, "")
        if gt is None or (isinstance(gt, float) and np.isnan(gt)):
            gt = ""
        gt = str(gt)

        cl = attrs.get(clone_vertex_col, np.nan)  # keep NA-like values
        v_rows.append({"id": int(vid), "GT": gt, "clone": cl})

    vdf = pd.DataFrame(v_rows)

    # Prepare clone_post join key as string (NA -> "")
    cp = clone_post.copy()
    cp[gt_opt_col] = cp[gt_opt_col].fillna("").astype(str)

    out: Dict[int, Dict[str, Any]] = {}

    for c in node_labels:
        reachable = list(nx.dfs_preorder_nodes(Gm, source=c))
        sub_v = vdf[vdf["id"].isin(reachable)]
        joined = sub_v.merge(cp, left_on="GT", right_on=gt_opt_col, how="inner")
        members = pd.unique(joined["GT"]).tolist()
        clones = pd.unique(joined["clone"]).tolist()
        clones = [str(clone) for clone in clones]
        cells = joined[cell_col].tolist()
        size = len(cells)

        out[str(c)] = {"sample": str(c),
                  "members": members,
                  "clones": clones,
                  "cells": cells,
                  "size": size,
                 }

    return out


def build_clones_from_clone_post(
    clone_post: pd.DataFrame,
    clone_opt_col: str = "clone_opt",
    gt_opt_col: str = "GT_opt",
    cell_col: str = "cell",
    ) -> Dict[Any, Dict[str, Any]]:
    """

    Returns
    -------
    Dict[key, dict]
        key is clone_opt (same as 'sample'), value is the per-clone dict.
    """
    # for col in (clone_opt_col, gt_opt_col, cell_col):
    #     if col not in clone_post.columns:
    #         raise ValueError(f"clone_post must contain column '{col}'.")

    cp = clone_post.copy()

    # map NA -> "".
    cp[gt_opt_col] = cp[gt_opt_col].fillna("")

    out: Dict[Any, Dict[str, Any]] = {}

    for clone_key, df in cp.groupby(clone_opt_col, sort=True, dropna=True):
        members = pd.unique(df[gt_opt_col]).tolist()
        cells = df[cell_col].tolist()
        size = len(cells)

        out[str(clone_key)] = {"sample": str(clone_key),
                          "members": members,
                          "cells": cells,
                          "size": size,
                         }

    return out


def check_convergence_and_update(
    segs_consensus_old: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    check_convergence: bool,
    ) -> Tuple[bool, pd.DataFrame]:
    """
    
    """
    if not check_convergence:
        return False, segs_consensus_old

    converged = bool(segs_equal(segs_consensus_old, segs_consensus))
    if converged:
        return True, segs_consensus_old
    return False, segs_consensus.copy()

