#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 24 00:33:04 2025

@author: lillux
"""

import pandas as pd
import numpy as np
from joblib import cpu_count, Parallel, delayed

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
        bulks (pd.DataFrame): Pseudobulk profiles.
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