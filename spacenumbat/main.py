#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 19:55:48 2025

@author: lillux
"""

import os
import tempfile
import logging

import numpy as np
import pandas as pd
from scipy import sparse

from spacenumbat import utils

from spacenumbat._log import configure, get_logger

def run_numbat(
    count_mat,
    lambdas_ref,
    df_allele,
    genome='hg38',
    out_dir=None,
    max_iter=2,
    max_nni=100,
    t=1e-5,
    gamma=20,
    min_LLR=5,
    alpha=1e-4,
    eps=1e-5,
    max_entropy=0.5,
    init_k=3,
    min_cells=50,
    tau=0.3,
    nu=1,
    max_cost=None,
    n_cut=0,
    min_depth=0,
    common_diploid=True,
    min_overlap=0.45,
    ncores=1,
    ncores_nni=None,
    random_init=False,
    segs_loh=None,
    call_clonal_loh=False,
    verbose=True,
    diploid_chroms=None,
    segs_consensus_fix=None,
    use_loh=None,
    min_genes=10,
    skip_nj=False,
    multi_allelic=True,
    p_multi=None,
    plot=True,
    check_convergence=False,
    exclude_neu=True
):
    """
    Run workflow to decompose tumor subclones.

    This function implements the workflow for deconvoluting tumor subclones using the given
    raw count matrices, allele counts, and additional parameters for model fitting and phylogenetic
    inference.

    Parameters
    ----------
    count_mat : scipy.sparse.csc_matrix or similar
        Raw count matrix where rows are genes and columns are cells.
    lambdas_ref : numpy.ndarray, pandas.DataFrame, dict, or similar
        Either a mapping with gene names as keys and normalized expression as values, or a matrix
        where rows are genes and columns are pseudobulk names.
    df_allele : pandas.DataFrame
        DataFrame of allele counts per cell, produced by preprocess_allele.
    genome : str, optional
        Genome version (e.g., 'hg38', 'hg19', or 'mm10'). Default is 'hg38'.
    out_dir : str, optional
        Output directory. Default is the system temporary directory.
    gamma : float, optional
        Dispersion parameter for the Beta-Binomial allele model. Default is 20.
    t : float, optional
        Transition probability. Default is 1e-5.
    init_k : int, optional
        Number of clusters in the initial clustering. Default is 3.
    min_cells : int, optional
        Minimum number of cells to run the hidden Markov model (HMM) on. Default is 50.
    min_genes : int, optional
        Minimum number of genes to call a segment. Default is 10.
    max_cost : float, optional
        Likelihood threshold to collapse internal branches. Default is set to the number of cells
        in count_mat multiplied by tau.
    n_cut : int, optional
        Number of cuts on the phylogeny to define subclones. Default is 0.
    tau : float, optional
        Factor (range 0–1) to determine max_cost as a function of the number of cells. Default is 0.3.
    nu : float, optional
        Phase switch rate. Default is 1.
    alpha : float, optional
        P-value cutoff for diploid finding. Default is 1e-4.
    min_overlap : float, optional
        Minimum copy number variation (CNV) overlap threshold. Default is 0.45.
    ncores : int, optional
        Number of threads to use. Default is 1.
    ncores_nni : int, optional
        Number of threads to use for nearest neighbor interchange (NNI). Default is the same as ncores.
    max_iter : int, optional
        Maximum number of iterations to run the phylogeny optimization. Default is 2.
    max_nni : int, optional
        Maximum number of iterations to run NNI in the maximum likelihood phylogeny inference. Default is 100.
    eps : float, optional
        Convergence threshold for maximum likelihood tree search. Default is 1e-5.
    multi_allelic : bool, optional
        Whether to call multi-allelic CNVs. Default is True.
    p_multi : float, optional
        P-value cutoff for calling multi-allelic CNVs. Default is 1 - alpha.
    use_loh : bool, optional
        Whether to include loss of heterozygosity (LOH) regions in the expression baseline.
    segs_loh : pandas.DataFrame or None, optional
        Segments of clonal LOH to be excluded. Default is None.
    call_clonal_loh : bool, optional
        Whether to call segments with clonal LOH. Default is False.
    diploid_chroms : list or None, optional
        Known diploid chromosomes. Default is None.
    segs_consensus_fix : pandas.DataFrame or None, optional
        Pre-determined segmentation of consensus CNVs. Default is None.
    check_convergence : bool, optional
        Whether to terminate iterations based on consensus CNV convergence. Default is False.
    random_init : bool, optional
        Whether to initiate phylogeny using a random tree (internal use only). Default is False.
    exclude_neu : bool, optional
        Whether to exclude neutral segments from CNV retesting (internal use only). Default is True.
    plot : bool, optional
        Whether to plot results. Default is True.
    verbose : bool, optional
        Flag to enable verbose output. Default is True.

    Returns
    -------
    int
        A status code indicating success or failure of the workflow.

    """


    configure(level="DEBUG", log_dir=out_dir)
    log = get_logger(__name__)
    log.info("This is an info message.")
    
    
    
    
    
    #count_mat = utils.check_anndata(count_mat)
    #df_allele = utils.annotate_genes(df=count_mat, gtf=gtf)
    

    
    