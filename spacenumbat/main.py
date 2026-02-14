#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  6 19:55:48 2025

@author: lillux
"""
from typing import Mapping
import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import spacenumbat
from spacenumbat import (utils, diagnostics, clustering, 
                         operations, plot, spatial_utils,
                         tree, phylo)

from spacenumbat._log import configure, get_logger

def run_numbat(
    count_mat,
    lambdas_ref,
    df_allele,
    gtf=None,
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
    max_cost=0,
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
    check_convergence=False,
    exclude_neu=True,
    p_min = 1e-10,
    plot_results=True,
    filter_hla_hg38=True, #Just added
    filter_chromosome_segments=None, # .tsv or pd.Dataframe with coordinate to skip. Needs: [CHROM, start, end]
    spatial=False,
    spatial_method="cpr",
    spatial_decay="gaussian",
    spatial_method_kwargs: Mapping = None,
    connectivity_key: str ="spatial_connectivities",
    distance_key: str = "weighted_adjacency",
    ):
    """
    Run workflow to decompose tumor subclones.

    This function implements the workflow for deconvoluting tumor subclones using the given
    raw count matrices, allele counts, and additional parameters for model fitting and phylogenetic
    inference.

    Parameters
    ----------
    count_mat : anndata.AnnData
        Raw count matrix where rows are genes and columns are cells.
    lambdas_ref : numpy.ndarray, pandas.DataFrame, dict, or similar
        Either a mapping with gene names as keys and normalized expression as values, or a matrix
        where rows are genes and columns are pseudobulk names.
    df_allele : pandas.DataFrame
        DataFrame of allele counts per cell, produced by preprocess_allele.
    gtf : str or Path    
        dataframe Transcript GTF, if NULL will use the default GTF for the specified genome 
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
    p_min : float, optional
        The minimum threshold for p_cnv. p_cnv values will be clamped to the interval [p_min, 1 - p_min]
    plot_results : bool, optional
        Whether to produce plot of the data during the analysis workoflow. Default is True.
    filter_hla_hg38 : bool
        Filter HLA region on hg38 genomic coordinates, default is True.
    filter_chromosome_segments=None. 
        .tsv or pd.Dataframe with coordinates to skip. Needs: [CHROM, start, end].
    spatial : bool
        Flag to activate spatial mode to take in account spatial context in the analysis.
    spatial_method : str
        You can choose one between:
            - "degree": unweighted neighbor mean using the connectivity matrix (adds self-loops).
            - "weighted": inverse-distance weighted mean using the distance matrix (adds self-loops).
            - "diffuse": iterative random-walk diffusion (uses _random_walk_diffuse).
            - "cpr": personalized PageRank–style diffusion (uses _pagerank_diffuse).
    spatial_decay : str
        Decay kernel mapping distance d to weight w:
            - "gaussian": exp(-(d^2) / sigma^2)
            - "exp":      exp(-d / ell)
            - "invdist":  1 / (d + 1e-6)^p
            - "cauchy":   1 / (1 + (d / sigma)^2)
    spatial_method_kwargs : dict, optional
        key:value pairs for get_joint_post function, as argument of the
        spatial smoothing functions. Depending on the chosen spatial method,
        here are the accepted key:value pairs:
            - diffuse: {"alpha": float, 
                        "steps": int}
            - cpr: {"alpha": float, 
                    "coifman_alpha":float, 
                    "lazy":float,
                    "steps": int} 
    distance_key: str
        default is "weighted_adjacency",
    connectivity_key: str
        default is "spatial_connectivities"
    
    
    verbose : bool, optional
        Flag to enable verbose output. Default is True.

    Returns
    -------
    int
        A status code indicating success or failure of the workflow.

    """
    configure(level="INFO", log_dir=out_dir)
    logging.getLogger('numba').setLevel(logging.WARNING)
    log = get_logger(__name__)
    log.info("Starting pipeline!")
    
    if not gtf:
        if genome == "hg38":
            gtf = spacenumbat.data.hg38
        elif genome == "hg19":
            gtf = spacenumbat.data.hg19
        elif genome == "mm10":
            gtf = spacenumbat.data.mm10
            filter_hla_hg38=False
            filter_chromosome_segments=None
        else:
            msg = f"genome version must be hg38, hg19, or mm10, not {genome}"
            raise ValueError(msg)
    else:
        filter_hla_hg38=False
        gtf = diagnostics.load_and_validate_annotation(gtf)
    
    gtf["CHROM"] = gtf["CHROM"].astype("string")
    
    count_mat = utils.check_anndata(count_mat, count_to_int=False)
    df_allele = utils.annotate_genes(df=df_allele, gtf=gtf)
    df_allele = utils.check_allele_df(df_allele)
    lambdas_ref = utils.check_exp_ref(lambdas_ref)
    
    # filter for annotated genes
    gene_shared = set(gtf['gene']).intersection(set(count_mat.var_names.values)).intersection(set(lambdas_ref.index.values))
    ordered_gene_shared = [i for i in gtf['gene'] if i in gene_shared]
    count_mat = count_mat[:,ordered_gene_shared]
    lambdas_ref = lambdas_ref.loc[ordered_gene_shared,:]
    
    # filter 0 coverage cells
    zero_cov = count_mat[count_mat.X.sum(1) == 0].obs_names.to_list()
    if len(zero_cov) > 0:
        log.info(f"Filtering out {len(zero_cov)} cells with 0 coverage")
        count_mat = count_mat[~count_mat.obs_names.isin(zero_cov),:]
        df_allele = df_allele[~df_allele.cell.isin(zero_cov)]
    
    # keep cells that have a transcriptome
    df_allele = df_allele[df_allele.cell.isin(count_mat.obs_names)]
    if df_allele.shape[0] == 0:
        msg = "No matching cell names between count_mat and df_allele. Breaking pipeline!"
        raise ValueError(msg)

    # check if conficts on given genomic information
    if (not (segs_loh is None) and (segs_loh.shape[0]>0)) and segs_consensus_fix:
        msg = "Cannot specify both segs_loh and segs_consensus_fix. Breaking pipeline!"
        raise ValueError(msg)
    
    # check provided consensus CNVs
    segs_consensus_fix = diagnostics.check_segs_fix(segs_consensus_fix)
    
    # check provided clonal LoH
    if (not (segs_loh is None) and (segs_loh.shape[0]>0)):
        if call_clonal_loh:
            msg = "Cannot specify both segs_loh and call_clonal_loh"
            raise ValueError(msg)
        segs_loh = diagnostics.check_segs_loh(segs_loh)
    
    # Check if filtering Chromosomal segments
    if (not (filter_chromosome_segments is None) and (filter_chromosome_segments.shape[0]>0)):
        filter_segments_df = diagnostics.check_filter_segments(filter_chromosome_segments)
    else:
        filter_segments_df = None
        
    # Prepare parameter log
    log_lines = [
        "",
        f"Spacenumbat version: {spacenumbat.__version__}",
        "Running under parameters:",
        f"genome = {genome}",
        f"out_dir = {out_dir}",
        f"max_iter = {max_iter}",
        f"max_nni = {max_nni}",
        f"t = {t}",
        f"gamma = {gamma}",
        f"min_LLR = {min_LLR}",
        f"alpha = {alpha}",
        f"eps = {eps}",
        f"max_entropy = {max_entropy}",
        f"init_k = {init_k}",
        f"min_cells = {min_cells}",
        f"tau = {tau}",
        f"nu = {nu}",
        f"max_cost = {max_cost}",
        f"n_cut = {n_cut}",
        f"min_depth = {min_depth}",
        f"min_genes = {min_genes}",
        f"min_overlap = {min_overlap}",
        f"use_loh = {'auto' if use_loh is None else use_loh}",
        f"segs_loh = {'None' if segs_loh is None else 'Given'}",
        f"call_clonal_loh = {call_clonal_loh}",
        f"segs_consensus_fix = {'None' if segs_consensus_fix is None else 'Given'}",
        f"exclude_neu = {exclude_neu}",
        f"common_diploid = {common_diploid}",
        f"diploid_chroms = {'None' if diploid_chroms is None else 'Given'}",
        f"skip_nj = {skip_nj}",
        f"random_init = {random_init}",
        f"multi_allelic = {multi_allelic}",
        f"p_multi = {('auto(1-alpha)' if p_multi is None else p_multi)}",
        f"p_min = {p_min}",
        f"check_convergence = {check_convergence}",
        f"plot_results = {plot_results}",
        f"ncores = {ncores}",
        f"ncores_nni = {ncores_nni}",
        f"Filter HLA region = {filter_hla_hg38}",
        f"Filtering custom chromosomal region = {'None' if filter_segments_df is None else 'Given'}",
        f"spatial = {spatial}",
        f"spatial_method = {spatial_method}",
        f"spatial_decay = {spatial_decay}",
        f"spatial_method_kwargs = {'None' if spatial_method_kwargs is None else 'Given'}",
        f"connectivity_key = {connectivity_key}",
        f"distance_key = {distance_key}",
        "Input metrics:",
        f"{count_mat.shape[0]} cells",
        ]

    log.info('\n'.join(log_lines))
    
    # Call clonal loss of heterozygosity inference if requested
    if call_clonal_loh:
        msg = "Calling segments with clonal LoH."
        log.info(msg)
        
        bulk = utils.get_bulk(count_mat, 
                              lambdas_ref,
                              df_allele, 
                              gtf, 
                              filter_hla=filter_hla_hg38,
                              filter_segments=filter_segments_df,
                              min_depth=min_depth,
                              nu=nu)
        segs_loh = utils.detect_clonal_loh(bulk, t=t, min_depth=min_depth)
        
        #TODO remove this
        log.info(f"segs_loh shape is: {segs_loh.shape}\nbulk shape is: {bulk.shape}")       
        
        if (segs_loh is not None) and (segs_loh.shape[0] > 0):
            segs_loh.to_csv(os.path.join(out_dir, "segs_loh.tsv"), sep="\t")
        else:
            log.info('No segments with clonal LoH detected.')
            
    # Calculate reference transcriptomic profile of cellwi th reference categories
    sc_refs = clustering.choose_ref_cor(count_mat, lambdas_ref, gtf)
    sc_refs.to_csv(os.path.join(out_dir, "sc_refs.tsv"), sep="\t")
    
    
    if random_init:
        log.info("")
        ## TODO
        
    elif init_k == 1:
        log.info("Initializing with all-cell pseudobulk ...")
        
        
    else:
        log.info("Approximating initial clusters using smoothed expression ...")
        clust = clustering.exp_hclust(count_mat=count_mat,
                           lambdas_ref=lambdas_ref,
                           gtf=gtf,
                           sc_refs=sc_refs,
                           ncores=ncores,
                           filter_hla=filter_hla_hg38,
                           filter_segments=filter_chromosome_segments,
                           verbose=verbose)
        # save window-smoothed normalized expression profiles as AnnData
        log.info("Saving clustering results")
        clust["gexp_roll_wide"].write_h5ad(os.path.join(out_dir, "gexp_roll_wide.h5ad"))
        pd.DataFrame(clust["hc"]).to_csv(os.path.join(out_dir, "hc_initial_hierarchical_clustering.tsv"), sep="\t")
        log.info(f"Normalized expression results saved at {os.path.join(out_dir, 'gexp_roll_wide.h5ad')}")
        log.info(f"Initial hierarchical clustering results saved at {os.path.join(out_dir, 'hc_initial_hierarchical_clustering.tsv')}")
        
        # extract cell groupings
        nodes_dict = clustering.get_nodes_celltree(clust, init_k)
        
        ## TODO: optional plot
        
    clones = {k:nodes_dict[str(k)] for k in range(init_k+1) if len(nodes_dict[str(k)]['members'])==1}
    
    normal_cells = []
    segs_consensus_old = pd.DataFrame()
    

    ######## Begin iterations #TODO
    
    # for i in max_iter:
    i = 0 # temporary placeholder for iteration
    log.info(f"Starting iteration {i}")
    
    subtrees = {k:v for k,v in nodes_dict.items() if v['size'] > min_cells}
    
    bulk_subtrees = utils.make_group_bulks(groups=subtrees,
                                           count_mat=count_mat,
                                           df_allele=df_allele,
                                           lambdas_ref=lambdas_ref,
                                           gtf=gtf,
                                           min_depth=min_depth,
                                           nu=nu,
                                           segs_loh=segs_loh,
                                           filter_hla=filter_hla_hg38,
                                           filter_segments=filter_chromosome_segments,
                                           ncores=ncores)
    
    ### Diagnostics ##TODO
    
    # if i == 0:
        
    bulk_subtrees0 = bulk_subtrees[bulk_subtrees["sample"] == "0"].copy()
    
    diagnostics.check_contam(bulk_subtrees0)
    diagnostics.check_exp_noise(bulk_subtrees0)
    
    if segs_consensus_fix is None:
        
        bulk_test = operations.run_group_hmms(bulk_subtrees,
                                              t = t,
                                              gamma = gamma,
                                              alpha = alpha,
                                              nu = nu,
                                              min_genes = min_genes,
                                              common_diploid = common_diploid,
                                              diploid_chroms = diploid_chroms,
                                              ncores = ncores,
                                              verbose = verbose)
        
        bulk_test.to_csv(os.path.join(out_dir, f"bulk_subtrees_{i}.tsv"), sep="\t")
        
        if plot_results:
            with plt.ioff():  # disables live rendering inside the block
    
                plot_subtrees = plot.plot_bulks(bulk_test, 
                                                  exp_limit=4, 
                                                  text_size=10, 
                                                  title_size=14,
                                                  panel_vspace=1)
                plot_subtrees.savefig(os.path.join(out_dir, f"bulk_subtrees{i}.jpg"), dpi=200)
                plt.close("all")
           
        # define consensus CNVs
        segs_consensus = operations.get_segs_consensus(bulk_test,
                                       min_LLR = min_LLR,
                                       min_overlap = min_overlap,
                                       retest = True)
                
        # check termination
        if np.all(segs_consensus.cnv_state_post == 'neu'):
            msg = 'No CNV remains after filtering by LLR in pseudobulks. Consider reducing min_LLR.'
            log.info(msg)
            return msg
        
        bulk_retest = operations.retest_bulks(bulk_test,
                                              segs_consensus,
                                              diploid_chroms=diploid_chroms,
                                              gamma=gamma,
                                              min_LLR=min_LLR,
                                              ncores=ncores)
        bulk_retest.to_csv(os.path.join(out_dir, f"bulk_subtrees_retest_{i}.tsv"), sep="\t")
        
        ## define consensus CNVs again
        segs_consensus_retest = operations.get_segs_consensus(bulk_retest, 
                                                   min_LLR=min_LLR, 
                                                   min_overlap=min_overlap, 
                                                   retest=False) 
        
        ## check termination again
        if np.all(segs_consensus_retest.cnv_state_post == 'neu'):
            msg = 'No CNV remains after filtering by LLR in pseudobulks. Consider reducing min_LLR.'
            log.info(msg)
            return msg
    
    else: # if seg_consensus_fix #TODO
    
        log.info('Using fixed consensus CNVs')
        segs_consensus = segs_consensus_fix
         
        bulk_subtrees = utils.classify_alleles(
            utils.annot_theta_mle(
            utils.annot_consensus(
                bulk_subtrees, 
                segs_consensus)
            ))


    # retest on clones
    clones_filt = {k:v for k, v in clones.items() if v['size'] > min_cells}
    
    if len(clones_filt) == 0:      
        msg = ('No clones remain after filtering by size. Consider reducing min_cells.\n'
               'Interrupting workflow...')
        log.info(msg)
        return(msg)
    
    bulk_clones = utils.make_group_bulks(groups = clones_filt,
                               count_mat = count_mat,
                               df_allele = df_allele,
                               lambdas_ref = lambdas_ref,
                               gtf = gtf,
                               min_depth = min_depth,
                               nu = nu,
                               segs_loh = segs_loh,
                               ncores = ncores)
    
    bulk_clones_group = operations.run_group_hmms(bulks = bulk_clones,
                                   t = t,
                                   gamma = gamma,
                                   alpha = alpha,
                                   nu = nu,
                                   min_genes = min_genes,
                                   common_diploid = common_diploid,
                                   diploid_chroms = diploid_chroms,
                                   ncores = ncores,
                                   verbose = verbose,
                                   retest = False)
    
    bulk_clones_retest = operations.retest_bulks(bulks = bulk_clones_group,
                                  segs_consensus = segs_consensus_retest,
                                  gamma = gamma,
                                  use_loh = use_loh,
                                  min_LLR = min_LLR,
                                  diploid_chroms = diploid_chroms,
                                  ncores = ncores)
    
    bulk_clones_retest.to_csv(os.path.join(out_dir, f"bulk_clones_{i}.tsv"), sep="\t")

    if plot_results:
        with plt.ioff():  # disables live rendering inside the block

            plot_subtrees = plot.plot_bulks(bulk_clones_retest, 
                                              exp_limit=4, 
                                              text_size=10, 
                                              title_size=14,
                                              panel_vspace=1)
            plot_subtrees.savefig(os.path.join(out_dir, f"bulk_clones_{i}.jpg"), dpi=200)
            plt.close("all")
            
    ### test for multi-allelic CNVs
    if multi_allelic:
        segs_consensus_retest = operations.test_multi_allelic(bulk_clones_retest, 
                                                              segs_consensus_retest, 
                                                              min_LLR = min_LLR, 
                                                              p_min = p_multi)
    
    segs_consensus_retest.to_csv(os.path.join(out_dir, f"segs_consensus_retest_{i}.tsv"), sep="\t")

    ### Evaluate CNV per cell
    log.info("Evaluating CNV per cell")
    
    segs_consensus_retest_corrected = segs_consensus_retest.copy()
    segs_consensus_retest_corrected.loc[:,'cnv_state'] = [row.cnv_state if row.cnv_state == 'neu' else row.cnv_state_post for idx, row in segs_consensus_retest_corrected.iterrows()]
    
    exp_post = operations.get_exp_post(segs_consensus_retest_corrected,
                        count_mat=count_mat,
                        gtf=gtf,
                        lambdas_ref=lambdas_ref,
                        use_loh = use_loh,
                        segs_loh = segs_loh,
                        sc_refs=sc_refs,
                        ncores=ncores,
                        verbose=True)
    
    haplotype = operations.get_haplotype_post(bulk_retest, 
                                              segs_consensus_retest_corrected)
    
    allele_post = operations.get_allele_post(df_allele=df_allele,
                                             haplotypes=haplotype,
                                             segs_consensus=segs_consensus_retest_corrected)
    
    count_mat = spatial_utils.get_spatial_info(counts_mat=count_mat,
                                               ncores=ncores,
                                               distance_key=distance_key,
                                               kind=spatial_decay,
                                               connectivity_key=connectivity_key)
    

    joint_post = operations.get_joint_post(
        exp_post=exp_post,
        allele_post=allele_post,
        segs_consensus=segs_consensus_retest,
        count_mat=count_mat,
        distance_key=distance_key,
        spatial=spatial,
        method=spatial_method,
        method_kwargs=spatial_method_kwargs
        )
        
    joint_post.loc[:,'avg_entropy'] = operations.joint_post_entropy(joint_post)
    
    if multi_allelic:
        exp_post = operations.expand_states(exp_post, segs_consensus_retest)
        allele_post = operations.expand_states(allele_post, segs_consensus_retest)
        joint_post = operations.expand_states(joint_post, segs_consensus_retest)

        
    exp_post.to_csv(os.path.join(out_dir, f"exp_post_{i}.tsv"), sep="\t")
    allele_post.to_csv(os.path.join(out_dir, f"allele_post_{i}.tsv"), sep="\t")
    joint_post.to_csv(os.path.join(out_dir, f"joint_post_{i}.tsv"), sep="\t")

    
    ### Build phylogeny  
    msg = "Phylogeny reconstruction started."
    log.info(msg)
    
    joint_post_filtered = joint_post[(joint_post.cnv_state != 'neu') & 
                                (joint_post.avg_entropy < max_entropy) & 
                                (joint_post.LLR > min_LLR)].copy()
        
    if joint_post_filtered.shape[0] == 0:
        log.info(f"No CNV remains after filtering by entropy in single cells.\n"
                 f"Consider increasing max_entropy. Current entropy is: {max_entropy}")
    else:
        n_cnv = joint_post_filtered.seg.unique().shape[0]
        log.info(f'Using {n_cnv} CNAs to construct phylogeny')
    
    # construct genotype probability matrix
    P = operations.get_joint_post_matrix(joint_post_filtered, p_min=p_min)
    P_saving_path = os.path.join(out_dir, f"geno_{i}.tsv")
    P.to_csv(P_saving_path, sep="\t")
    log.info(f"P matrix has been saved at {P_saving_path}")
    
    treeML = tree.P_to_candidate_tree(P_df=P,
                                         n_jobs=ncores)
    
    gtree = tree.get_gtree(treeML,
                           P,
                           n_cut=n_cut,
                           max_cost=max_cost)
    
    G_m = tree.label_genotype(tree.get_mut_graph(gtree))
    
    log.info(f"Tree building completed, pass {i}")
    
    clone_post = phylo.get_clone_post(gtree, exp_post, allele_post)
    clone_post.to_csv(os.path.join(out_dir, f"clone_post_{i}.tsv"), sep="\t")

    normal_cells = clone_post[clone_post.p_cnv <= 0.5].cell
    msg = f"Found {len(normal_cells)} normal cells."
    log.info(msg)
    
   #if plot_results:
        #TODO: make plot
        
    clone_to_node = operations.clone_to_node_from_Gm(G_m)
    subtrees = operations.build_subtrees_from_Gm(G_m, clone_post)
    clones = operations.build_clones_from_clone_post(clone_post)
    
    if check_convergence:
        # convergence
        converged, segs_consensus_old = operations.check_convergence_and_update(segs_consensus_old=segs_consensus_old,
                                                                                segs_consensus=segs_consensus_retest,
                                                                                check_convergence=check_convergence)
        if converged:
            log.info("converged")
            # break

    
    
    return exp_post, allele_post, segs_consensus_retest, count_mat, treeML, clone_post, G_m, subtrees, clones
    #return clone_post, G_m

    
    
    
    
    
    
    
    
    
    
    