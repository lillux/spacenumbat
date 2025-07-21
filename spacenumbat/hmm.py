#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:06:08 2024

@author: lillux
"""
import numpy as np
from spacenumbat.dist_prob import dnbinom, dpoilog, log_beta_binomial_pmf
import logging



def viterbi_loh(HMM):
    n = len(HMM['x'])
    m = HMM['Pi'][:,:,1].shape[0]
    nu = np.zeros([n,m])
    #mu = np.zeros([n,m+1])
    z = np.zeros(n)
    
    nu[0,:] = np.log(HMM['delta'])
    logPi = np.log(HMM['Pi'])

    for i in range(n):
        if i > 0:
            matrixnu = np.full([m,m], nu[i-1,])
            nu[i,] = np.max(matrixnu + logPi[:,:,i], axis=1)
        nu[i,] = nu[i,] + dnbinom(x=HMM['x'][i], mu=HMM['pm']*HMM['pn'][i], size=HMM['snp_sig'])
        nu[i,] = nu[i,] + dpoilog(x = np.repeat(HMM['y'][i], m),
                                    sig = np.repeat(HMM['sig'], m),
                                    mu = HMM['mu'] + np.log(HMM['phi'] * HMM['d'] * HMM['lambda_star'][i]),
                                 log=True)

    z[-1] = np.argmax(nu[-1,])
    for i in range(n-2,-1,-1):
        z[i] = np.argmax(logPi[:,:,i+1][:,np.int32(z[i+1])] + nu[i,])
    z = z.astype(np.int32)
    #LL = np.max(nu[n-1,])
    HMM['states'] = np.array(HMM['states'])[z]

    return HMM['states']


def viterbi_compute(log_delta, logprob, logPi):
    """
    Viterbi algorithm for HMM with allele counts.

    Parameters:
        log_delta (numpy.ndarray): Log initial probabilities of shape (M,)
        logprob (numpy.ndarray): Log probabilities of observations, shape (N, M)
        logPi (numpy.ndarray): Log transition probabilities, shape (N, M, M)

    Returns:
        numpy.ndarray: Decoded states (indices), shape (N,)
    """
    N, M = logprob.shape
    nu = np.zeros((N, M))
    z = np.zeros(N, dtype=int)

    # Initialization
    nu[0, :] = log_delta + logprob[0, :]

    # Forward pass
    for i in range(1, N):
        # Compute nu[i, :] for each state
        nu_prev = nu[i - 1, :].reshape(M, 1)  # Shape (M, 1)
        logPi_i = logPi[i, :, :]              # Shape (M, M)
        sum_matrix = nu_prev + logPi_i        # Broadcasting to shape (M, M)
        nu[i, :] = np.max(sum_matrix, axis=0) + logprob[i, :]

    # Backtracking
    z[N - 1] = np.argmax(nu[N - 1, :])
    for i in range(N - 2, -1, -1):
        logPi_i1 = logPi[i + 1, :, :]         # Transition matrix for next step
        z[i] = np.argmax(logPi_i1[:, z[i + 1]] + nu[i, :])

    return z


def viterbi_allele(hmm):
    """
    Viterbi algorithm for allele HMM.

    Parameters:
        hmm (dict): HMM parameters including 'x', 'd', 'logPi', 'delta', 'alpha', 'beta', 'states', 'N', 'M'

    Returns:
        list: Decoded state names
    """
    N = hmm['N']
    M = hmm['M']
    x = hmm['x']
    d = hmm['d']
    logPi = hmm['logPi']
    delta = hmm['delta']
    alpha = hmm['alpha']
    beta = hmm['beta']
    states = hmm['states']

    # Compute log probabilities for observations using beta-binomial distribution
    logprob = np.zeros((N, M))
    for m in range(M):
        logprob[:, m] = log_beta_binomial_pmf(x, d, alpha[:, m], beta[:, m])
        #logprob[:, m] = scipy.stats.betabinom(x, d, alpha[:, m], beta[:, m])

    # Handle NaN values
    logprob = np.nan_to_num(logprob, nan=0.0)
    # Run Viterbi algorithm
    z = viterbi_compute(np.log(delta), logprob, logPi)
    # Map indices to state names
    decoded_states = [states[idx] for idx in z]

    return decoded_states


def run_allele_hmm_s5(pAD, DP, p_s, t=1e-5, theta_min=0.08, theta_max=0.4, gamma=20, prior=None):
    """
    Run a 5-state allele-only HMM with two theta levels.

    Parameters:
        pAD (numpy.ndarray): Paternal allele counts
        DP (numpy.ndarray): Total allele counts
        p_s (numpy.ndarray): Phase switch probabilities
        t (float): Transition probability between copy number states
        theta_min (float): Minimum haplotype frequency deviation threshold
        gamma (float): Overdispersion in the allele-specific expression
            Multiplying by gamma scales the distribution. A higher gamma means
            more "confidence" (less variance) in the expected frequency,
            making the Beta distribution more peaked.
        prior (numpy.ndarray): Prior probabilities for each state (length M)

    Returns:
        list: Decoded state names
    """
    import numpy as np

    gamma = np.unique(np.array(gamma))
    if len(gamma) > 1:
        raise ValueError('More than one gamma parameter')

    # States
    states = ["neu", "theta_1_up", "theta_1_down", "theta_2_up", "theta_2_down"]
    N = len(pAD)
    M = 5

    # Transition matrix
    def calc_trans_mat_s5(p_s_i, t):
        trans_mat = np.array([
            [1 - t, t / 4, t / 4, t / 4, t / 4],
            [t / 2, (1 - t) * (1 - p_s_i), (1 - t) * p_s_i, t / 4, t / 4],
            [t / 2, (1 - t) * p_s_i, (1 - t) * (1 - p_s_i), t / 4, t / 4],
            [t / 2, t / 4, t / 4, (1 - t) * (1 - p_s_i), (1 - t) * p_s_i],
            [t / 2, t / 4, t / 4, (1 - t) * p_s_i, (1 - t) * (1 - p_s_i)]
        ])
        return trans_mat

    # Build log transition matrices for each position
    logPi = np.zeros((N, M, M))
    for i in range(N):
        trans_mat = calc_trans_mat_s5(p_s[i], t)
        logPi[i] = np.log(trans_mat)

    # Initial probabilities
    if prior is None:
        prior = np.full(M, 1 / M)

    # Beta-binomial parameters
    theta_1 = theta_min
    theta_2 = theta_max
    alphas = gamma * np.array([0.5, 0.5 + theta_1, 0.5 - theta_1, 0.5 + theta_2, 0.5 - theta_2])
    betas = gamma * np.array([0.5, 0.5 - theta_1, 0.5 + theta_1, 0.5 - theta_2, 0.5 + theta_2])

    alpha_mat = np.tile(alphas, (N, 1))
    beta_mat = np.tile(betas, (N, 1))

    # Build HMM parameters
    hmm = {
        'x': pAD,
        'd': DP,
        'logPi': logPi,
        'delta': prior,
        'alpha': alpha_mat,
        'beta': beta_mat,
        'N': N,
        'M': M,
        'states': states
    }

    # Run Viterbi algorithm
    mpc = viterbi_allele(hmm)

    return mpc


def smooth_segs(bulk, min_genes=10):
    """
    Smooth the segments after HMM decoding.

    Parameters:
        bulk (pd.DataFrame): Pseudobulk profile.
        min_genes (int): Minimum number of genes to call a segment.

    Returns:
        pd.DataFrame: Pseudobulk profile with smoothed segments.

    Raises:
        ValueError: If any chromosome has no segments with more than min_genes genes.
    """
    # Copy the DataFrame to avoid modifying the original
    bulk = bulk.copy()
    # Within each segment, set 'cnv_state' to NaN if 'n_genes' <= min_genes
    # Get the number of genes per segment
    n_genes_per_seg = bulk.groupby('seg', observed=True)['n_genes'].first().reset_index()
    # Identify segments with insufficient genes
    small_segs = n_genes_per_seg.loc[n_genes_per_seg['n_genes'] <= min_genes, 'seg']
    # Set 'cnv_state' to NaN for these segments
    bulk.loc[bulk['seg'].isin(small_segs), 'cnv_state'] = np.nan
    # Fill NaN values in 'cnv_state' forward and backward within each chromosome
    bulk['cnv_state'] = bulk.groupby('CHROM', observed=True)['cnv_state'].ffill().bfill()
    # Check if any chromosome has all NaN in 'cnv_state'
    chrom_na = bulk.groupby('CHROM', observed=True)['cnv_state'].apply(lambda x: x.isna().all()).reset_index(name='all_na')

    # THIS RAISE ERROR IF FEW GENES ARE FOUND IN A CHROMOSOME
    if chrom_na['all_na'].any():
        chroms_na = ','.join(chrom_na.loc[chrom_na['all_na'], 'CHROM'].astype(str))
        msg = f"No segments containing more than {min_genes} genes for CHROM {chroms_na}."
        # Log the error message
        print(msg)
        logging.error(msg)
        # Raise an exception
        raise ValueError(msg)

    return bulk