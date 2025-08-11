#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:06:08 2024

@author: lillux
"""
import numpy as np
from spacenumbat.dist_prob import dnbinom, dpoilog, log_beta_binomial_pmf
from typing import List, Optional, Dict, Any
import pandas as pd

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("This is an info message.")



def viterbi_loh(HMM:Dict):
    """
    Run Viterbi decoding in log-space for a joint LOH HMM and return the most
    likely sequence of state *labels* (not state indices).

    The recursion combines:
    - an allele-count emission term (via `dnbinom`, evaluated at x[t] with mean
      `pm * pn[t]` and dispersion `snp_sig`), and
    - an expression-count emission term (via `dpoilog`, evaluated at y[t] with
      Poisson–lognormal parameters `mu + log(phi * d * lambda_star[t])` and `sig`).

    Transition probabilities are provided per position as a 3D tensor `Pi`
    (shape (m, m, n)), and all computations are done in log-space.

    Parameters
    ----------
    HMM : Dict
        Mapping containing observations, parameters, transition and initial
        probabilities, and the list of state labels.
        
        Required keys
        -------------
        x : NDArray[np.int_]
            Length-n array of allele-depth–derived counts (per position).
        y : NDArray[np.int_]
            Length-n array of total-expression–derived counts (per position).
        Pi : NDArray[np.float_]
            Transition probability tensor of shape (m, m, n).
        delta : NDArray[np.float_]
            Initial state distribution of shape (m,).
        pm : float | NDArray[np.float_]
            Mean parameter (or vector) for the allele model (paired with `pn`).
        pn : NDArray[np.float_]
            Length-n vector used with `pm` to form the per-position allele mean `pm * pn[t]`.
        snp_sig : float
            Dispersion/size parameter for the Beta-Binomial/Negative Binomial allele model.
        sig : float
            Dispersion (sigma) for the Poisson–lognormal expression model.
        mu : float
            Baseline log-mean for the Poisson–lognormal expression model.
        phi : float
            Expression fold-change (multiplicative) parameter.
        d : float
            Library-size / depth scaling factor for the expression model.
        lambda_star : NDArray[np.float_]
            Length-n baseline rate modifiers for expression.
        states : Sequence[Any]
            Sequence of state labels (length m). The returned Viterbi path will be
            these labels indexed by the decoded argmax path.

    Returns
    -------
    np.ndarray
        Array of length n with the decoded Viterbi path **labels**,
        obtained by indexing `HMM['states']` with the argmax state indices.

    Raises
    ------
    ValueError
        If array shapes are inconsistent (e.g., `Pi` not (m, m, n), or vector
        lengths do not equal `n`).
    """
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


def viterbi_compute(log_delta: np.ndarray, 
                    logprob: np.ndarray, 
                    logPi: np.ndarray) -> np.ndarray:
    """
    Viterbi algorithm for a Hidden Markov Model (HMM) with allele counts.
    
    This function computes the most likely sequence of hidden states
    for the given HMM parameters.
    
    Parameters
    ----------
    log_delta : np.ndarray
        Log of the initial state probabilities. Shape: (M,).
    logprob : np.ndarray
        Log probabilities of observations at each time step.
        Shape: (N, M) where N is the number of observations and M the number of states.
    logPi : np.ndarray
        Log of the transition probability matrices.
        Shape: (N, M, M), where for each time step i, logPi[i, :, :] gives the 
        log transition probabilities from each state to each state.
    
    Returns
    -------
    np.ndarray
        Decoded state indices (as integers) of length N representing the most
        likely state at each time step.
    
    Notes
    -----
    The algorithm performs:
      - A forward pass to compute the dynamic programming matrix 'nu'
        where nu[i, j] is the log probability of the most likely path
        ending in state j at time i.
      - A backtracking pass that recovers the most likely state sequence.
    """
    N, M = logprob.shape
    nu = np.zeros((N, M))
    z = np.zeros(N, dtype=np.int64)

    # Initialization
    nu[0, :] = log_delta + logprob[0, :]

    # Forward pass
    for i in range(1, N):
        # Reshape previous step (M,) into (M,1) for broadcasting
        nu_prev = nu[i - 1, :].reshape(M, 1)
        # Get the transition matrix for time i (M x M)
        logPi_i = logPi[i, :, :]
        # Compute the "score" matrix for all transitions from previous states.
        # The broadcasting adds nu_prev (M x 1) to each column of logPi_i (M x M).
        sum_matrix = nu_prev + logPi_i
        # For each destination state, choose the max incoming log probability.
        nu[i, :] = np.max(sum_matrix, axis=0) + logprob[i, :]

    # Backtracking
    z[N - 1] = np.argmax(nu[N - 1, :])
    for i in range(N - 2, -1, -1):
        # For time i+1, pick the transition into state z[i+1] that maximized the probability
        logPi_i1 = logPi[i + 1, :, :]
        z[i] = np.argmax(logPi_i1[:, z[i + 1]] + nu[i, :])

    return z


def viterbi_allele(hmm):
    """
    Viterbi algorithm for allele HMM. This function builds the log observation 
    probabilities from beta-binomial models and then decodes the most likely state 
    sequence using the Viterbi algorithm.
    
    Parameters
    ----------
    hmm : dict
        A dictionary of HMM parameters. Expected keys:
            'x'      : np.ndarray, allele counts.
            'd'      : np.ndarray, total allele counts.
            'logPi'  : np.ndarray, log transition probability matrices (shape: (N, M, M)).
            'delta'  : np.ndarray, initial state probabilities.
            'alpha'  : np.ndarray, alpha parameters for the beta-binomial distribution 
                       (shape: (N, M)).
            'beta'   : np.ndarray, beta parameters for the beta-binomial distribution 
                       (shape: (N, M)).
            'N'      : int, number of observations.
            'M'      : int, number of states.
            'states' : List[str], names of the states.
    
    Returns
    -------
    List[str]
        A list of decoded state names for each observation.
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

    # Handle NaN values
    logprob = np.nan_to_num(logprob, nan=0.0)
    # Run Viterbi algorithm
    z = viterbi_compute(np.log(delta), logprob, logPi)
    # Map indices to state names
    decoded_states = [states[idx] for idx in z]

    return decoded_states


def run_allele_hmm_s5(pAD: np.ndarray, 
                      DP: np.ndarray, 
                      p_s: np.ndarray, 
                      t: float = 1e-5, 
                      theta_min: float = 0.08, 
                      theta_max: float = 0.4, 
                      gamma: float = 20, 
                      prior: Optional[np.ndarray] = None) -> List[str]:
    """
    Run a 5-state allele-only Hidden Markov Model (HMM) with two theta levels.
    
    The model has the following states:
      - "neu" (neutral)
      - "theta_1_up"
      - "theta_1_down"
      - "theta_2_up"
      - "theta_2_down"
      
    The function builds the transition matrices, computes beta-binomial emission parameters 
    based on the provided theta thresholds and gamma value, and then decodes the most likely 
    state sequence using the Viterbi algorithm.
    
    Parameters
    ----------
    pAD : np.ndarray
        Paternal allele counts.
    DP : np.ndarray
        Total allele counts.
    p_s : np.ndarray
        Phase switch probabilities.
    t : float
        Transition probability between copy number states (default: 1e-5).
    theta_min : float
        Minimum haplotype frequency deviation threshold (default: 0.08).
    theta_max : float
        Maximum haplotype frequency deviation threshold (default: 0.4).
    gamma : float
        Overdispersion parameter for the beta-binomial model. A higher gamma yields a more peaked 
        Beta distribution (default: 20).
    prior : Optional[np.ndarray]
        Prior probabilities for each state (default: None, which results in a uniform distribution).
    
    Returns
    -------
    List[str]
        A list of decoded state names for each observation.
    """
    # Ensure gamma is a single value
    gamma_arr = np.unique(np.array(gamma))
    if len(gamma_arr) > 1:
        raise ValueError('More than one gamma parameter')
    
    # Define the possible states for the HMM.
    states = ["neu", "theta_1_up", "theta_1_down", "theta_2_up", "theta_2_down"]
    N = len(pAD)
    M = 5

    # Function to compute the transition matrix for the 5-state model.
    def calc_trans_mat_s5(p_s_i: float, t_val: float) -> np.ndarray:
        return np.array([
            [1 - t_val, t_val / 4, t_val / 4, t_val / 4, t_val / 4],
            [t_val / 2, (1 - t_val) * (1 - p_s_i), (1 - t_val) * p_s_i, t_val / 4, t_val / 4],
            [t_val / 2, (1 - t_val) * p_s_i, (1 - t_val) * (1 - p_s_i), t_val / 4, t_val / 4],
            [t_val / 2, t_val / 4, t_val / 4, (1 - t_val) * (1 - p_s_i), (1 - t_val) * p_s_i],
            [t_val / 2, t_val / 4, t_val / 4, (1 - t_val) * p_s_i, (1 - t_val) * (1 - p_s_i)]
        ])

    # Build log transition matrices for each observation.
    logPi = np.zeros((N, M, M))
    for i in range(N):
        trans_mat = calc_trans_mat_s5(p_s[i], t)
        logPi[i] = np.log(trans_mat)
    
    # Set uniform prior if not provided.
    if prior is None:
        prior = np.full(M, 1 / M)
    
    # Compute beta-binomial parameters.
    theta_1 = theta_min
    theta_2 = theta_max
    alphas = gamma * np.array([0.5, 0.5 + theta_1, 0.5 - theta_1, 0.5 + theta_2, 0.5 - theta_2])
    betas = gamma * np.array([0.5, 0.5 - theta_1, 0.5 + theta_1, 0.5 - theta_2, 0.5 + theta_2])

    alpha_mat = np.tile(alphas, (N, 1))
    beta_mat = np.tile(betas, (N, 1))

    # Assemble the HMM parameters into a dictionary.
    hmm: Dict[str, Any] = {
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

    # Run the Viterbi algorithm for allele HMM decoding.
    mpc = viterbi_allele(hmm)
    return mpc


def smooth_segs(bulk: pd.DataFrame, min_genes: int = 10) -> pd.DataFrame:
    """
    Smooth CNV segment states after HMM decoding by removing small segments and
    propagating neighboring states within each chromosome.

    This function treats segments with too few genes as unreliable, clears their
    CNV state, and then forward-/back-fills the remaining states per chromosome.

    Parameters
    ----------
    bulk : pd.DataFrame
        Pseudobulk profile. Must contain at least the following columns:
        - 'seg' (hashable): segment identifier for each row.
        - 'n_genes' (int): number of genes in the segment indicated by 'seg'.
          Values are assumed to be constant within a segment (the function uses
          the first value per segment).
        - 'cnv_state' (object / string): categorical/label state produced by the HMM.
        - 'CHROM' (object / string / int): chromosome identifier used to limit
          the fill operations within a chromosome.
    min_genes : int, default=10
        Minimum number of genes required for a segment to be considered valid.
        Segments with ``n_genes <= min_genes`` will have their 'cnv_state' set
        to NaN before smoothing.

    Returns
    -------
    pd.DataFrame
        A copy of `bulk` with 'cnv_state' smoothed:
        - 'cnv_state' is set to NaN for short segments,
        - then forward-filled and back-filled within each chromosome.

    Raises
    ------
    ValueError
        If any chromosome ends up with all-NaN 'cnv_state' after clearing short
        segments (i.e., no segment with more than ``min_genes`` genes in that
        chromosome). The error message lists the affected chromosomes.

    Notes
    -----
    - The operation is non-destructive: a copy of `bulk` is returned.
    - Groupby operations use ``observed=True`` and ``sort=False`` to preserve
      the original ordering behavior and avoid adding unused categories.
    """
    # Copy the DataFrame to avoid modifying the original
    bulk = bulk.copy()
    # Within each segment, set 'cnv_state' to NaN if 'n_genes' <= min_genes
    # Get the number of genes per segment
    n_genes_per_seg = bulk.groupby('seg', observed=True, sort=False)['n_genes'].first().reset_index()
    # Identify segments with insufficient genes
    small_segs = n_genes_per_seg.loc[n_genes_per_seg['n_genes'] <= min_genes, 'seg']
    # Set 'cnv_state' to NaN for these segments
    bulk.loc[bulk['seg'].isin(small_segs), 'cnv_state'] = np.nan
    # Fill NaN values in 'cnv_state' forward and backward within each chromosome
    bulk['cnv_state'] = bulk.groupby('CHROM', observed=True, sort=False)['cnv_state'].ffill().bfill()
    # Check if any chromosome has all NaN in 'cnv_state'
    chrom_na = bulk.groupby('CHROM', observed=True, sort=False)['cnv_state'].apply(lambda x: x.isna().all()).reset_index(name='all_na')

    # THIS RAISE ERROR IF FEW GENES ARE FOUND IN A CHROMOSOME
    if chrom_na['all_na'].any():
        chroms_na = ','.join(chrom_na.loc[chrom_na['all_na'], 'CHROM'].astype(str))
        # Log the error message
        msg = f"No segments containing more than {min_genes} genes for CHROM {chroms_na}."
        log.error(msg)
        # Raise an exception
        raise ValueError(msg)

    return bulk