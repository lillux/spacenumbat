#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 17:06:08 2024

@author: lillux
"""
import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import List, Optional, Dict, Any, Mapping, Sequence, Union
import pandas as pd
from spacenumbat.dist_prob import dnbinom, dpoilog, log_beta_binomial_pmf
from spacenumbat import utils

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("This is an info message.")



def viterbi_loh(HMM:Mapping[str, Any]):
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
    HMM : Mapping[str, Any]
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


def viterbi_allele(hmm:Mapping[str, Any]):
    """
    Viterbi algorithm for allele HMM. This function builds the log observation 
    probabilities from beta-binomial models and then decodes the most likely state 
    sequence using the Viterbi algorithm.
    
    Parameters
    ----------
    hmm : Mapping[str, Any]
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
    bulk.seg = bulk.seg.astype("string") # TODO check validity of conversion
    n_genes_per_seg = bulk.groupby('seg', observed=True, sort=False)['n_genes'].first().reset_index()
    # Identify segments with insufficient genes
    small_segs = n_genes_per_seg.loc[n_genes_per_seg['n_genes'] <= min_genes, 'seg']
    # Set 'cnv_state' to NaN for these segments
    bulk.loc[bulk['seg'].isin(small_segs), 'cnv_state'] = np.nan
    # Fill NaN values in 'cnv_state' forward and backward within each chromosome
    bulk['cnv_state'] = bulk.groupby('CHROM', observed=True, sort=False)['cnv_state'].ffill().bfill()
    # Check if any chromosome has all NaN in 'cnv_state'
    bulk.CHROM = bulk.CHROM.astype("string") # TODO check validity of conversion
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


def viterbi_joint(hmm: Mapping[str, Any]) -> NDArray[np.integer]:
    """
    Viterbi decoding for the joint (allele + expression) HMM.

    This function computes per-state log-emission probabilities by combining:
      1) a **Beta–Binomial** (via ``log_beta_binomial_pmf``) for allele counts, and
      2) a **Poisson–lognormal** (via ``dpoilog``) for expression counts,
    then runs a log-space Viterbi recursion using the provided initial state
    probabilities and (genomic coordinates-varying) transition log-probabilities.

    Parameters
    ----------
    hmm : Mapping[str, Any]
        A dictionary-like container with the following required keys and shapes
        (N = number of observations, M = number of states):

        Allele model (required)
        - "x" : NDArray[int], shape (N,)
            Alternative (or minor) allele counts per position.
        - "d" : NDArray[int], shape (N,)
            Total allele depths per position.
        - "alpha" : NDArray[float], shape (N, M) or (1, M)
            Alpha parameters (can be coordinates-varying or broadcastable over N).
        - "beta" : NDArray[float], shape (N, M) or (1, M)
            Beta parameters (can be coordinates-varying or broadcastable over N).

        Transitions / initialization (required)
        - "delta" : NDArray[float], shape (M,)
            Initial state probabilities (must be strictly positive).
        - "logPi" : NDArray[float], shape (N, M, M)
            Transition **log**-probabilities for each coordinates step *t*:
            "logPi[t, i, j] = log P(z_t = j | z_{t-1} = i)".
            (The code treats these as log-probabilities and does **not** take logs again.)

        Expression model (optional; used only if present)
        - "y" : NDArray[int], shape (N,)
            Expression counts per position.
        - "l" : NDArray[float], shape (N,)
            Per-position scaling factor (gene length).
        - "lambda" : NDArray[float], shape (N,)
            Baseline expression rate modifier (reference expression).
        - "mu" : NDArray[float], shape (N,)
            Baseline log-mean for the Poisson–lognormal.
        - "sig" : NDArray[float], shape (N,)
            Lognormal standard deviation (sigma).
        - "phi" : NDArray[float], shape (M,)
            State-specific multiplicative fold-change applied to expression.

        Misc (optional)
        - "states" : Sequence[Any]
            State labels; not used here, but often paired with the Viterbi path.

    Returns
    -------
    numpy.ndarray of int
        The decoded Viterbi state sequence "z" of length N.

    Notes
    -----
    - All log-domain inputs must be finite (no zeros before taking logs).
    - NaNs in the allele/emission terms are treated as 0 contribution via
      "np.nan_to_num(..., nan=0.0)" in the allele component.
    - If any expression key (e.g., "y") is absent or "None", only the
      allele component contributes to the emission log-probability.

    Examples
    --------
    >>> z = viterbi_joint({
    ...     "x": x, "d": d, "alpha": alpha, "beta": beta,
    ...     "delta": delta, "logPi": logPi,
    ...     "y": y, "l": L, "lambda": lam, "mu": mu, "sig": sig, "phi": phi
    ... })
    """
    # N = number of observations
    x = hmm["x"]              # shape (N,)
    d = hmm["d"]              # shape (N,)
    alpha = hmm["alpha"]      # expected shape (..., M) or (N, M) if time-varying
    beta  = hmm["beta"]       # same shape as alpha
    delta = hmm["delta"]      # shape (M,)
    logPi = hmm["logPi"]      # shape (N, M, M)
    
    N = len(x)
    M = logPi.shape[1]  # number of states
    
    # If we have expression data (y, l, lambda, mu, sig, phi), handle that
    has_expression = "y" in hmm and hmm["y"] is not None
    
    # Compute logprob[i, m] for each observation i and state m
    logprob = np.zeros((N, M))
    for m in range(M):
        # Beta-Binomial
        l_x = log_beta_binomial_pmf(x, d, alpha[:, m], beta[:, m])  
        # Replace NaNs with 0
        l_x = np.nan_to_num(l_x, nan=0.0)
        
        if has_expression:
            y     = hmm["y"]
            valid = ~np.isnan(y)
            l_y   = np.zeros(N)
            # calculate Poisson-Log distribution. 
            l_y[valid] = dpoilog(
                x=y[valid],
                mu=hmm["mu"][valid] + np.log(hmm["phi"][m] * hmm["l"][valid] * hmm["lambda"][valid]),
                sig=hmm["sig"][valid],
                log=True
            )
        else:
            l_y = 0
        
        logprob[:, m] = l_x + l_y
    
    z = viterbi_compute(
        log_delta = np.log(delta),
        logprob   = logprob,
        logPi     = logPi
    )
    
    return z


def get_trans_probs_s15(
    t: float,
    p_s: ArrayLike | float,
    w: Mapping[str, float],
    cn_from: str,
    phase_from: Optional[str],
    cn_to: str,
    phase_to: Optional[str],
    ) -> np.ndarray:
    """
    Compute transition probabilities for the 15-state joint HMM for a single
    (from_state -> to_state) pair, optionally across multiple phase-switch
    probabilities.

    Parameters
    ----------
    t: float
        CNV state transition probability (scalar in [0, 1]).
    p_s: ArrayLike | float
        Phase switch probability; scalar or 1-D array. The function returns one
        probability per element of "p_s".
    w: Mapping[str, float]
        Relative abundances (mixture weights) per CNV state name, e.g.
        {"neu": 0.5, "del_1": 0.1, ...}. Used to allocate mass when the
        CNV state changes.
    cn_from: str
        Origin CNV state label (e.g., "neu", "del_1", "loh_2").
    phase_from: Optional[str]
        Origin phase label ("up", "down", or None when phase is not
        applicable for the CNV state).
    cn_to: str
        Destination CNV state label.
    phase_to: Optional[str]
        Destination phase label ("up", "down", or None).

    Returns
    -------
    np.ndarray
        1-D array of shape (len(p_s),) with transition probabilities for the
        specified (from, to) pair. If "p_s" is a scalar, the shape is (1,).

    Notes
    -----
    - Special case: for neu -> neu, the effective phase switch probability is
      forced to 0.5.
    - When cn_from == cn_to:
        * No phase (both None): probability is "1 - t".
        * Same phase: (1 - t) * (1 - p_s).
        * Different phase: (1 - t) * p_s.
    - When cn_from != cn_to:
        * Mass "t" is distributed to other CNV states in proportion to their
          weights "w[cn_to]" (normalized by the sum of weights excluding "cn_from").
        * If the destination has a phase, the probability is split equally
          between its two phase labels (division by 2).

    Raises
    ------
    KeyError
        If "cn_to" (or "cn_from") is missing from "w".
    ZeroDivisionError
        If the weight denominator (sum of weights excluding "cn_from") is zero.
    """
    p_s = np.array(p_s, ndmin=1)  # ensure array
    
    # Special case: if going from neu -> neu, then p_s = 0.5
    if cn_from == 'neu' and cn_to == 'neu':
        p_s = np.full_like(p_s, 0.5)
    
    if cn_from == cn_to:
        if phase_from is None and phase_to is None:
            p = 1.0 - t
            p_out = np.full_like(p_s, p)
        elif phase_from == phase_to:
            p_out = (1.0 - t) * (1.0 - p_s)
        else:
            p_out = (1.0 - t) * p_s
    else:
        denom = sum([v for k,v in w.items() if k != cn_from])
        p_cnv = t * (w[cn_to] / denom)
        
        # If there's a phase_to, we split in half
        if phase_to is not None:
            p_cnv = p_cnv / 2.0
        
        p_out = np.full_like(p_s, p_cnv)
    
    return p_out


def calc_trans_mat_s15(
    t: float,
    p_s: ArrayLike | float,
    w: Mapping[str, float],
    states_cn: Sequence[str],
    states_phase: Sequence[Optional[str]],
    ) -> np.ndarray:
    """
    Build the full transition tensor for the 15-state joint HMM across one or
    more phase-switch probabilities.

    Parameters
    ----------
    t: float
        CNV state transition probability (scalar in [0, 1]).
    p_s: ArrayLike | float
        Phase switch probability; scalar or 1-D array. The output contains one
        transition matrix per element of "p_s".
    w: Mapping[str, float]
        Relative abundances (mixture weights) per CNV state name.
    states_cn: Sequence[str]
        Sequence of CNV state labels (length "n_states"), aligned with
        "states_phase". Each pair "(states_cn[i], states_phase[i])" defines
        one HMM state.
    states_phase
        Sequence of phase labels ("up", "down", or None) aligned to states_cn.

    Returns
    -------
    np.ndarray
        Transition tensor "A" of shape "(n_states, n_states, len(p_s))",
        where "A[i, j, k]" is the transition probability from state "i" to
        state "j" given "p_s[k]".

    Raises
    ------
    ValueError
        If ``states_cn`` and ``states_phase`` differ in length.
    """
    n_states = len(states_cn)
    p_s = np.array(p_s, ndmin=1)  # ensure array
    n_ps = len(p_s)
    # build a (n_states, n_states, n_ps) array
    A = np.zeros((n_states, n_states, n_ps), dtype=float)

    for i in range(n_states):
        for j in range(n_states):
            trans_ij = get_trans_probs_s15(
                t         = t,
                p_s       = p_s,
                w         = w,
                cn_from   = states_cn[i],
                phase_from= states_phase[i],
                cn_to     = states_cn[j],
                phase_to  = states_phase[j]
            )
            # trans_ij is shape (n_ps,)
            A[i, j, :] = trans_ij

    return A


def run_joint_hmm_s15(
    pAD: ArrayLike,
    DP: ArrayLike,
    p_s: Union[ArrayLike, float],
    Y_obs: Optional[ArrayLike] = None,
    lambda_ref: Optional[ArrayLike] = None,
    d_total: Optional[ArrayLike] = None,
    theta_min: float = 0.08,
    theta_neu: float = 0.0,
    bal_cnv: bool = True,
    phi_del: float = 2 ** (-0.25),
    phi_amp: float = 2 ** 0.25,
    phi_bamp: Optional[float] = None,
    phi_bdel: Optional[float] = None,
    mu: Union[float, ArrayLike] = 0.0,
    sig: Union[float, ArrayLike] = 1.0,
    t: float = 1e-5,
    gamma: float = 18,
    prior: Optional[ArrayLike] = None,
    exp_only: bool = False,
    allele_only: bool = False,
    classify_allele: bool = False,
    debug: bool = False,
    ) -> List[str]:
    """
    Decode CNV states with the 15-state joint HMM (allele + expression).

    This wrapper builds a 15-state model (optionally reduced based on flags),
    assembles emission/transition parameters, runs a Viterbi routine, and
    returns the most probable state labels per position.

    Model states
    ------------
    The full model includes 15 states:
        neu,
        del_1_(up|down), del_2_(up|down),
        loh_1_(up|down), loh_2_(up|down),
        amp_1_(up|down), amp_2_(up|down),
        bamp, bdel
    Flags can reduce this set:
        - bal_cnv=False    -> drop {bamp, bdel}
        - allele_only=True -> keep {neu, loh_1_(up|down), loh_2_(up|down)}
        - classify_allele  -> keep {loh_1_up, loh_1_down} only
        - exp_only=True    -> expression-only mode (allele channels disabled)

    Parameters
    ----------
    pAD : ArrayLike
        Allelic fraction per position (minor-allele fraction); length N.
    DP : ArrayLike
        Total allele depth per position; length N.
    p_s : ArrayLike or float
        Phase-switch probability per position (length N) or a scalar.
        Used when constructing transition tensors.
    Y_obs : ArrayLike, optional
        Expression counts per position; length N. If None, filled with zeros.
    lambda_ref : ArrayLike, optional
        Baseline expression rate modifiers per position; length N. If None, zeros.
    d_total : ArrayLike, optional
        Library / depth scaling per position; length N or scalar. If None, zeros.
    theta_min : float, default=0.08
        Minimum allelic imbalance for single-copy LOH/DEL/AMP (affects Beta–Binomial α/β).
    theta_neu : float, default=0.0
        Neutral allelic offset (affects neutral α/β).
    bal_cnv : bool, default=True
        If False, removes balanced CNV states (bamp/bdel) from the state space.
    phi_del : float, default=2**(-0.25)
        Fold-change for single-copy deletion in expression emissions.
    phi_amp : float, default=2**0.25
        Fold-change for single-copy amplification in expression emissions.
    phi_bamp : float, optional
        Fold-change for balanced amplification state; defaults to `phi_amp` if None.
    phi_bdel : float, optional
        Fold-change for balanced deletion state; defaults to `phi_del` if None.
    mu : float or ArrayLike, default=0.0
        Baseline log-mean for the Poisson–lognormal expression model; scalar or length N.
    sig : float or ArrayLike, default=1.0
        Lognormal sigma for the expression model; scalar or length N.
    t : float, default=1e-5
        CNV state transition probability.
    gamma : float, default=18
        Dispersion scale for the Beta–Binomial (α = γ * θ, β = γ * (1-θ)).
    prior : ArrayLike, optional
        Initial state probabilities over the active state set. If None, a heuristic
        prior is constructed from transitions starting at 'neu'.
    exp_only : bool, default=False
        Expression-only mode: disables allele channel (pAD set to NaN; p_s set to 0).
    allele_only : bool, default=False
        Allele-only mode: restricts states to {neu, loh_1_(up|down), loh_2_(up|down)}
        and disables expression channel (Y_obs set to NaN).
    classify_allele : bool, default=False
        Keep only {loh_1_up, loh_1_down} for allele-classification tasks.
    debug : bool, default=False
        Reserved for debugging; not used here.

    Returns
    -------
    list[str]
        Most probable state labels (length N), e.g., ``['neu', 'del_1_up', ...]``.

    Notes
    -----
    - Inputs are converted to numpy arrays; scalars for `mu`, `sig`, or `d_total`
      are broadcast to length N.
    - Transition tensor `A` is built for the full 15-state set and then subset to
      the active states. It is converted to log-space (`logPi`) for Viterbi.
    - Balanced CNV fold-changes default to the corresponding single-copy values
      if not provided (``phi_bamp <- phi_amp``, ``phi_bdel <- phi_del``).
      
    Raises
    ------
    KeyError
        If a CNV state required for transition weighting is missing from `w`.
    ValueError
        May be raised downstream for inconsistent shapes in helper functions.
    """

    # Default arguments
    if Y_obs is None:
        Y_obs = np.zeros_like(pAD, dtype=float)
    if lambda_ref is None:
        lambda_ref = np.zeros_like(pAD, dtype=float)
    if d_total is None:
        d_total = np.zeros_like(pAD, dtype=int)
    if phi_bamp is None:
        phi_bamp = phi_amp
    if phi_bdel is None:
        phi_bdel = phi_del
    
    # Define the 15 states
    states = ["neu",
              "del_1_up", "del_1_down", "del_2_up", "del_2_down",
              "loh_1_up", "loh_1_down", "loh_2_up", "loh_2_down",
              "amp_1_up", "amp_1_down", "amp_2_up", "amp_2_down",
              "bamp", "bdel"]

    # Extract CN parts and up/down from state names
    states_cn = [utils.remove_up_down(s) for s in states] 
    states_phase = [utils.extract_up_down(s) for s in states]

    # Relative abundance of states (w)
    w = {"neu":   1,
         "del_1": 1,
         "del_2": 1e-10,
         "loh_1": 1,
         "loh_2": 1e-10,
         "amp_1": 1,
         "amp_2": 1e-10,
         "bamp":  1e-4,
         "bdel":  1e-10,}

    # If `prior` is None, build an initial prior (one for each of the 15 states).
    if prior is None:
        prior = []
        for i, st in enumerate(states):
            cn_to = utils.remove_up_down(st)
            phase_to = utils.extract_up_down(st)
            t_inflate = min(t * 100, 1.0)
            p_init = get_trans_probs_s15(
                t=t_inflate,
                p_s=0.0,
                w=w,
                cn_from='neu',
                phase_from=None,
                cn_to=cn_to,
                phase_to=phase_to)
            # p_init is array shape (1,); we want scalar
            prior.append(p_init[0])
        prior = np.array(prior, dtype=float)

    # Possibly drop states
    # The code reindexes states according to bal_cnv, exp_only, allele_only, classify_allele.
    states_index = np.arange(len(states))  # 0..14
    if not bal_cnv:
        states_index = np.arange(13)  # drop indices 13,14 (bamp, bdel)
    if exp_only:
        pAD = np.full_like(pAD, np.nan, dtype=float)
        p_s = np.zeros_like(p_s, dtype=float)
    if allele_only:
        states_index = np.array([0, 5, 6, 7, 8], dtype=int)
        Y_obs = np.full_like(Y_obs, np.nan, dtype=float)
    if classify_allele:
        states_index = np.array([5, 6], dtype=int)

    # Subset
    prior         = prior[states_index]
    states_sub    = [states[i] for i in states_index]
    #states_cn_sub = [states_cn[i] for i in states_index]
    #states_ph_sub = [states_phase[i] for i in states_index]

    # Build the 3D transition matrix
    # Then subset
    As_full = calc_trans_mat_s15(t, p_s, w, states_cn, states_phase)  # shape (15,15,len(p_s))
    As_sub  = As_full[states_index][:, states_index, :]

    # Define allele parameters for each of the 15 states
    theta_u_1   = 0.5 + theta_min
    theta_d_1   = 0.5 - theta_min
    theta_u_2   = 0.9
    theta_d_2   = 0.1
    theta_u_neu = 0.5 + theta_neu
    theta_d_neu = 0.5 - theta_neu

    alpha_states = gamma * np.array([
        theta_u_neu,
        theta_u_1,  theta_d_1,  theta_u_2,  theta_d_2,
        theta_u_1,  theta_d_1,  theta_u_2,  theta_d_2,
        theta_u_1,  theta_d_1,  theta_u_2,  theta_d_2,
        theta_u_neu,
        theta_u_neu
    ], dtype=float)

    beta_states = gamma * np.array([
        theta_d_neu,
        theta_d_1,  theta_u_1,  theta_d_2,  theta_u_2,
        theta_d_1,  theta_u_1,  theta_d_2,  theta_u_2,
        theta_d_1,  theta_u_1,  theta_d_2,  theta_u_2,
        theta_d_neu,
        theta_d_neu
    ], dtype=float)

    # Expression fold changes:
    phi_vec = np.array([
        1.0,         # neu
        phi_del,     # del_1_up
        phi_del,     # del_1_down
        0.5,         # del_2_up
        0.5,         # del_2_down
        1.0,         # loh_1_up
        1.0,         # loh_1_down
        1.0,         # loh_2_up
        1.0,         # loh_2_down
        phi_amp,     # amp_1_up
        phi_amp,     # amp_1_down
        2.5,         # amp_2_up
        2.5,         # amp_2_down
        phi_bamp,    # bamp
        phi_bdel     # bdel
    ], dtype=float)

    # Subset these to states_index
    alpha_states_sub = alpha_states[states_index]
    beta_states_sub  = beta_states[states_index]
    phi_states_sub   = phi_vec[states_index]

    # Build the final HMM dictionary
    #    N = len(Y_obs)
    #    M = number of chosen states
    pAD = np.array(pAD, dtype=float)  # ensure float or nan
    DP  = np.array(DP,  dtype=float)
    Y_obs = np.array(Y_obs, dtype=float)
    d_total = np.array(d_total, dtype=float)  # library sizes

    N = len(Y_obs)
    #M = len(states_sub)

    # Expand mu, sig if necessary
    mu_arr  = np.array(mu,  ndmin=1, dtype=float)
    sig_arr = np.array(sig, ndmin=1, dtype=float)
    if mu_arr.size == 1:
        mu_arr  = np.full(N, mu_arr[0],  dtype=float)
        sig_arr = np.full(N, sig_arr[0], dtype=float)

    # Expand d_total if needed
    if d_total.size == 1:
        d_total = np.full(N, d_total[0], dtype=float)

    # Build alpha/beta as an (N x M) matrix each:
    alpha_mat = np.tile(alpha_states_sub, (N, 1))  # shape (N, M)
    beta_mat  = np.tile(beta_states_sub,  (N, 1))  # shape (N, M)

    # Build final logPi by taking log of As_sub
    As_sub_np = np.array(As_sub)  # shape (M, M, N)
    As_sub_reordered = np.moveaxis(As_sub_np, 2, 0)  # shape (N, M, M)
    logPi = np.log(As_sub_reordered, out=np.zeros_like(As_sub_reordered))

    # Construct the HMM dictionary
    hmm = {"x":      pAD,
           "d":      DP,
           "y":      Y_obs,
           "l":      d_total,
           "lambda": np.array(lambda_ref, dtype=float),
           "mu":     mu_arr,
           "sig":    sig_arr,
           "logPi":  logPi,         # shape (N, M, M)
           "phi":    phi_states_sub,
           "delta":  prior,
           "alpha":  alpha_mat,     # shape (N, M)
           "beta":   beta_mat,      # shape (N, M)
           "states": states_sub,
           "p_s":    p_s}

    # Call Viterbi algorithm:
    z_idx = viterbi_joint(hmm)

    # Map indices back to the state names
    MPC = [states_sub[i] for i in z_idx]
    return MPC


def log_sum_exp(vals: NDArray[np.floating]) -> float:
    """
    Compute log(sum(exp(vals))) in a numerically stable way.

    This function uses the standard trick of subtracting the maximum entry
    before exponentiation to avoid overflow/underflow:

      log sum_i exp(v_i) = v_max + log sum_i exp(v_i - v_max)

    Special cases:
    - If vals is empty, returns -np.inf (log of zero).
    - If the maximum value is non-finite (for example, all entries are -inf),
      that maximum is returned (for example, -inf).

    Parameters
    ----------
    vals : numpy.ndarray, shape (n,)
        One-dimensional array of real values in log-space.

    Returns
    -------
    float
        The scalar value log(sum(exp(vals))) computed stably.
    """
    if len(vals) == 0:
        return -np.inf
    max_val = np.max(vals)
    if not np.isfinite(max_val):
        return max_val  # e.g. -inf if all -inf
    cumsum = np.sum(np.exp(vals - max_val))
    return max_val + np.log(cumsum)


def likelihood_compute(logphi: NDArray, logprob: NDArray, logPi: NDArray,) -> float:
    """
    Compute the total log-likelihood of a time-inhomogeneous HMM in log-space.

    This implements the forward algorithm with per-time (coordinates), per-state scaling.
    At each time step i, it:
      1) (for i > 0) propagates the previous log-forward vector
         through the time-i transition matrix in log-space;
      2) adds the emission log-likelihoods logprob[i, :];
      3) normalizes by subtracting logSumExp to keep numbers stable and
         accumulates that normalizer into the running total log-likelihood.

    Shapes and semantics
    --------------------
    - logphi : shape (m,)
        Initial log state probabilities at time i = 0 (that is, log(delta)).
    - logprob : shape (n, m)
        Emission log-likelihoods where logprob[i, s] is the log-probability
        of the observation at time i given state s.
    - logPi : shape (n, m, m)
        Time-dependent log transition probabilities. For i > 0,
        logPi[i, r, c] equals log P(S_i = c | S_{i-1} = r).
        The slice logPi[0, ...] is not used by this implementation.

    Parameters
    ----------
    logphi : numpy.ndarray, shape (m,)
        Initial log state distribution.
    logprob : numpy.ndarray, shape (n, m)
        Emission log-likelihoods per time and state.
    logPi : numpy.ndarray, shape (n, m, m)
        Log transition matrices per time step.

    Returns
    -------
    float
        The total log-likelihood of the observation sequence under the HMM.

    Notes
    -----
    - All computations are performed in log-space for numerical stability.
    - The per-time normalization and accumulation scheme (subtracting
      logSumExp from the forward vector and adding it to LL)
      is equivalent to computing the log-likelihood without scaling.
    - This function assumes logprob and logPi are consistent in their
      time (coordinates) indexing and that probabilities (in non-log space) are valid
      (rows of transition matrices sum to 1, entries are non-negative).
    """
    n, m = logprob.shape
    LL = 0.0

    # keep track of current logphi
    curr_logphi = logphi.copy()

    for i in range(n):
        if i > 0:
            # compute next logphi
            next_logphi = np.full(m, -np.inf, dtype=float)
            # subset_logPi = logPi[t, :, :] => shape (m,m)
            subset_logPi = logPi[i, :, :]
            # For each c: logphi_new[c] = logSumExp( logphi + subset_logPi[:, c] )
            for c in range(m):
                # elementwise sum of logphi[r] + subset_logPi[r, c]
                sums = curr_logphi + subset_logPi[:, c]
                next_logphi[c] = log_sum_exp(sums)
            curr_logphi = next_logphi

        # add the observation log-likelihood:  logprob[i, :]
        curr_logphi = curr_logphi + logprob[i, :]
        # normalize
        logSumPhi = log_sum_exp(curr_logphi)
        curr_logphi = curr_logphi - logSumPhi
        # accumulate
        LL += logSumPhi

    return LL


def likelihood_allele(hmm: Mapping[str, Any]) -> float:
    """
    Compute the total log-likelihood of a 2-state allele HMM under a
    Beta-Binomial emission model and time/position-dependent transitions.

    The HMM is passed as a mapping with the following required entries:

    - "x": 1D array (length N)
      Observed alternate-allele counts per position (k in Binomial).
    - "d": 1D array (length N)
      Observed total depths per position (n in Binomial).
    - "alpha": 2D array of shape (N, M)
      Alpha parameters of the Beta-Binomial emissions for each position
      and state.
    - "beta": 2D array of shape (N, M)
      Beta parameters of the Beta-Binomial emissions for each position
      and state.
    - "logPi": 3D array of shape (N, M, M)
      Log transition matrices between states at each position (that is,
      the log of the transition probabilities).
    - "delta": 1D array of shape (M,)
      Initial state distribution (sum to 1). Its log is the initial log prior.
    - "N": int
      Number of positions (sequence length).
    - "M": int
      Number of HMM states (here expected to be 2).
    - "states": list-like of length M
      State labels.

    The function forms per-state log-emission probabilities using
    log_beta_binomial_pmf for each position, replaces any NaNs with 0
    (neutral contribution), then calls likelihood_compute(logphi, logprob, logPi)
    to obtain the total log-likelihood.

    Parameters
    ----------
    hmm : Mapping[str, Any]
        Dictionary-like object containing the entries listed above.

    Returns
    -------
    float
        Total log-likelihood of the observed sequence under the HMM.
    """

    x = hmm["x"]
    d = hmm["d"]
    alpha_mat = hmm["alpha"]  # shape (N, M)
    beta_mat  = hmm["beta"]

    M = hmm["M"]
    N = hmm["N"]

    # build logprob array shape (N, M)
    logprob = np.zeros((N, M), dtype=float)
    for m_idx in range(M):
        l_x = log_beta_binomial_pmf(
            k = x,
            n = d,
            alpha = alpha_mat[:, m_idx],
            beta = beta_mat[:, m_idx],
        )
        # Replace NaN with 0
        l_x = np.where(np.isnan(l_x), 0, l_x)
        logprob[:, m_idx] = l_x

    logphi = np.log(hmm["delta"])

    # compute total LL
    LL = likelihood_compute(logphi, logprob, hmm["logPi"])
    return LL


def get_allele_hmm(
    pAD: ArrayLike,
    DP: ArrayLike,
    p_s: ArrayLike,
    theta: Union[float, ArrayLike],
    gamma: float = 20
    ) -> Dict[str, Any]:
    """
    Build a 2-state allele HMM (dictionary) with Beta-Binomial emissions.

    The model comprises two states representing positive/negative allelic
    imbalance ("theta_up", "theta_down"). Emission parameters are
    Beta-Binomial with:
        alpha_up  = (0.5 + theta) * gamma
        beta_up   = (0.5 - theta) * gamma
        alpha_down = beta_up
        beta_down  = alpha_up

    Transitions are position-dependent and derived from the switch probability
    p_s as:
        Pi[i] = [[1 - p_s[i], p_s[i]],
                 [p_s[i],     1 - p_s[i]]]

    Parameters
    ----------
    pAD : ArrayLike
        Alternate-allele counts (length N).
    DP : ArrayLike
        Total read depths (length N).
    p_s : ArrayLike
        Per-position switch probabilities in [0, 1] (length N).
    theta : float or ArrayLike
        Allelic-imbalance parameter. If scalar, it is broadcast to length N.
    gamma : float, optional
        Concentration parameter controlling the sharpness of the Beta prior
        (default 20).

    Returns
    -------
    Dict[str, Any]
        HMM components as a dictionary with keys:
        - "x" : np.ndarray (N,) - pAD as float
        - "d" : np.ndarray (N,) - DP as float
        - "alpha" : np.ndarray (N, 2)
        - "beta" : np.ndarray (N, 2)
        - "logPi" : np.ndarray (N, 2, 2) - log transition matrices
        - "delta" : np.ndarray (2,) - initial state distribution (0.5, 0.5)
        - "N" : int - sequence length
        - "M" : int - number of states (=2)
        - "states" : list[str] - state names (["theta_up", "theta_down"])
    """

    states = ["theta_up", "theta_down"]
    N = len(p_s)

    Pi_3d = np.zeros((N,2,2), dtype=float)
    for i in range(N):
        psval = p_s[i]
        Pi_3d[i,0,0] = 1-psval
        Pi_3d[i,0,1] = psval
        Pi_3d[i,1,0] = psval
        Pi_3d[i,1,1] = 1-psval

    if np.isscalar(theta):
        theta = np.full(N, theta, dtype=float)

    prior = np.array([0.5, 0.5], dtype=float)

    alpha_up = (0.5 + theta) * gamma
    beta_up  = (0.5 - theta) * gamma
    alpha_down = beta_up
    beta_down  = alpha_up

    alpha_mat = np.column_stack([alpha_up, alpha_down])  # shape(N,2)
    beta_mat  = np.column_stack([beta_up,  beta_down ])  # shape(N,2)

    hmm = {
        "x": np.asarray(pAD, dtype=float),
        "logPi": np.log(Pi_3d),  # shape(N,2,2)
        "delta": prior,          # shape(2,)
        "alpha": alpha_mat,      # shape(N,2)
        "beta":  beta_mat,       # shape(N,2)
        "d":     np.asarray(DP, dtype=float),
        "N":     N,
        "M":     2,
        "states": states
    }
    return hmm


def calc_allele_lik(
    pAD: ArrayLike,
    DP: ArrayLike,
    p_s: ArrayLike,
    theta: Union[float, ArrayLike],
    gamma: float = 20
    ) -> float:
    """
    Wrapper to build the allele HMM and return its total log-likelihood.

    Parameters
    ----------
    pAD : ArrayLike
        Alternate-allele counts (length N).
    DP : ArrayLike
        Total read depths (length N).
    p_s : ArrayLike
        Per-position switch probabilities in [0, 1] (length N).
    theta : float or ArrayLike
        Allelic-imbalance parameter (scalar or length N).
    gamma : float, optional
        Concentration parameter for the Beta prior (default 20).

    Returns
    -------
    float
        Total log-likelihood under the constructed HMM, as computed by
        likelihood_allele.
    """
    hmm = get_allele_hmm(pAD, DP, p_s, theta, gamma)
    LL = likelihood_allele(hmm)
    return LL


def forward_backward_compute(
    logphi: NDArray,
    logprob: NDArray,
    logPi: NDArray,
    n: int,
    m: int
    ) -> NDArray:
    """
    Compute posterior state probabilities for a time-inhomogeneous HMM using the
    forward-backward algorithm in log space.

    This function performs:
      - Forward pass: builds logalpha with scaling accumulation (lscale).
      - Backward pass: builds logbeta with reverse recursion and scaling.
      - Combination: exp(logalpha + logbeta - LL) to obtain posteriors.

    Parameters
    ----------
    logphi : numpy.ndarray, shape (m,)
        Initial log state probabilities.
    logprob : numpy.ndarray, shape (n, m)
        Emission log-likelihoods for each time t and state j.
    logPi : numpy.ndarray, shape (n, m, m)
        Transition log-probabilities. Entry logPi[t, r, c] is
        log P(S_{t+1} = c | S_t = r).
    n : int
        Number of time steps (length of the sequence).
    m : int
        Number of HMM states.

    Returns
    -------
    numpy.ndarray
        Matrix of posterior probabilities with shape (n, m).

    Notes
    -----
    - All arithmetic is done in log space to improve numerical stability.
    - The returned matrix rows sum to approximately 1, subject to rounding.
    """

    # forward pass: logalpha shape (n x m)
    logalpha = np.zeros((n, m), dtype=float)
    lscale = 0.0

    # logphi is the running log-dist over states
    current_logphi = logphi.copy()

    for t in range(n):
        if t > 0:
            # update current_logphi by summing transitions from previous step
            logphi_new = np.zeros(m, dtype=float)
            subset_logPi = logPi[t]  # shape (m,m)
            for j in range(m):
                arr = current_logphi + subset_logPi[:, j]
                logphi_new[j] = log_sum_exp(arr)
            current_logphi = logphi_new

        # add logprob[t,:]
        current_logphi = current_logphi + logprob[t, :]
        # normalize
        logSumPhi = log_sum_exp(current_logphi)
        current_logphi = current_logphi - logSumPhi
        lscale += logSumPhi
        # store in logalpha[t, :]
        logalpha[t, :] = current_logphi + lscale

    LL = lscale

    # backward pass
    logbeta = np.zeros((n, m), dtype=float)
    logphi_ = np.log(np.full(m, 1.0/m))
    lscale_ = np.log(m)

    for t in range(n-2, -1, -1):
        logphi_new = np.zeros(m, dtype=float)
        subset_logPi = logPi[t+1]  # shape(m,m)
        for j in range(m):
            arr = logphi_ + logprob[t+1, :] + subset_logPi[j, :]
            logphi_new[j] = log_sum_exp(arr)

        logphi_ = logphi_new
        # store in logbeta[t,:]
        logbeta[t, :] = logphi_ + lscale_

        # normalize
        logSumPhi = log_sum_exp(logphi_)
        logphi_ = logphi_ - logSumPhi
        lscale_ += logSumPhi

    # combine
    expoutput = np.zeros((n, m), dtype=float)
    for i in range(n):
        for j in range(m):
            sum_ij = logalpha[i,j] + logbeta[i,j] - LL
            expoutput[i,j] = np.exp(sum_ij)

    return expoutput


def forward_back_allele(hmm: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Run a forward-backward computation on an allele HMM and return posterior
    state probabilities (marginals) for each time step.

    Steps
    -----
    1) If hmm["N"] == 1, return None.
    2) Build the log emission matrix logprob with shape (N, M) using a
       Beta-Binomial log PMF for each state.
    3) Initialize logphi = log(hmm["delta"]).
    4) Call forward_backward_compute to obtain an (N, M) matrix of posterior
       state probabilities.

    Parameters
    ----------
    hmm : dict
        HMM components with keys:
        - "N", "M": int lengths N and number of states M
        - "x", "d": arrays of allele counts and depths (length N)
        - "alpha", "beta": Beta parameters with shape (N, M)
        - "logPi": transition log-probabilities with shape (N, M, M)
        - "delta": initial state distribution with shape (M,)
        - "states": list of state names (optional)

    Returns
    -------
    numpy.ndarray or None
        Array of shape (N, M) with posterior marginals, or None if N == 1.

    Notes
    -----
    - NaN emission values are treated as zero contribution in log space.
    """
    if hmm['N'] == 1:
        return None  # or np.nan

    N = hmm['N']
    M = hmm['M']

    # Build logprob (N x M)
    logprob = np.zeros((N, M), dtype=float)

    for m_idx in range(M):
        # beta binomial probability
        l_x = log_beta_binomial_pmf(hmm['x'], hmm['d'], hmm['alpha'][:, m_idx], hmm['beta'][:, m_idx])
        l_x = np.where(np.isnan(l_x), 0.0, l_x)
        logprob[:, m_idx] = l_x

    # logphi = log(hmm['delta'])
    logphi = np.log(hmm['delta'])

    #  forward_backward_compute
    marginals = forward_backward_compute(logphi, logprob, hmm['logPi'], N, M)

    # return posterior
    return marginals





