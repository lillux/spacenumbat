#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:08:02 2024

@author: lillux
"""
import numpy as np
from scipy.special import betaln, gammaln
from scipy.optimize import minimize
from scipy.stats import nbinom
from numba import njit, prange
import math
import warnings
from typing import Union, Sequence, Tuple


## For poilog stuff
# Ported and modified from:
# https://github.com/cran/poilog/blob/master/src/bipoilog_s_cint.c

@njit
def maxf_numba(x: float, my: float, sig: float) -> float:
    """
    Finds the 'z' value for which (x - 1) - exp(z) - (1/sig)*(z - my) crosses zero,
    by repeatedly halving the step size 'd' and adjusting 'z' up or down.

    Parameters
    ----------
    x : float
        Observed count
    my : float
        Mean (mu) parameter of the log-scale distribution.
    sig : float
        Variance parameter on the log-scale.

    Returns
    -------
    float
        The final 'z' value where the search has converged.
    """
    d = 100.0
    z = 0.0
    while d > 0.00001:
        lhs = x - 1.0 - math.exp(z) - (1.0/sig)*(z - my)
        if lhs > 0.0:
            z += d
        else:
            z -= d
        d *= 0.5
    return z


@njit
def lower_bound_numba(x: float, m: float, my: float, sig: float) -> float:
    """
    Computes a lower bound for the integration range.
    Uses a manual half-step approach from an initial guess with step=10
    and shrinks until the condition is met.

    Parameters
    ----------
    x : float
        Observed count.
    m : float
        The 'z' value found by maxf_numba.
    my : float
        Mean (mu) parameter of the log-scale distribution.
    sig : float
        Variance parameter on the log-scale.

    Returns
    -------
    float
        A lower bound 'a' for the integration range.
    """
    mf = (x - 1.0)*m - math.exp(m) - 0.5/sig*((m - my)**2)
    z = m - 20.0
    d = 10.0
    while d > 0.000001:
        lhs = (x - 1.0)*z - math.exp(z) - 0.5/sig*((z - my)**2) - mf + math.log(1000000.0)
        if lhs > 0.0:
            z -= d
        else:
            z += d
        d *= 0.5
    return z


@njit
def upper_bound_numba(x: float, m: float, my: float, sig: float) -> float:
    """
    Computes an upper bound 'b' for the integration range.
    Uses a half-step approach, starting from a guess with step=10.

    Parameters
    ----------
    x : float
        Observed count.
    m : float
        The 'z' value found by maxf_numba.
    my : float
        Mean (mu) parameter of the log-scale distribution.
    sig : float
        Variance parameter on the log-scale.

    Returns
    -------
    float
        An upper bound 'b' for the integration range.
    """
    mf = (x - 1.0)*m - math.exp(m) - 0.5/sig*((m - my)**2)
    z = m + 20.0
    d = 10.0
    while d > 0.000001:
        lhs = (x - 1.0)*z - math.exp(z) - 0.5/sig*((z - my)**2) - mf + math.log(1000000.0)
        if lhs > 0.0:
            z += d
        else:
            z -= d
        d *= 0.5
    return z


@njit
def integrand_numba(z: float, x: float, my: float, sig: float, fac: float) -> float:
    """
    The integrand for Poisson lognormal calculations, 
    used when performing trapezoidal (or other) numerical integration.

    Parameters
    ----------
    z : float
        The current evaluation point in the integral range.
    x : float
        Observed count.
    my : float
        Mean (mu) parameter on log-scale.
    sig : float
        Variance parameter on log-scale.
    fac : float
        A precomputed log(gamma(x+1.0)) factor to avoid repeated cost.

    Returns
    -------
    float
        The integrand value at 'z'.
    """
    return math.exp(z*x - math.exp(z) - 0.5/sig*((z - my)**2) - fac)


@njit
def poilog_numba(x: float, my: float, sig: float, n_points: int = 128) -> float:
    """
    Poisson log-normal density approximation with Numba-accelerated 
    half-step bounding and trapezoid integration.

    Steps:
      1) Finds 'm' (peak location).
      2) Finds integration bounds [a, b].
      3) Creates a uniform mesh of size 'n_points' in [a, b].
      4) Evaluates the integrand on each mesh point using trapezoid rule.
      5) Multiplies by 1 / sqrt(2*pi*sig).

    Parameters
    ----------
    x : float
        Observed count.
    my : float
        Mean (mu) parameter on log-scale.
    sig : float
        Variance parameter on log-scale.
    n_points : int, optional
        Number of subintervals for trapezoid integration. 
        Larger values are more accurate but slower (default=128).

    Returns
    -------
    float
        The approximate Poisson log-normal probability density at (x,my,sig).
    """
    m_ = maxf_numba(x, my, sig)
    a_ = lower_bound_numba(x, m_, my, sig)
    b_ = upper_bound_numba(x, m_, my, sig)
    fac = math.lgamma(x + 1.0)

    zvals = np.linspace(a_, b_, n_points)
    fvals = np.empty(n_points, dtype=np.float64)
    for i in range(n_points):
        fvals[i] = integrand_numba(zvals[i], x, my, sig, fac)

    integral = 0.0
    for i in range(n_points - 1):
        trape = 0.5*(fvals[i] + fvals[i+1])*(zvals[i+1] - zvals[i])
        integral += trape

    val = integral * (1.0 / math.sqrt(2.0 * math.pi * sig))
    return val


@njit(parallel=True)
def poilog1(xs: Union[np.ndarray, Sequence[float]], 
            mys: Union[np.ndarray, Sequence[float]], 
            sigs: Union[np.ndarray, Sequence[float]], 
            n_points: int = 256
           ) -> np.ndarray:
    """
    Parallel approach in Numba for computing the Poisson log-normal density
    for multiple inputs (x, my, sig).

    Each iteration uses poilog_numba() inside a parallel for-loop 
    to compute the density. The half-step bounding and trapezoid integration 
    logic is fully compiled.

    Parameters
    ----------
    xs : array-like of float
        Observed counts for each item. 
    mys : array-like of float
        log-scale means for each item.
    sigs : array-like of float
        log-scale variances for each item.
    n_points : int, optional
        Number of subintervals for trapezoid integration in each call 
        (default=256).

    Returns
    -------
    np.ndarray
        1D array of shape (len(xs),) with the approximate Poisson log-normal density 
        for each input triple (x[i], my[i], sig[i]).
    """
    nrN = len(xs)
    val = np.empty(nrN, dtype=np.float64)
    for i in prange(nrN):
       #val[i] = poilog_numba(xs[i], mys[i], sigs[i]**2, n_points) #TODO: evaluate differences
        val[i] = poilog_numba(xs[i], mys[i], sigs[i], n_points)


    return val

    
def dpoilog(
    x: Union[np.ndarray, list, tuple],
    mu: Union[np.ndarray, list, tuple],
    sig: Union[np.ndarray, list, tuple],
    log: bool = False,
    n_points: int = 256
    ) -> np.ndarray:
    """
    Compute the (log) probability mass function of the Poisson-lognormal distribution.

    This function wraps the `poilog1` numba implementation,
    modified from:
        (https://github.com/evanbiederstedt/poilogcpp and
         https://github.com/kharchenkolab/hahmmr/blob/master/src/poilog.cpp).

    Parameters
    ----------
    x : array-like of int
        Observed counts; non-negative integers. Must have the same length as `mu` and `sig`.
    mu : array-like of float
        Log-scale mean parameters for the Poisson-lognormal distribution.
    sig : array-like of float
        Log-scale standard deviation parameters; must be strictly positive.
    log : bool, optional
        If True, return the log of the PMF; otherwise, return the PMF. Default is False.
    n_points : int, optional
        Number of points used internally in the approximation. Default is 256.

    Returns
    -------
    np.ndarray
        An array of PMF values (or log-PMF if `log=True`) of the same length as inputs.

    Raises
    ------
    ValueError
        If input lengths do not match, or if inputs violate domain constraints:
        - `x` must be non-negative integers
        - `mu` and `sig` must be finite numbers
        - `sig` must be strictly positive
        - None of the inputs can contain NaNs

    Notes
    -----
    - The parameter `sig` is squared internally before calling the underlying C++ routine.
    - Zero probabilities are replaced with a small positive value (1e-15) to avoid log(0).
    """
    if not (len(x) == len(mu) == len(sig)):
        raise ValueError("All parameters must be same length")
    
    x_arr = np.array(x)
    mu_arr = np.array(mu)
    sig_arr = np.array(sig)
    
    # Check that all x are integers
    if np.any(x_arr[x_arr != 0] % 1 != 0):
        raise ValueError("all x must be integers")
    
    # Check that all x are non-negative
    if np.any(x_arr < 0):
        raise ValueError("one or several values of x are negative")
    
    # Check that all mu and sig are finite
    if not np.all(np.isfinite(np.concatenate((mu_arr, sig_arr)))):
        raise ValueError("all parameters should be finite")
    
    # Check that none of the parameters are NaN
    if np.any(np.isnan(np.concatenate((x_arr, mu_arr, sig_arr)))):
        raise ValueError("Parameters cannot be NA")
    
    # Check that all sig are larger than 0
    if np.any(sig_arr <= 0):
        raise ValueError("sig is not larger than 0")
        
    p = poilog1(
        xs=x_arr.astype(int),
        mys=mu_arr.astype(float),
        sigs=np.square(sig_arr.astype(float)),
        n_points=n_points
    )
    
    # Replace 0 values in p with a small number to avoid log(0)
    p[p == 0] = 1e-15
    
    if log:
        return np.log(p)
    else:
        return p


def l_lnpois(
    Y_obs: np.ndarray,
    lambda_ref: np.ndarray,
    d: Union[float, np.ndarray],
    mu: float,
    sig: Union[float, np.ndarray],
    phi: float = 1,
    n_points: int = 256
    ) -> float:
    """
    Compute the log-likelihood of observed counts `Y_obs` under a lognormal-Poisson model.

    Parameters
    ----------
    Y_obs : np.ndarray
        Observed counts, array of non-negative integers.
    lambda_ref : np.ndarray
        Reference expression profile, positive values.
    d : float or np.ndarray
        Scaling factor(s) for expression.
    mu : float
        Mean parameter of the lognormal distribution on log-scale.
    sig : float or np.ndarray
        Standard deviation parameter of the lognormal distribution on log-scale.
        Must be positive.
    phi : float, optional
        Additional scaling parameter (default is 1).
    n_points : int, optional
        Number of points used internally in the approximation (default 256).

    Returns
    -------
    float
        The sum of log densities of the observed data under the model.

    Raises
    ------
    ValueError
        If any `sig` value is not positive.
    """
    if np.any(sig <= 0):
        raise ValueError(f"All sigma values must be positive. Received: {sig}")

    # If sig is a scalar, expand it to match Y_obs length
    if sig.size == 1:
        sig = np.repeat(sig, len(Y_obs))

    # Compute log densities
    log_densities = np.log(dpoilog(
        Y_obs,
        mu + np.log(phi * d * lambda_ref),
        sig,
        n_points=n_points
    ))
    return float(np.sum(log_densities))


def fit_lnpois(
    Y_obs: np.ndarray,
    lambda_ref: np.ndarray,
    d: Union[float, np.ndarray],
    disp: bool = False,
    n_points: int = 256
    ) -> Tuple[float, float]:
    """
    Fit the lognormal-Poisson model parameters (mu, sigma) by maximizing likelihood.

    Parameters
    ----------
    Y_obs : np.ndarray
        Observed counts; zero or positive integers.
    lambda_ref : np.ndarray
        Reference expression profile; positive values only.
    d : float or np.ndarray
        Scaling factor(s).
    disp : bool, optional
        Whether to display optimizer output (default False).
    n_points : int, optional
        Number of points for internal approximation (default 256).

    Returns
    -------
    Tuple[float, float]
        The optimized parameters (mu, sigma).

    Warnings
    --------
    Emits a warning if optimization fails, returning last attempted parameter values.

    Notes
    -----
    Filters out observations where `lambda_ref <= 0` before fitting.
    Sigma is constrained to be at least 0.01 during optimization.
    """
    # Filter invalid lambda_ref values
    valid_mask = lambda_ref > 0
    Y_obs = Y_obs[valid_mask]
    lambda_ref = lambda_ref[valid_mask]

    def negative_log_likelihood(params: Tuple[float, float]) -> float:
        mu, sig = params
        if sig < 0:
            raise ValueError('optim is trying negative sigma')
            # return np.inf
        return -l_lnpois(Y_obs, lambda_ref, d, mu, sig, n_points=n_points)

    initial_guess = [0.0, 1.0]
    bounds = [(-np.inf, np.inf), (0.01, np.inf)]

    result = minimize(
        negative_log_likelihood,
        initial_guess,
        method='L-BFGS-B',
        bounds=bounds,
        #options={'disp': disp},
        tol=1e-5
    )

    if not result.success:
        warnings.warn(
            "Optimization failed! Last step values are returned. This may be a problem!!!!",
            RuntimeWarning
        )
    return result.x


## For dnbinom stuff
def dnbinom(
    x: Union[int, np.ndarray],
    mu: Union[float, np.ndarray],
    size: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
    """
    Compute the log-probability mass function of the Negative Binomial distribution
    parameterized by mean (`mu`) and dispersion (`size`), evaluated at `x`.

    Parameters
    ----------
    x : int or np.ndarray
        Number of observed failures (non-negative integers).
    mu : float or np.ndarray
        Mean of the distribution (must be positive).
    size : float or np.ndarray
        Dispersion parameter (shape), sometimes called 'size' or 'r' (must be positive).

    Returns
    -------
    float or np.ndarray
        The log probability mass function evaluated at `x`. Returns a scalar if inputs are scalars,
        or an array if inputs are arrays (broadcasting).

    Notes
    -----
    The Negative Binomial is parameterized so that:
        p = size / (mu + size),
        n = size

    """
    p = size / (mu + size)
    return nbinom.logpmf(x, n=size, p=p)


def log_beta_binomial_pmf(k: np.ndarray, 
                          n: np.ndarray, 
                          alpha: np.ndarray, 
                          beta: np.ndarray) -> np.ndarray:
    """
    Compute the log of the beta-binomial probability mass function (PMF).

    The beta-binomial PMF is given by:
    
        PMF(k; n, α, β) = (n choose k) * (B(k + α, n - k + β) / B(α, β))
    
    where B is the beta function.

    Parameters
    ----------
    k : np.ndarray
        Success counts.
    n : np.ndarray
        Total counts.
    alpha : np.ndarray
        Alpha parameters.
    beta : np.ndarray
        Beta parameters.
    
    Returns
    -------
    np.ndarray
        An array of log PMF values corresponding to the inputs.
    """
    # Compute log of the combination coefficient (n choose k).
    #log_coef = np.log(comb(n, k)) # log(0) warnings
    valid = (k >= 0) & (k <= n)
    log_coef = np.full_like(k, -np.inf, dtype=float)
    log_coef[valid] = (gammaln(n[valid] + 1)
                       - gammaln(k[valid] + 1)
                       - gammaln(n[valid] - k[valid] + 1))

    # Calculate PMF using beta functions.
    log_pmf = log_coef + betaln(k + alpha, n - k + beta) - betaln(alpha, beta)
    return log_pmf





