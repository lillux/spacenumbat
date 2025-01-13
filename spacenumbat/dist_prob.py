#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:08:02 2024

@author: lillux
"""
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaln, comb, betaln
from scipy.optimize import minimize
from scipy.stats import nbinom


## For poilog stuff
# ported from:
# https://github.com/cran/poilog/blob/master/src/bipoilog_s_cint.c

def maxf(x, my, sig):
    # print(f'Maxf x:poisson{x}\nmy: {my}\nsig: {sig}')
    d = 100
    z = 0
    while d > 0.00001:
        if x - 1 - np.exp(z) - 1 / sig * (z - my) > 0:
            z += d
        else:
            z -= d
        d /= 2
    return z

def upper(x, m, my, sig):
    mf = (x - 1) * m - np.exp(m) - 0.5 / sig * ((m - my) ** 2)
    z = m + 20
    d = 10
    while d > 0.000001:
        if (x - 1) * z - np.exp(z) - 0.5 / sig * ((z - my) ** 2) - mf + np.log(1000000) > 0:
            z += d
        else:
            z -= d
        d /= 2
    return z

def lower(x, m, my, sig):
    mf = (x - 1) * m - np.exp(m) - 0.5 / sig * ((m - my) ** 2)
    z = m - 20
    d = 10
    while d > 0.000001:
        if (x - 1) * z - np.exp(z) - 0.5 / sig * ((z - my) ** 2) - mf + np.log(1000000) > 0:
            z -= d
        else:
            z += d
        d /= 2
    return z

def my_f(z, x, my, sig, fac):
    return np.exp(z * x - np.exp(z) - 0.5 / sig * ((z - my) ** 2) - fac)

def poilog(x, my, sig):
    # print(f'poilog: x: {x}\nmy: {my}\nsig: {sig}')
    # Step 1: Calculate parameters
    m = maxf(x, my, sig)
    a = lower(x, m, my, sig)
    b = upper(x, m, my, sig)
    fac = gammaln(x + 1)

    # Step 2: Perform integration using scipy's quad
    result, abserr = quad(my_f, a, b, args=(x, my, sig, fac), epsabs=0.00001, epsrel=0.00001)

    # Step 3: Calculate the final value
    val = result * (1 / np.sqrt(2 * np.pi * sig))
    # print(f'val: {val}')
    return val

def poilog1(x, my, sig):
    nrN = len(x)
    val = np.zeros(nrN)
    for i in range(nrN):
        val[i] = poilog(x[i], my[i], sig[i])
    return val

## refer to https://github.com/evanbiederstedt/poilogcpp
## https://github.com/kharchenkolab/hahmmr/blob/master/src/poilog.cpp
def dpoilog(x, mu, sig, log=False):
    # Check that all input parameters are the same length
    if not (len(x) == len(mu) == len(sig)):
        raise ValueError("dpoilog: All parameters must be same length")
    
    # Check that all x are integers
    if any((np.array(x)[np.array(x) != 0] % 1) != 0):
        raise ValueError("dpoilog: all x must be integers")
    
    # Check that all x are non-negative
    if any(np.array(x) < 0):
        raise ValueError("dpoilog: one or several values of x are negative")
    
    # Check that all mu and sig are finite
    if not np.all(np.isfinite(np.concatenate((mu, sig)))):
        raise ValueError("dpoilog: all parameters should be finite")
    
    # Check that none of the parameters are NA
    if any(np.isnan(np.concatenate((x, mu, sig)))):
        raise ValueError("dpoilog: Parameters cannot be NA")
    
    # Check that all sig are larger than 0
    if any(np.array(sig) <= 0):
        raise ValueError("dpoilog: sig is not larger than 0")
    
    p = poilog1(np.array(x).astype(int), np.array(mu).astype(float), np.array(sig).astype(float)**2)
    # Replace 0 values in p with a small number to avoid log(0)
    p[p == 0] = 1e-15
    
    if log:
        return np.log(p)
    else:
        return p

## From hahmmr
def l_lnpois(Y_obs, lambda_ref, d, mu, sig, phi=1):
    # print(f'l_lnpois: mu is: {mu}\nsig is: {sig}')
    if sig <= 0:
        raise ValueError(f"negative sigma. value: {sig}")
    
    if sig.size == 1:
        sig = np.repeat(sig, len(Y_obs))
    
    log_densities = np.log(dpoilog(Y_obs, mu + np.log(phi * d * lambda_ref), sig))
    return np.sum(log_densities)


def fit_lnpois(Y_obs, lambda_ref, d):
    Y_obs = Y_obs[lambda_ref > 0]
    lambda_ref = lambda_ref[lambda_ref > 0]

    def negative_log_likelihood(params):
        mu, sig = params
        if sig < 0:
            raise ValueError('optim is trying negative sigma')
        return -l_lnpois(Y_obs, lambda_ref, d, mu, sig)

    initial_guess = [0, 1]
    bounds = [(-np.inf, np.inf), (0.01, np.inf)]
    result = minimize(negative_log_likelihood, initial_guess, bounds=bounds, options={'disp': True})

    if not result.success:
        raise RuntimeError("Optimization failed")

    return result.x


## For dnbinom stuff
def dnbinom(x, mu, size):
    p = size / (mu + size)
    return nbinom.logpmf(x, p=p, n=size)


def log_beta_binomial_pmf(k, n, alpha, beta):
    """
    Compute the log of the beta-binomial PMF.

    Parameters:
        k (numpy.ndarray): Success counts
        n (numpy.ndarray): Total counts
        alpha (numpy.ndarray): Alpha parameters
        beta (numpy.ndarray): Beta parameters

    Returns:
        numpy.ndarray: Log PMF values
    """
    # Calculate the number of combination of n:DP taken k:pAD times
    log_coef = np.log(comb(n, k))
    # Calculate PMF using
    log_pmf = log_coef + betaln(k + alpha, n - k + beta) - betaln(alpha, beta)
    return log_pmf