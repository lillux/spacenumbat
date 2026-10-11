#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Donor-aware conditional information-bottleneck RNA reference builder.

This module replaces the greedy pairwise JSD subtype-merging stage in
``build_rna_ref.py`` with a formal conditional information bottleneck (CIB).
It uses deterministic Blahut--Arimoto updates:
the input label, compressed label, context and gene are finite categorical
variables, so the required expectations can be evaluated exactly.

The optimized objective within each biological family is

    I(L; Z) - beta * I(Z; G | C),

where L is the original atlas label, Z is the compressed label, G is gene
identity and C is tissue/anatomical context. Equivalently, up to a constant,
the minimized rate--distortion objective is

    I(L; Z) + beta * E[D_KL(p(G|L,C) || p(G|Z,C))].

Donor pseudobulks are normalized independently before model fitting. Thus a
donor contributes one probability profile rather than a weight proportional
to its cell or read count. A beta grid can be selected by held-out-donor
predictive cross-entropy. The most compressed solution whose validation loss
is within ``max_excess_nll`` nats/count of the uncompressed reference is used.

The public ``build_rna_reference_ib`` function mirrors ``build_rna_reference``
and returns a SpaceNumbat-compatible gene-by-reference matrix, manifest, gene
annotation and optional diagnostics.
"""

#from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp


log = logging.getLogger(__name__)
__version__ = "1.0.0"


@dataclass
class ConditionalIBResult:
    """Fitted conditional information-bottleneck model."""

    labels: list[str]
    contexts: list[str]
    q_cluster_given_label: np.ndarray
    decoder: np.ndarray
    decoder_pan_context: np.ndarray
    label_prior: np.ndarray
    context_given_label: np.ndarray
    objective: float
    rate: float
    distortion: float
    source_information: float
    retained_information: float
    retained_fraction: float
    beta: float
    converged: bool
    n_iter: int

    @property
    def hard_assignment(self) -> np.ndarray:
        """Contiguous hard-cluster index for every label."""
        raw = np.argmax(self.q_cluster_given_label, axis=1)
        used = sorted(np.unique(raw))
        remap = {old: new for new, old in enumerate(used)}
        return np.asarray([remap[x] for x in raw], dtype=np.int64)

    @property
    def n_clusters(self) -> int:
        return int(np.unique(self.hard_assignment).size)

    @property
    def effective_clusters(self) -> float:
        qz = self.label_prior @ self.q_cluster_given_label
        qz = qz[qz > 0]
        return float(np.exp(-np.sum(qz * np.log(qz))))


@dataclass
class IBSelection:
    """Selected model and its rate--distortion/model-selection curve."""

    model: ConditionalIBResult
    curve: pd.DataFrame


def _check_counts(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 2 or counts.shape[0] == 0 or counts.shape[1] < 2:
        raise ValueError("counts must be a non-empty pseudobulk-by-gene matrix.")
    if not np.isfinite(counts).all() or np.any(counts < 0):
        raise ValueError("counts must be finite and non-negative.")
    if np.any(counts.sum(axis=1) <= 0):
        raise ValueError("Every pseudobulk must contain at least one count.")
    return counts


def _normalize_counts(counts: np.ndarray, prior_mass: float) -> np.ndarray:
    """Symmetric-Dirichlet posterior mean for rows of a count matrix."""
    counts = _check_counts(counts)
    if not np.isfinite(prior_mass) or prior_mass <= 0:
        raise ValueError("prior_mass must be finite and positive.")
    depth = counts.sum(axis=1, keepdims=True)
    return (counts + prior_mass / counts.shape[1]) / (depth + prior_mass)


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def _build_problem(
    counts: np.ndarray,
    obs: pd.DataFrame,
    *,
    label_col: str,
    context_col: str,
    donor_col: str,
    prior_mass: float,
    labels: Sequence[str] | None = None,
    contexts: Sequence[str] | None = None,
) -> dict:
    """Construct donor-balanced p(gene | label, context)."""
    counts = _check_counts(counts)
    if len(obs) != counts.shape[0]:
        raise ValueError("obs and counts have different numbers of rows.")
    missing = {label_col, context_col, donor_col} - set(obs.columns)
    if missing:
        raise ValueError(f"obs is missing columns: {sorted(missing)}")

    frame = obs[[label_col, context_col, donor_col]].astype("string").reset_index(drop=True)
    if frame.isna().any().any():
        raise ValueError("label, context and donor values must be non-missing.")
    labels = sorted(frame[label_col].unique()) if labels is None else list(labels)
    contexts = sorted(frame[context_col].unique()) if contexts is None else list(contexts)
    if len(labels) == 0 or len(contexts) == 0:
        raise ValueError("At least one label and one context are required.")

    label_index = {value: i for i, value in enumerate(labels)}
    context_index = {value: i for i, value in enumerate(contexts)}
    unknown_labels = set(frame[label_col]) - set(label_index)
    unknown_contexts = set(frame[context_col]) - set(context_index)
    if unknown_labels or unknown_contexts:
        raise ValueError("labels/contexts do not cover all rows.")

    # Pool accidental duplicate donor-label-context rows before normalization.
    keys = pd.MultiIndex.from_frame(frame)
    codes, groups = pd.factorize(keys, sort=True)
    grouping = sparse.csr_matrix(
        (np.ones(len(codes)), (codes, np.arange(len(codes)))),
        shape=(len(groups), len(codes)),
    )
    donor_counts = np.asarray(grouping @ counts)
    donor_obs = groups.to_frame(index=False)
    donor_obs.columns = [label_col, context_col, donor_col]
    donor_profiles = _normalize_counts(donor_counts, prior_mass)

    n_label = len(labels)
    n_context = len(contexts)
    n_gene = counts.shape[1]
    profile = np.zeros((n_label, n_context, n_gene), dtype=np.float64)
    support = np.zeros((n_label, n_context), dtype=np.float64)

    for (label, context), idx in donor_obs.groupby(
        [label_col, context_col], sort=False
    ).indices.items():
        li = label_index[str(label)]
        ci = context_index[str(context)]
        idx = np.asarray(idx)
        profile[li, ci] = donor_profiles[idx].mean(axis=0)
        support[li, ci] = donor_obs.iloc[idx][donor_col].nunique()

    if np.any(support.sum(axis=1) == 0):
        absent = [labels[i] for i in np.flatnonzero(support.sum(axis=1) == 0)]
        raise ValueError(f"No training pseudobulk for labels: {absent}")

    # Equal label prior prevents common atlas annotations from dominating the
    # rate term. Context weights within a label reflect independent donors.
    label_prior = np.full(n_label, 1.0 / n_label)
    context_given_label = support / support.sum(axis=1, keepdims=True)
    joint = label_prior[:, None] * context_given_label

    context_profile = np.zeros((n_context, n_gene), dtype=np.float64)
    for ci in range(n_context):
        mass = joint[:, ci].sum()
        if mass > 0:
            context_profile[ci] = np.sum(
                joint[:, ci, None] * profile[:, ci, :], axis=0
            ) / mass
        else:
            context_profile[ci] = donor_profiles.mean(axis=0)
        context_profile[ci] /= context_profile[ci].sum()

    pan_label_profile = np.sum(
        context_given_label[:, :, None] * profile, axis=1
    )
    pan_label_profile /= pan_label_profile.sum(axis=1, keepdims=True)

    return {
        "labels": labels,
        "contexts": contexts,
        "profile": profile,
        "support": support,
        "label_prior": label_prior,
        "context_given_label": context_given_label,
        "joint": joint,
        "context_profile": context_profile,
        "pan_label_profile": pan_label_profile,
    }


def _decoder(problem: dict, q: np.ndarray, decoder_prior: float) -> tuple[np.ndarray, np.ndarray]:
    """Compute p(gene | cluster, context) and pan-context decoders."""
    profile = problem["profile"]
    joint = problem["joint"]
    context_profile = problem["context_profile"]
    pan_label = problem["pan_label_profile"]
    label_prior = problem["label_prior"]
    n_cluster = q.shape[1]
    n_context = profile.shape[1]
    n_gene = profile.shape[2]
    result = np.empty((n_cluster, n_context, n_gene), dtype=np.float64)

    for ci in range(n_context):
        weights = joint[:, ci, None] * q
        mass = weights.sum(axis=0)
        numerator = weights.T @ profile[:, ci, :]
        numerator += decoder_prior * context_profile[ci][None, :]
        result[:, ci, :] = numerator / (mass[:, None] + decoder_prior)
        result[:, ci, :] /= result[:, ci, :].sum(axis=1, keepdims=True)

    pan_weights = label_prior[:, None] * q
    pan_mass = pan_weights.sum(axis=0)
    global_profile = label_prior @ pan_label
    pan = pan_weights.T @ pan_label
    pan += decoder_prior * global_profile[None, :]
    pan /= pan_mass[:, None] + decoder_prior
    pan /= pan.sum(axis=1, keepdims=True)
    return result, pan


def _distortion_matrix(problem: dict, decoder: np.ndarray, eps: float) -> np.ndarray:
    """Conditional KL distortion d(label, cluster)."""
    profile = problem["profile"]
    weights = problem["context_given_label"]
    n_label, n_context, _ = profile.shape
    n_cluster = decoder.shape[0]
    distortion = np.zeros((n_label, n_cluster), dtype=np.float64)
    for li in range(n_label):
        for ci in np.flatnonzero(weights[li] > 0):
            p = np.maximum(profile[li, ci], eps)
            p /= p.sum()
            log_p = np.log(p)
            log_q = np.log(np.maximum(decoder[:, ci, :], eps))
            kl = np.sum(p[None, :] * (log_p[None, :] - log_q), axis=1)
            distortion[li] += weights[li, ci] * np.maximum(kl, 0.0)
    return distortion


def _collapse_equivalent_clusters(
    problem: dict,
    q: np.ndarray,
    decoder: np.ndarray,
    pan: np.ndarray,
    *,
    threshold: float,
    decoder_prior: float,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove duplicated IB codewords caused by categorical symmetry."""
    n_cluster = q.shape[1]
    parent = np.arange(n_cluster)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[b] = a

    def js_rows(a, b):
        a = np.maximum(a, eps)
        b = np.maximum(b, eps)
        a /= a.sum(axis=-1, keepdims=True)
        b /= b.sum(axis=-1, keepdims=True)
        center = 0.5 * (a + b)
        return 0.5 * np.sum(a * (np.log(a) - np.log(center)), axis=-1) + 0.5 * np.sum(
            b * (np.log(b) - np.log(center)), axis=-1
        )

    for i in range(n_cluster):
        for j in range(i):
            context_js = np.max(js_rows(decoder[i], decoder[j]))
            pan_js = float(js_rows(pan[i][None, :], pan[j][None, :])[0])
            if max(float(context_js), pan_js) <= threshold:
                union(i, j)

    groups = {}
    for cluster in range(n_cluster):
        groups.setdefault(find(cluster), []).append(cluster)
    if len(groups) == n_cluster:
        return q, decoder, pan
    collapsed = np.column_stack([q[:, members].sum(axis=1) for members in groups.values()])
    collapsed /= collapsed.sum(axis=1, keepdims=True)
    decoder, pan = _decoder(problem, collapsed, decoder_prior)
    return collapsed, decoder, pan


def _source_information(problem: dict, eps: float) -> float:
    """I(label; gene | context) under the empirical donor-balanced measure."""
    profile = problem["profile"]
    joint = problem["joint"]
    context_profile = problem["context_profile"]
    value = 0.0
    for li in range(profile.shape[0]):
        for ci in np.flatnonzero(joint[li] > 0):
            p = np.maximum(profile[li, ci], eps)
            p /= p.sum()
            q = np.maximum(context_profile[ci], eps)
            q /= q.sum()
            value += joint[li, ci] * np.sum(p * (np.log(p) - np.log(q)))
    return float(max(value, 0.0))


def fit_conditional_information_bottleneck(
    counts: np.ndarray,
    obs: pd.DataFrame,
    *,
    beta: float,
    label_col: str = "label",
    context_col: str = "context",
    donor_col: str = "donor",
    prior_mass: float = 1.0,
    n_clusters: int | None = None,
    n_restarts: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-9,
    decoder_prior: float = 1e-8,
    cluster_merge_tol: float = 1e-10,
    eps: float = 1e-12,
    random_state: int = 0,
    labels: Sequence[str] | None = None,
    contexts: Sequence[str] | None = None,
) -> ConditionalIBResult:
    """Fit a formal conditional information bottleneck.

    Higher ``beta`` preserves more label-specific transcriptomic information;
    lower values favor stronger compression. ``n_clusters`` is an upper bound,
    not the requested final number: unused or redundant clusters disappear
    after hardening the posterior assignments.
    """
    for name, value in {
        "beta": beta,
        "prior_mass": prior_mass,
        "tol": tol,
        "decoder_prior": decoder_prior,
        "cluster_merge_tol": cluster_merge_tol,
        "eps": eps,
    }.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if n_restarts < 1 or max_iter < 1:
        raise ValueError("n_restarts and max_iter must be positive.")

    problem = _build_problem(
        counts,
        obs,
        label_col=label_col,
        context_col=context_col,
        donor_col=donor_col,
        prior_mass=prior_mass,
        labels=labels,
        contexts=contexts,
    )
    n_label = len(problem["labels"])
    n_clusters = n_label if n_clusters is None else min(int(n_clusters), n_label)
    if n_clusters < 1:
        raise ValueError("n_clusters must be positive.")
    label_prior = problem["label_prior"]
    source_information = _source_information(problem, eps)
    rng = np.random.default_rng(random_state)
    best = None

    for restart in range(n_restarts):
        if restart == 0 and n_clusters == n_label:
            q = np.full((n_label, n_clusters), 1e-3 / max(n_clusters - 1, 1))
            np.fill_diagonal(q, 1.0 - 1e-3)
            q /= q.sum(axis=1, keepdims=True)
        else:
            q = rng.dirichlet(np.ones(n_clusters), size=n_label)

        previous = np.inf
        converged = False
        for iteration in range(1, max_iter + 1):
            decoder, pan = _decoder(problem, q, decoder_prior)
            distortion_matrix = _distortion_matrix(problem, decoder, eps)
            qz = np.maximum(label_prior @ q, eps)
            logits = np.log(qz)[None, :] - beta * distortion_matrix
            q_new = _softmax_rows(logits)

            qz_new = np.maximum(label_prior @ q_new, eps)
            rate = float(
                np.sum(
                    label_prior[:, None]
                    * q_new
                    * (np.log(np.maximum(q_new, eps)) - np.log(qz_new)[None, :])
                )
            )
            distortion = float(
                np.sum(label_prior[:, None] * q_new * distortion_matrix)
            )
            objective = rate + beta * distortion
            change = max(np.max(np.abs(q_new - q)), abs(previous - objective))
            q = q_new
            if change < tol:
                converged = True
                break
            previous = objective

        decoder, pan = _decoder(problem, q, decoder_prior)
        q, decoder, pan = _collapse_equivalent_clusters(
            problem,
            q,
            decoder,
            pan,
            threshold=cluster_merge_tol,
            decoder_prior=decoder_prior,
            eps=eps,
        )
        distortion_matrix = _distortion_matrix(problem, decoder, eps)
        qz = np.maximum(label_prior @ q, eps)
        rate = float(
            np.sum(
                label_prior[:, None]
                * q
                * (np.log(np.maximum(q, eps)) - np.log(qz)[None, :])
            )
        )
        distortion = float(np.sum(label_prior[:, None] * q * distortion_matrix))
        objective = rate + beta * distortion
        retained = max(0.0, source_information - distortion)
        fraction = retained / source_information if source_information > eps else 1.0
        candidate = ConditionalIBResult(
            labels=list(problem["labels"]),
            contexts=list(problem["contexts"]),
            q_cluster_given_label=q,
            decoder=decoder,
            decoder_pan_context=pan,
            label_prior=label_prior,
            context_given_label=problem["context_given_label"],
            objective=objective,
            rate=rate,
            distortion=distortion,
            source_information=source_information,
            retained_information=retained,
            retained_fraction=float(np.clip(fraction, 0.0, 1.0)),
            beta=float(beta),
            converged=converged,
            n_iter=iteration,
        )
        if best is None or candidate.objective < best.objective:
            best = candidate

    return best


def _predictive_nll(
    model: ConditionalIBResult,
    train_counts: np.ndarray,
    train_obs: pd.DataFrame,
    validation_counts: np.ndarray,
    validation_obs: pd.DataFrame,
    *,
    label_col: str,
    context_col: str,
    donor_col: str,
    prior_mass: float,
    eps: float,
) -> tuple[float, float, int]:
    """Candidate and uncompressed held-out cross-entropy in nats/count."""
    problem = _build_problem(
        train_counts,
        train_obs,
        label_col=label_col,
        context_col=context_col,
        donor_col=donor_col,
        prior_mass=prior_mass,
        labels=model.labels,
        contexts=model.contexts,
    )
    label_index = {x: i for i, x in enumerate(model.labels)}
    context_index = {x: i for i, x in enumerate(model.contexts)}
    candidate_loss = 0.0
    baseline_loss = 0.0
    total = 0

    validation_labels = validation_obs[label_col].astype(str).to_numpy()
    validation_contexts = validation_obs[context_col].astype(str).to_numpy()
    for row, label, context in zip(
        np.asarray(validation_counts, float), validation_labels, validation_contexts
    ):
        if label not in label_index:
            continue
        li = label_index[label]
        if (
            context in context_index
            and problem["support"][:, context_index[context]].sum() > 0
        ):
            ci = context_index[context]
            prediction = model.q_cluster_given_label[li] @ model.decoder[:, ci, :]
            baseline = (
                problem["profile"][li, ci]
                if problem["support"][li, ci] > 0
                else problem["pan_label_profile"][li]
            )
        else:
            prediction = model.q_cluster_given_label[li] @ model.decoder_pan_context
            baseline = problem["pan_label_profile"][li]
        depth = int(np.sum(row))
        if depth <= 0:
            continue
        candidate_loss -= float(row @ np.log(np.maximum(prediction, eps)))
        baseline_loss -= float(row @ np.log(np.maximum(baseline, eps)))
        total += depth
    return candidate_loss, baseline_loss, total


def select_information_bottleneck(
    counts: np.ndarray,
    obs: pd.DataFrame,
    *,
    betas: Sequence[float] = (0.5, 1, 2, 4, 8, 16, 32, 64),
    label_col: str = "label",
    context_col: str = "context",
    donor_col: str = "donor",
    prior_mass: float = 1.0,
    n_folds: int = 3,
    max_excess_nll: float = 1e-3,
    min_retained_fraction: float = 0.98,
    n_restarts: int = 3,
    max_iter: int = 1000,
    tol: float = 1e-9,
    decoder_prior: float = 1e-8,
    eps: float = 1e-12,
    random_state: int = 0,
) -> IBSelection:
    """Select compression using donor-blocked validation.

    The selected model is the solution with the fewest hard clusters whose
    held-out loss is within ``max_excess_nll`` nats/count of the uncompressed
    label-by-context reference and whose in-sample retained information is at
    least ``min_retained_fraction``. If no candidate satisfies both safeguards,
    the candidate with the smallest held-out excess loss is selected.
    """
    counts = _check_counts(counts)
    obs = obs.reset_index(drop=True).copy()
    if len(obs) != counts.shape[0]:
        raise ValueError("obs and counts have different numbers of rows.")
    betas = sorted({float(x) for x in betas})
    if not betas or any(not np.isfinite(x) or x <= 0 for x in betas):
        raise ValueError("betas must contain finite positive values.")
    if max_excess_nll < 0 or not 0 < min_retained_fraction <= 1:
        raise ValueError("Invalid model-selection tolerances.")

    if n_folds < 1:
        raise ValueError("n_folds must be positive.")
    labels = sorted(obs[label_col].astype(str).unique())
    contexts = sorted(obs[context_col].astype(str).unique())
    donors = np.asarray(sorted(obs[donor_col].astype(str).unique()))
    rng = np.random.default_rng(random_state)
    rng.shuffle(donors)
    folds = [x for x in np.array_split(donors, min(n_folds, len(donors))) if len(x)]
    rows = []

    for beta_index, beta in enumerate(betas):
        candidate_loss = 0.0
        baseline_loss = 0.0
        total = 0
        valid_folds = 0
        for fold_index, held_out in enumerate(folds):
            validation = obs[donor_col].astype(str).isin(held_out).to_numpy()
            training = ~validation
            train_labels = set(obs.loc[training, label_col].astype(str))
            if not set(labels).issubset(train_labels) or not validation.any():
                continue
            try:
                model = fit_conditional_information_bottleneck(
                    counts[training],
                    obs.loc[training].reset_index(drop=True),
                    beta=beta,
                    label_col=label_col,
                    context_col=context_col,
                    donor_col=donor_col,
                    prior_mass=prior_mass,
                    n_restarts=n_restarts,
                    max_iter=max_iter,
                    tol=tol,
                    decoder_prior=decoder_prior,
                    eps=eps,
                    random_state=random_state + 1009 * beta_index + fold_index,
                    labels=labels,
                    contexts=contexts,
                )
                cand, base, depth = _predictive_nll(
                    model,
                    counts[training],
                    obs.loc[training].reset_index(drop=True),
                    counts[validation],
                    obs.loc[validation].reset_index(drop=True),
                    label_col=label_col,
                    context_col=context_col,
                    donor_col=donor_col,
                    prior_mass=prior_mass,
                    eps=eps,
                )
            except ValueError:
                continue
            candidate_loss += cand
            baseline_loss += base
            total += depth
            valid_folds += int(depth > 0)

        full_model = fit_conditional_information_bottleneck(
            counts,
            obs,
            beta=beta,
            label_col=label_col,
            context_col=context_col,
            donor_col=donor_col,
            prior_mass=prior_mass,
            n_restarts=n_restarts,
            max_iter=max_iter,
            tol=tol,
            decoder_prior=decoder_prior,
            eps=eps,
            random_state=random_state + 7919 * beta_index,
            labels=labels,
            contexts=contexts,
        )
        if total:
            validation_nll = candidate_loss / total
            baseline_nll = baseline_loss / total
            excess = validation_nll - baseline_nll
        else:
            validation_nll = baseline_nll = excess = np.nan
        rows.append(
            {
                "beta": beta,
                "n_clusters": full_model.n_clusters,
                "effective_clusters": full_model.effective_clusters,
                "rate": full_model.rate,
                "distortion": full_model.distortion,
                "source_information": full_model.source_information,
                "retained_information": full_model.retained_information,
                "retained_fraction": full_model.retained_fraction,
                "validation_nll": validation_nll,
                "uncompressed_validation_nll": baseline_nll,
                "excess_validation_nll": excess,
                "valid_folds": valid_folds,
                "converged": full_model.converged,
            }
        )

    curve = pd.DataFrame(rows)
    has_cv = curve.excess_validation_nll.notna()
    eligible = (
        has_cv
        & (curve.excess_validation_nll <= max_excess_nll)
        & (curve.retained_fraction >= min_retained_fraction)
    )
    if eligible.any():
        selected_index = (
            curve.loc[eligible]
            .sort_values(
                ["n_clusters", "rate", "excess_validation_nll", "beta"],
                ascending=[True, True, True, True],
            )
            .index[0]
        )
    elif has_cv.any():
        selected_index = (
            curve.loc[has_cv]
            .sort_values(
                ["excess_validation_nll", "n_clusters", "retained_fraction"],
                ascending=[True, True, False],
            )
            .index[0]
        )
    else:
        eligible = curve.retained_fraction >= min_retained_fraction
        pool = curve.loc[eligible] if eligible.any() else curve
        selected_index = (
            pool.sort_values(
                ["n_clusters", "retained_fraction", "rate"],
                ascending=[True, False, True],
            ).index[0]
        )
    selected_beta = float(curve.loc[selected_index, "beta"])
    selected_beta_index = betas.index(selected_beta)
    selected_model = fit_conditional_information_bottleneck(
        counts,
        obs,
        beta=selected_beta,
        label_col=label_col,
        context_col=context_col,
        donor_col=donor_col,
        prior_mass=prior_mass,
        n_restarts=n_restarts,
        max_iter=max_iter,
        tol=tol,
        decoder_prior=decoder_prior,
        eps=eps,
        random_state=random_state + 7919 * selected_beta_index,
        labels=labels,
        contexts=contexts,
    )
    curve["selected"] = curve.index == selected_index
    return IBSelection(model=selected_model, curve=curve)


def build_rna_reference_ib(
    adata,
    gtf=None,
    *,
    donor_col: str = "donor",
    tissue_col: str = "tissue",
    cell_type_col: str = "free_annotation",
    assay_col: str = "method",
    assays: Sequence[str] | None = ("10x",),
    anatomy_col: str | None = None,
    gene_col: str | None = None,
    layer: str | None = None,
    family_mapping: Mapping[str, str] | None = None,
    exclude_cell_types: Iterable[str] | None = None,
    min_cells: int = 50,
    min_counts: int = 50_000,
    min_donors: int = 2,
    min_context_donors: int = 2,
    tissue_ratio_threshold: float = 2.0,
    noise_floor: float = 1e-8,
    prior_mass: float = 1.0,
    eps: float = 1e-12,
    min_cell_counts: int = 100,
    min_cell_genes: int = 100,
    max_pct_counts_mt: float | None = None,
    chunk_size: int | None = 8192,
    memmap_path=None,
    ib_betas: Sequence[float] = (0.5, 1, 2, 4, 8, 16, 32, 64),
    ib_validation_folds: int = 3,
    ib_max_excess_nll: float = 1e-3,
    ib_min_retained_fraction: float = 0.98,
    ib_restarts: int = 3,
    ib_max_iter: int = 1000,
    ib_tol: float = 1e-9,
    ib_decoder_prior: float = 1e-8,
    random_state: int = 0,
    return_diagnostics: bool = False,
):
    """Build a donor-balanced RNA reference using conditional information bottleneck.

    This function converts a normal single-cell RNA atlas into a
    gene-by-reference probability matrix suitable for the RNA component of
    SpaceNumbat. It follows the preprocessing, context-testing and final
    reference-estimation stages of :func:`build_rna_reference`, but replaces
    greedy pairwise cell-type merging with a formal conditional information
    bottleneck (CIB). Compression is learned independently within each assay
    and curated biological family, so labels from unrelated lineages and
    different library protocols are never merged.

    The bottleneck variables are the original cell-type label ``L``, compressed
    label ``Z``, tissue/anatomical context ``C`` and gene identity ``G``. For a
    fixed value of ``beta``, the implementation minimizes

    ``I(L; Z) - beta * I(Z; G | C)``.

    Equivalently, apart from the constant ``I(L; G | C)``, it minimizes

    ``I(L; Z) + beta * E[KL(p(G | L, C) || p(G | Z, C))]``.

    ``I(L; Z)`` is the coding rate, and therefore penalizes retaining excessive
    label detail. The conditional mutual information ``I(Z; G | C)`` measures
    how much label-associated transcriptomic information remains after
    compression while treating context as side information. Larger ``beta``
    values preserve more cell-type-specific expression; smaller values favor
    stronger compression. Because all variables are discrete, the
    self-consistent bottleneck equations are solved directly by deterministic
    Blahut--Arimoto updates rather than by Monte Carlo variational inference.

    The complete procedure is:

    1. Validate cell and feature identifiers and map input features to the
       supplied autosomal gene annotation. Version suffixes are removed from
       Ensembl identifiers, and input columns mapping to the same gene are
       summed.
    2. Normalize cell-type and assay names, select assays, remove excluded
       annotations and construct tissue or tissue--anatomy contexts.
    3. Apply per-cell count, detected-gene and optional mitochondrial filters,
       then construct raw donor-by-context-by-label pseudobulks.
    4. Retain pseudobulks meeting ``min_cells`` and ``min_counts`` for learning
       the compression. Each donor pseudobulk is independently converted into
       a Dirichlet-smoothed gene-probability distribution. Donor profiles are
       averaged within label and context, so donors rather than individual
       cells are the biological replicates.
    5. Within every assay and biological family, fit the CIB over
       ``ib_betas``. For each beta, donor-blocked validation compares the
       compressed predictor with the corresponding uncompressed
       label-by-context predictor.
    6. Select the solution with the fewest hard clusters that both retains at
       least ``ib_min_retained_fraction`` of ``I(L; G | C)`` and increases
       held-out negative log likelihood by no more than
       ``ib_max_excess_nll`` nats per count. Ties favor lower information rate,
       then lower validation loss. If no candidate satisfies both constraints,
       the best held-out predictive solution is used. If donor folds cannot be
       evaluated, selection falls back to the empirical rate--distortion curve.
    7. Collapse numerically equivalent bottleneck codewords and convert the
       soft posterior ``q(Z | L)`` to a hard label mapping. Labels without an
       initial QC-passing pseudobulk are retained in a parent-family fallback
       group rather than silently discarded.
    8. Reaggregate *all* original pseudobulks using the compressed mapping and
       reapply ``min_cells`` and ``min_counts``. Thus several sparse related
       labels can jointly form a supported reference.
    9. For each compressed type, test whether context carries reproducible
       expression information relative to within-context donor variability.
       Emit context-specific references when the context-information ratio
       exceeds ``tissue_ratio_threshold``; otherwise emit one pan-context
       reference.
    10. Construct each final reference by normalizing donors independently,
        taking the component-wise median in log-probability space, exponentiating
        and renormalizing. Every retained donor therefore contributes equally
        to the final profile, irrespective of its cell or library count.

    Parameters
    ----------
    adata : anndata.AnnData
        Cell-by-feature atlas. ``adata.X`` or ``adata.layers[layer]`` must
        contain finite, non-negative, integer-valued raw counts. Normalized,
        log-transformed or scaled expression is not accepted. ``obs_names`` and
        ``var_names`` must be unique. The object is read but not modified. The
        atlas should represent non-malignant cells appropriate for construction
        of a diploid expression baseline.
    gtf : pandas.DataFrame, optional
        Gene annotation containing ``gene``, ``CHROM``, ``gene_start`` and
        ``gene_end``. An optional ``gene_ensembl`` column enables matching by
        Ensembl identifier. Chromosome prefixes are removed, only chromosomes
        1--22 are retained, and genes are ordered by chromosome and genomic
        position. Each value of ``gene`` must identify one unambiguous row.
        When omitted, ``spacenumbat.data.hg38`` is used.
    donor_col : str, default: ``"donor"``
        Column in ``adata.obs`` identifying independent biological donors.
        Donors define validation folds, replicate profiles, donor-noise
        estimates and equal-weight contributions to final references. This
        column must not encode individual cells or technical batches.
    tissue_col : str, default: ``"tissue"``
        Column in ``adata.obs`` containing the tissue label. Tissue defines the
        context when ``anatomy_col`` is ``None`` and supplies the tissue part of
        a composite context otherwise.
    cell_type_col : str, default: ``"free_annotation"``
        Column in ``adata.obs`` containing input cell-type annotations. Values
        are stripped, whitespace-normalized and case-folded before exclusions,
        family lookup and model fitting. Consequently, output source labels use
        their normalized representations.
    assay_col : str, default: ``"method"``
        Column in ``adata.obs`` identifying the sequencing protocol. Common
        spellings of 10x Chromium and Smart-seq2 are canonicalized to ``"10x"``
        and ``"smartseq2"``. Assays are processed independently, preventing
        protocol effects from driving cell-type compression.
    assays : sequence of str or None, default: ``("10x",)``
        Assays to retain after name canonicalization. Pass ``None`` to use all
        available assays. When more than one assay is processed, the assay name
        is appended to emitted reference names. Profiles from different assays
        are never pooled.
    anatomy_col : str or None, default: None
        Optional ``adata.obs`` column describing anatomical location within a
        tissue. When provided, context is encoded as
        ``"<tissue>::<anatomy>"``; missing anatomy values become ``"NA"``.
        Tissue alone defines context when this argument is ``None``.
    gene_col : str or None, default: None
        Column in ``adata.var`` containing gene symbols or Ensembl identifiers.
        By default, ``adata.var_names`` is used. Ensembl version suffixes are
        removed. Multiple input features mapping to the same annotated gene are
        summed through a sparse projection matrix.
    layer : str or None, default: None
        AnnData layer containing raw counts. ``None`` uses ``adata.X``.
    family_mapping : mapping of str to str or None, default: None
        Additional or replacement normalized-label-to-family assignments.
        Entries update the bundled biological family map. Family membership
        permits comparison and compression but does not force a merge. An
        unmapped label forms a singleton family and therefore cannot merge with
        biologically unrelated labels.
    exclude_cell_types : iterable of str or None, default: None
        Additional annotations to remove, after the same label normalization
        applied to ``cell_type_col``. These values extend rather than replace
        the bundled exclusions for composite, overly broad, cycling,
        germline/meiotic and anucleate populations.
    min_cells : int, default: 50
        Minimum number of QC-passing cells in a donor-by-context-by-label
        pseudobulk. Applied once before CIB fitting and again after labels have
        been compressed and counts reaggregated.
    min_counts : int, default: 50000
        Minimum mapped autosomal raw counts in each pseudobulk, before and after
        compression. Increase this value to reduce sampling noise, but do not
        use it as the primary control over the number of references because it
        removes biological groups rather than optimizing compression.
    min_donors : int, default: 2
        Minimum number of distinct donors required to emit a final reference.
        A fitted or fallback group with fewer donors is omitted.
    min_context_donors : int, default: 2
        Minimum donors required for a context to participate in the downstream
        context-information test. At least two eligible contexts are necessary
        to emit context-specific references. When a split is selected, only
        eligible contexts are emitted.
    tissue_ratio_threshold : float, default: 2.0
        Context-information-to-donor-noise ratio above which a compressed cell
        type is separated into context-specific references. The numerator is
        generalized Jensen--Shannon divergence among donor-balanced context
        consensuses. The denominator is the maximum of local donor variability,
        assay-wide donor variability and ``noise_floor``. This is an empirical
        decision statistic, not a p-value. Larger values favor pan-context
        references.
    noise_floor : float, default: 1e-8
        Positive lower bound used in donor-noise ratios to prevent unstable
        decisions when replicate profiles are nearly identical. This parameter
        affects context splitting, not the CIB rate--distortion objective.
    prior_mass : float, default: 1.0
        Total concentration of the symmetric Dirichlet prior used to convert
        raw donor pseudobulks into gene probabilities. For ``J`` retained genes,
        ``prior_mass / J`` is added to every gene count. The resulting posterior
        mean is ``(count_j + prior_mass/J) / (depth + prior_mass)``. This avoids
        zero probabilities and stabilizes KL divergences without making the
        total prior strength depend on the number of genes.
    eps : float, default: 1e-12
        Numerical lower bound used in logarithms, KL calculations and the final
        log-median consensus. It should be positive and much smaller than
        biologically meaningful gene probabilities.
    min_cell_counts : int, default: 100
        Minimum mapped autosomal raw counts required for an individual cell
        before it contributes to a pseudobulk.
    min_cell_genes : int, default: 100
        Minimum number of detected mapped autosomal genes required for an
        individual cell.
    max_pct_counts_mt : float or None, default: None
        Optional maximum value of ``adata.obs["pct_counts_mt"]``. When set,
        cells with missing, non-numeric or larger mitochondrial percentages are
        excluded. No mitochondrial filter is applied when ``None``.
    chunk_size : int or None, default: 8192
        Number of cells processed in each block during feature projection,
        cell-level QC and pseudobulk construction. Reducing this value lowers
        transient sparse-matrix memory at the cost of additional slicing,
        sparse-multiplication and accumulation overhead. Larger values can be
        faster when sufficient memory is available. Pass ``None`` to process
        all cells in one block. This affects execution strategy and peak memory,
        not the intended statistical estimator or information-bottleneck model.
    memmap_path : path-like or None, default: None
        Optional path for the dense float64 initial pseudobulk-by-gene count
        matrix. The path must not already exist and is not a resumable cache.
        When ``None``, this matrix is held in memory. Subsequent bottleneck
        tensors are still allocated in memory.
    ib_betas : sequence of float, default: ``(0.5, 1, 2, 4, 8, 16, 32, 64)``
        Positive CIB trade-off values evaluated separately for every assay and
        multi-label family. Small values strongly penalize retained label
        information and commonly yield fewer references; large values favor
        transcriptomic fidelity. The final reference count is *not* forced to
        equal the length of this sequence or any user-specified target. Include
        a sufficiently broad logarithmic grid so the observed curve spans both
        compressed and nearly uncompressed regimes.
    ib_validation_folds : int, default: 3
        Number of donor-blocked folds used to evaluate each beta. Donors are
        shuffled reproducibly and kept intact within folds. A fold is skipped
        when its training partition lacks any label in the family. If no fold
        is evaluable, selection uses the full-data rate--distortion curve rather
        than held-out likelihood. Values of 3--5 are generally appropriate;
        the number cannot exceed the practical donor support of rare labels.
    ib_max_excess_nll : float, default: 1e-3
        Largest permitted increase in held-out negative log likelihood, in
        natural-log units per raw count, relative to an uncompressed
        label-by-context predictor. Smaller values are more conservative. A
        negative value observed in diagnostics means that compression improved
        held-out prediction through shrinkage, although the argument itself
        must be non-negative.
    ib_min_retained_fraction : float, default: 0.98
        Minimum acceptable fraction of conditional label--gene information:
        ``I(Z; G | C) / I(L; G | C)``. Must lie in ``(0, 1]``. Values near one
        protect subtle transcriptomic subtypes; lower values permit stronger
        compression. For a conservative reference, consider 0.99 together with
        a stricter ``ib_max_excess_nll``.
    ib_restarts : int, default: 3
        Number of initializations used for each beta and validation fold. The
        lowest CIB objective is retained. The objective is non-convex, so more
        restarts reduce sensitivity to local stationary points while increasing
        runtime approximately linearly.
    ib_max_iter : int, default: 1000
        Maximum Blahut--Arimoto updates for each initialization.
    ib_tol : float, default: 1e-9
        Convergence tolerance applied jointly to the largest change in
        ``q(Z | L)`` and the change in the rate--distortion objective. Relaxing
        the tolerance shortens runtime; tightening it may be useful when beta
        values lie near a compression transition.
    ib_decoder_prior : float, default: 1e-8
        Small probability mass that shrinks each cluster--context decoder
        toward the pooled profile of that context. It prevents undefined
        decoders when a tentative cluster has negligible support in a context.
        This value is on the empirical probability-mass scale, not the raw-count
        or Dirichlet-prior scale.
    random_state : int, default: 0
        Seed controlling donor-fold shuffling and randomized bottleneck
        initializations. Identical input ordering, numerical libraries and seed
        are required for exact reproducibility.
    return_diagnostics : bool, default: False
        If ``True``, append a diagnostics dictionary to the returned tuple.

    Returns
    -------
    reference : pandas.DataFrame
        Gene-by-reference probability matrix. Rows are expressed autosomal genes
        in genomic order. Columns are emitted pan-context or context-specific
        profiles. Values are finite and strictly positive, and every column sums
        to one. This matrix can be passed as the RNA ``lambdas_ref`` input to
        SpaceNumbat.
    manifest : pandas.DataFrame
        One row per column of ``reference``. Important fields include:

        - ``reference``: exact reference column name;
        - ``cell_type`` and ``family``: compressed identity and parent family;
        - ``assay``: independently processed protocol;
        - ``scope``: ``"pan_context"`` or ``"context"``;
        - ``context``, ``contexts`` and ``tissues``: emitted and contributing
          biological contexts;
        - ``donors`` and ``n_donors``: contributing independent donors;
        - ``n_cells`` and ``n_counts``: aggregate support;
        - ``source_cell_types``: original annotations represented by the column;
        - ``context_js``, ``context_noise`` and ``context_ratio``: statistics
          governing context splitting.

        List-valued fields are JSON-encoded strings.
    gene_gtf : pandas.DataFrame
        Genomic annotation for the rows of ``reference``. It contains at least
        ``gene``, ``CHROM``, ``gene_start`` and ``gene_end`` and is ordered
        identically to the reference matrix.
    diagnostics : dict, optional
        Returned only when ``return_diagnostics=True``. The dictionary contains:

        - ``cell_type_map``: original-to-compressed mapping, hard cluster,
          selected beta and maximum soft assignment probability;
        - ``ib_models``: selected beta, rate, distortion, source information,
          retained information, retained fraction, hard/effective cluster counts
          and convergence status for every modeled assay--family pair;
        - ``ib_curves``: complete beta-wise rate--distortion and held-out
          prediction curve. ``selected`` marks the chosen row;
        - ``context_tests``: downstream pan-context versus context-specific
          decisions;
        - ``initial_pseudobulk_qc`` and ``merged_pseudobulk_qc``: support and QC
          status before and after compression;
        - ``global_donor_noise``: assay-specific empirical donor JSD;
        - ``feature_mapping``: input-feature-to-gene mapping and mapping status;
        - ``run_info``: algorithm version, dimensions, thresholds, random seed,
          cell-QC summary and family-map checksum.

    Raises
    ------
    ValueError
        If required metadata or annotation fields are absent; cell or feature
        identifiers are duplicated; the selected layer is absent; input counts
        are non-finite, negative or non-integer; parameter values are invalid;
        fewer than two expressed autosomal genes remain; or no initial/final
        pseudobulks satisfy support requirements.
    FileExistsError
        If ``memmap_path`` already exists. Existing files are never overwritten.
    RuntimeError
        If emitted names collide or the final profiles fail positivity,
        finiteness or unit-sum validation.

    Notes
    -----
    This procedure is donor-balanced but not a complete hierarchical generative
    model. Inter-donor variation is handled through independent donor profiles,
    blocked validation, empirical donor-noise estimates and a robust final
    consensus; no explicit donor random-effect distribution is fitted.

    ``beta`` is a Lagrange multiplier, not a probability, p-value or direct
    requested cluster count. The final number of references emerges from the
    rate--distortion curve, biological-family constraints, donor support,
    context decisions and post-compression QC. There is intentionally no
    ``target_references`` parameter.

    The CIB objective is non-convex. Blahut--Arimoto iterations converge to a
    self-consistent stationary solution, not necessarily the unique global
    optimum. Multiple restarts and donor-blocked validation mitigate, but do
    not eliminate, initialization and finite-sample uncertainty.

    Conditioning on context allows labels observed in different tissues to
    share a compressed code when context itself explains their expression
    difference. Unlike the former zero-score rule, labels without a shared
    context are not automatically declared identical: their compatibility is
    evaluated through the learned context-conditioned decoders and pooled
    decoder prior. Nevertheless, annotations perfectly confounded with context
    remain only weakly identifiable and should be inspected manually.

    The final context-splitting stage still uses the Jensen--Shannon
    information-to-donor-noise rule from :func:`build_rna_reference`. Thus CIB
    controls cell-type compression, while context granularity is controlled by
    ``tissue_ratio_threshold``.

    Memory use during CIB fitting is dominated by family-specific tensors of
    approximate size ``O((L*C + K*C)*G)``, where ``L``, ``K``, ``C`` and ``G``
    are the numbers of labels, bottleneck codewords, contexts and genes in the
    current family. Runtime additionally scales with the beta grid, validation
    folds, restarts and iterations. Process assays or reduce the beta grid and
    restart count during exploratory runs when memory or runtime is limiting.

    A reference intended as a diploid baseline should be constructed from
    independently verified non-malignant cells. Normal-tissue annotation alone
    does not exclude occult aneuploidy, malignant contamination, doublets or
    strong target-atlas domain shift. Model selection on reference donors also
    does not guarantee reduced SpaceNumbat residual noise in a different target
    assay; validate the final reference on held-out diploid target samples.

    Examples
    --------
    Build a conservative 10x reference and request diagnostics:

    >>> from spacenumbat.preprocessing.build_rna_ref_ib import build_rna_reference_ib
    >>> reference, manifest, gene_gtf, diagnostics = build_rna_reference_ib(
    ...     adata_full,
    ...     gtf=spacenumbat.data.hg38,
    ...     donor_col="donor",
    ...     tissue_col="tissue",
    ...     cell_type_col="free_annotation",
    ...     assay_col="method",
    ...     assays=("10x",),
    ...     anatomy_col=None,
    ...     min_cells=50,
    ...     min_counts=50_000,
    ...     min_donors=2,
    ...     ib_betas=(1, 2, 4, 8, 16, 32, 64, 128),
    ...     ib_validation_folds=5,
    ...     ib_max_excess_nll=5e-4,
    ...     ib_min_retained_fraction=0.99,
    ...     ib_restarts=5,
    ...     random_state=42,
    ...     return_diagnostics=True,
    ... )

    Verify output invariants:

    >>> reference.index.equals(pd.Index(gene_gtf["gene"], name="gene"))
    True
    >>> reference.columns.tolist() == manifest["reference"].tolist()
    True
    >>> bool(np.isfinite(reference.to_numpy()).all())
    True
    >>> bool((reference.to_numpy() > 0).all())
    True
    >>> bool(np.allclose(reference.sum(axis=0), 1.0))
    True

    Inspect the chosen rate--distortion points and uncertain assignments:

    >>> selected = diagnostics["ib_curves"].query("selected")
    >>> selected[["assay", "family", "beta", "n_clusters",
    ...           "retained_fraction", "excess_validation_nll"]]
    >>> uncertain = diagnostics["cell_type_map"].query(
    ...     "assignment_probability < 0.9"
    ... )

    A negative ``excess_validation_nll`` indicates that the compressed model
    predicted held-out donor counts better than the uncompressed reference.
    Such improvement is consistent with beneficial shrinkage, but should still
    be confirmed using independent diploid target data and downstream
    SpaceNumbat residual and false-positive CNA metrics.
    """
    from .build_rna_ref import (
        EXCLUDE_CELL_TYPES,
        FAMILY_MAPPING,
        _aggregate_counts,
        _aggregate_rows,
        _assay,
        _gene_mapping,
        _global_donor_noise,
        _label,
        _robust_donor_consensus,
        _test_context_information,
        _valid,
    )

    if gtf is None:
        from spacenumbat.data import hg38

        gtf = hg38
    if adata.obs_names.has_duplicates or adata.var_names.has_duplicates:
        raise ValueError("Cell and feature identifiers must be unique.")
    if layer is not None and layer not in adata.layers:
        raise ValueError(f"AnnData has no layer {layer!r}.")
    required = [donor_col, tissue_col, cell_type_col, assay_col]
    if anatomy_col is not None:
        required.append(anatomy_col)
    if max_pct_counts_mt is not None:
        required.append("pct_counts_mt")
    missing = set(required) - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing obs columns: {sorted(missing)}")

    gene_gtf, projection, feature_map = _gene_mapping(adata, gtf, gene_col)
    source_obs = adata.obs
    meta = pd.DataFrame(index=source_obs.index)
    meta["donor"] = source_obs[donor_col].astype("string").str.strip()
    meta["tissue"] = source_obs[tissue_col].astype("string").str.strip()
    meta["label"] = source_obs[cell_type_col].astype("string").map(_label)
    meta["assay"] = source_obs[assay_col].astype("string").map(_assay)
    if anatomy_col is None:
        meta["context"] = meta.tissue
    else:
        anatomy = source_obs[anatomy_col].astype("string").fillna("NA").str.strip()
        meta["context"] = meta.tissue.astype(str) + "::" + anatomy.astype(str)
    families = dict(FAMILY_MAPPING)
    if family_mapping is not None:
        families.update({_label(k): str(v) for k, v in family_mapping.items()})
    meta["family"] = meta.label.map(families).fillna(meta.label)
    excluded = set(EXCLUDE_CELL_TYPES)
    if exclude_cell_types is not None:
        excluded |= {_label(x) for x in exclude_cell_types}
    keep = (
        _valid(source_obs[donor_col]).to_numpy()
        & _valid(source_obs[tissue_col]).to_numpy()
        & _valid(source_obs[cell_type_col]).to_numpy()
        & _valid(source_obs[assay_col]).to_numpy()
        & ~meta.label.isin(excluded).to_numpy()
    )
    if assays is not None:
        selected_assays = {_assay(x) for x in assays}
        keep &= meta.assay.isin(selected_assays).to_numpy()
    meta["_keep"] = keep

    counts, pb, cell_qc = _aggregate_counts(
        adata,
        meta,
        projection,
        layer=layer,
        chunk_size=chunk_size,
        min_cell_counts=min_cell_counts,
        min_cell_genes=min_cell_genes,
        max_pct_counts_mt=max_pct_counts_mt,
        memmap_path=memmap_path,
    )
    expressed = counts.sum(axis=0) > 0
    counts = np.asarray(counts[:, expressed], dtype=np.float64)
    gene_gtf = gene_gtf.loc[expressed].reset_index(drop=True)
    if counts.shape[1] < 2:
        raise ValueError("Fewer than two expressed autosomal genes remain.")
    initial_good = ((pb.n_cells >= min_cells) & (pb.n_counts >= min_counts)).to_numpy()
    pb["initial_qc_pass"] = initial_good
    test_counts = counts[initial_good]
    test_obs = pb.loc[initial_good].reset_index(drop=True)
    if test_obs.empty:
        raise ValueError("No donor/context/cell-type pseudobulks passed QC.")

    references: dict[str, np.ndarray] = {}
    manifest_rows = []
    mapping_rows = []
    ib_curve_rows = []
    ib_model_rows = []
    context_rows = []
    merged_qc_rows = []
    noise_rows = []
    assay_values = sorted(test_obs.assay.unique())

    for assay_index, assay in enumerate(assay_values):
        assay_mask = test_obs.assay.eq(assay).to_numpy()
        assay_counts = test_counts[assay_mask]
        assay_obs = test_obs.loc[assay_mask].reset_index(drop=True)
        assay_all_obs = pb.loc[pb.assay.eq(assay)].reset_index(drop=True)
        global_noise = _global_donor_noise(assay_counts, assay_obs, prior_mass, noise_floor)
        noise_rows.append({"assay": assay, "global_donor_js": global_noise})
        original_to_merged: dict[str, str] = {}

        for family_index, family in enumerate(sorted(assay_all_obs.family.unique())):
            all_labels = sorted(
                assay_all_obs.loc[assay_all_obs.family.eq(family), "label"].unique()
            )
            family_mask = assay_obs.family.eq(family).to_numpy()
            family_counts = assay_counts[family_mask]
            family_obs = assay_obs.loc[family_mask].reset_index(drop=True)
            fitted_labels = sorted(family_obs.label.unique())

            if len(fitted_labels) <= 1:
                selected = None
                assignment = np.zeros(len(fitted_labels), dtype=int)
            else:
                selected = select_information_bottleneck(
                    family_counts,
                    family_obs,
                    betas=ib_betas,
                    label_col="label",
                    context_col="context",
                    donor_col="donor",
                    prior_mass=prior_mass,
                    n_folds=ib_validation_folds,
                    max_excess_nll=ib_max_excess_nll,
                    min_retained_fraction=ib_min_retained_fraction,
                    n_restarts=ib_restarts,
                    max_iter=ib_max_iter,
                    tol=ib_tol,
                    decoder_prior=ib_decoder_prior,
                    eps=eps,
                    random_state=random_state + 100_003 * assay_index + family_index,
                )
                assignment = selected.model.hard_assignment
                curve = selected.curve.copy()
                curve.insert(0, "family", family)
                curve.insert(0, "assay", assay)
                ib_curve_rows.append(curve)
                model = selected.model
                ib_model_rows.append(
                    {
                        "assay": assay,
                        "family": family,
                        "beta": model.beta,
                        "n_input_labels": len(model.labels),
                        "n_clusters": model.n_clusters,
                        "effective_clusters": model.effective_clusters,
                        "rate": model.rate,
                        "distortion": model.distortion,
                        "source_information": model.source_information,
                        "retained_information": model.retained_information,
                        "retained_fraction": model.retained_fraction,
                        "converged": model.converged,
                    }
                )

            groups = {}
            for label, cluster in zip(fitted_labels, assignment):
                groups.setdefault(int(cluster), []).append(label)
            if len(groups) == 1:
                names = {next(iter(groups)): str(family)}
            else:
                names = {
                    cluster: (
                        labels_in_cluster[0]
                        if len(labels_in_cluster) == 1
                        else f"{family} IB group {index}"
                    )
                    for index, (cluster, labels_in_cluster) in enumerate(
                        sorted(groups.items()), start=1
                    )
                }
            for cluster, labels_in_cluster in groups.items():
                for label in labels_in_cluster:
                    original_to_merged[label] = names[cluster]
                    row = {
                        "assay": assay,
                        "original_cell_type": label,
                        "family": family,
                        "merged_cell_type": names[cluster],
                        "ib_cluster": cluster,
                    }
                    if selected is not None:
                        li = selected.model.labels.index(label)
                        row["assignment_probability"] = float(
                            selected.model.q_cluster_given_label[li].max()
                        )
                        row["beta"] = selected.model.beta
                    else:
                        row["assignment_probability"] = 1.0
                        row["beta"] = np.nan
                    mapping_rows.append(row)

            # Labels lacking an initial QC-passing pseudobulk remain available
            # as a parent-family fallback rather than being silently discarded.
            absent_labels = sorted(set(all_labels) - set(fitted_labels))
            fallback_name = f"{family} fallback"
            for label in absent_labels:
                original_to_merged[label] = fallback_name
                mapping_rows.append(
                    {
                        "assay": assay,
                        "original_cell_type": label,
                        "family": family,
                        "merged_cell_type": fallback_name,
                        "ib_cluster": -1,
                        "assignment_probability": np.nan,
                        "beta": np.nan,
                    }
                )

        source_mask = pb.assay.eq(assay).to_numpy()
        source_group_obs = pb.loc[source_mask].copy().reset_index(drop=True)
        source_group_counts = counts[source_mask]
        source_group_obs["merged_type"] = source_group_obs.label.map(original_to_merged)
        valid = source_group_obs.merged_type.notna().to_numpy()
        merged_counts, merged_obs = _aggregate_rows(
            source_group_counts[valid],
            source_group_obs.loc[valid].reset_index(drop=True),
            ["context", "family", "merged_type", "donor"],
        )
        merged_good = (
            (merged_obs.n_cells >= min_cells) & (merged_obs.n_counts >= min_counts)
        ).to_numpy()
        merged_obs["qc_pass"] = merged_good
        merged_obs["assay"] = assay
        merged_qc_rows.append(merged_obs.copy())
        merged_counts = np.asarray(merged_counts[merged_good], dtype=np.float64)
        merged_obs = merged_obs.loc[merged_good].reset_index(drop=True)
        source_types = (
            pd.DataFrame(
                [{"original": key, "merged": value} for key, value in original_to_merged.items()]
            )
            .groupby("merged")
            .original.agg(list)
            .to_dict()
        )

        for merged_type, idx in merged_obs.groupby("merged_type", sort=True).indices.items():
            idx = np.asarray(idx)
            type_counts = merged_counts[idx]
            type_obs = merged_obs.iloc[idx].reset_index(drop=True)
            family = str(type_obs.family.iloc[0])
            context_result = _test_context_information(
                type_counts,
                type_obs,
                min_context_donors=min_context_donors,
                prior_mass=prior_mass,
                global_noise=global_noise,
                ratio_threshold=tissue_ratio_threshold,
                noise_floor=noise_floor,
            )
            context_rows.append(
                {
                    "assay": assay,
                    "cell_type": merged_type,
                    "family": family,
                    "split": context_result["split"],
                    "context_js": context_result["context_js"],
                    "donor_noise": context_result["donor_noise"],
                    "ratio": context_result["ratio"],
                    "eligible_contexts": json.dumps(context_result["eligible_contexts"]),
                }
            )
            if context_result["split"]:
                output_groups = []
                for context in context_result["eligible_contexts"]:
                    mask = type_obs.context.astype(str).eq(context).to_numpy()
                    output_groups.append(
                        ("context", context, type_counts[mask], type_obs.loc[mask])
                    )
            else:
                output_groups = [("pan_context", "pan_context", type_counts, type_obs)]

            for scope, context, group_counts, group_obs in output_groups:
                reference, donors_in_reference = _robust_donor_consensus(
                    group_counts,
                    group_obs.reset_index(drop=True),
                    prior_mass=prior_mass,
                    eps=eps,
                )
                if len(donors_in_reference) < min_donors:
                    continue
                suffix = f"|{assay}" if len(assay_values) > 1 else ""
                name = (
                    f"{merged_type}|{context}{suffix}"
                    if scope == "context"
                    else f"{merged_type}{suffix}"
                )
                if name in references:
                    raise RuntimeError(f"Reference name collision: {name}")
                references[name] = reference
                contexts_used = sorted(group_obs.context.astype(str).unique())
                tissues = sorted({value.split("::", 1)[0] for value in contexts_used})
                manifest_rows.append(
                    {
                        "reference": name,
                        "cell_type": merged_type,
                        "family": family,
                        "assay": assay,
                        "scope": scope,
                        "context": context,
                        "contexts": json.dumps(contexts_used),
                        "tissues": json.dumps(tissues),
                        "donors": json.dumps(sorted(donors_in_reference)),
                        "n_donors": len(donors_in_reference),
                        "n_cells": int(group_obs.n_cells.sum()),
                        "n_counts": float(group_obs.n_counts.sum()),
                        "source_cell_types": json.dumps(
                            sorted(source_types.get(merged_type, []))
                        ),
                        "context_js": context_result["context_js"],
                        "context_noise": context_result["donor_noise"],
                        "context_ratio": context_result["ratio"],
                    }
                )

    if not references:
        raise ValueError("No final RNA reference profiles were generated.")
    reference = pd.DataFrame(
        references, index=pd.Index(gene_gtf.gene, name="gene"), dtype=np.float64
    )
    if not np.isfinite(reference.to_numpy()).all() or (reference <= 0).any().any():
        raise RuntimeError("Reference profiles must be finite and positive.")
    if not np.allclose(reference.sum(axis=0), 1.0, atol=1e-12):
        raise RuntimeError("Reference normalization failed.")
    manifest = pd.DataFrame(manifest_rows)
    diagnostics = {
        "cell_type_map": pd.DataFrame(mapping_rows),
        "ib_models": pd.DataFrame(ib_model_rows),
        "ib_curves": (
            pd.concat(ib_curve_rows, ignore_index=True) if ib_curve_rows else pd.DataFrame()
        ),
        "context_tests": pd.DataFrame(context_rows),
        "initial_pseudobulk_qc": pb,
        "merged_pseudobulk_qc": (
            pd.concat(merged_qc_rows, ignore_index=True)
            if merged_qc_rows
            else pd.DataFrame()
        ),
        "global_donor_noise": pd.DataFrame(noise_rows),
        "feature_mapping": feature_map,
        "run_info": {
            "builder_version": __version__,
            "algorithm": "donor_aware_conditional_information_bottleneck",
            "n_input_cells": int(adata.n_obs),
            "n_input_genes": int(adata.n_vars),
            "n_genes": int(reference.shape[0]),
            "n_references": int(reference.shape[1]),
            "cell_qc": cell_qc,
            "ib_betas": list(map(float, ib_betas)),
            "ib_validation_folds": ib_validation_folds,
            "ib_max_excess_nll": ib_max_excess_nll,
            "ib_min_retained_fraction": ib_min_retained_fraction,
            "tissue_ratio_threshold": tissue_ratio_threshold,
            "random_state": random_state,
            "family_mapping_sha256": hashlib.sha256(
                json.dumps(families, sort_keys=True).encode()
            ).hexdigest(),
        },
    }
    result = (reference, manifest, gene_gtf)
    return (*result, diagnostics) if return_diagnostics else result


__all__ = [
    "ConditionalIBResult",
    "IBSelection",
    "fit_conditional_information_bottleneck",
    "select_information_bottleneck",
    "build_rna_reference_ib",
]


