#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 19:30:40 2025

@author: lillux
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, Dict, Tuple, Any, Iterable, Set, List, Union
from pathlib import Path

import numpy as np
import pandas as pd

import anndata as ad

import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
from matplotlib import gridspec

from natsort import natsorted, natsort_keygen

from scipy.cluster.hierarchy import dendrogram, leaves_list, fcluster, linkage
from scipy.spatial.distance import pdist


def plot_psbulk(
    bulk: pd.DataFrame,
    use_pos: bool = True,
    allele_only: bool = False,
    min_LLR: float = 5.0,
    min_depth: int = 8,
    exp_limit: float = 2.0,
    phi_mle: bool = True,
    theta_roll: bool = False,
    dot_size: float = 8.0,
    dot_alpha: float = 0.5,
    legend: bool = True,
    exclude_gap: bool = True,
    genome: str = "hg38",
    text_size: int = 10,
    raster: bool = False,
    *,
    parent: Optional[plt.Figure] = None,   # draw into this Figure/SubFigure if provided
    cnv_colors: Optional[Dict[str, str]] = None,
    cnv_labels: Optional[Dict[str, str]] = None,
    gaps: Optional[pd.DataFrame] = None,   # columns: CHROM, start, end
    acen: Optional[pd.DataFrame] = None,   # columns: CHROM, start, end
    ) -> Tuple[plt.Figure, np.ndarray]:
    """
    Plot a single pseudobulk HMM panel (optionally inside a provided Figure/SubFigure).

    The panel shows up to two tracks for each chromosome, arranged as facet columns:
    - ``logFC`` (expression log fold-change, optional)
    - ``pHF`` (pseudo-heterozygous fraction from SNP alleles)

    Chromosomes are arranged using natural sorting (for example: 1, 2, ..., 10, X, Y),
    and each chromosome column width is proportional to its x-span (POS or snp_index).
    If ``parent`` is provided, the subplots are created inside that Figure/SubFigure;
    otherwise a new Figure is created and returned.

    Parameters
    ----------
    bulk
        Long or wide pseudobulk table with at least these columns:
        ``CHROM``, ``state_post`` or ``state``, ``cnv_state_post`` or ``cnv_state``,
        and columns for the tracks being drawn:
          - Expression track requires: ``logFC`` and ``mu`` (baseline), unless
            ``allele_only=True``.
          - Allele track requires: ``pBAF`` and ``DP`` (depth), used to form ``pHF``.
        Optional columns used when present:
        ``LLR`` (event filter), ``loh`` (boolean), ``p_up`` (allele HMM), segment
        metadata (``seg``, ``seg_start``, ``seg_end``, ``seg_start_index``,
        ``seg_end_index``), ``phi_mle`` or ``phi_mle_roll``, and ``theta_hat_roll``.
    use_pos
        If True, use ``POS`` as x; otherwise use ``snp_index``.
    allele_only
        If True, plot only the allele track (``pHF``) and omit ``logFC``.
    min_LLR
        LLR threshold below which events are set to neutral in ``state_post`` and
        ``cnv_state_post`` (if ``LLR`` column exists).
    min_depth
        Minimum depth (``DP``) for a SNP to contribute to ``pHF``; others are masked.
    exp_limit
        Symmetric y-limits for the expression track (``[-exp_limit, +exp_limit]``).
        Values outside this range are hidden.
    phi_mle
        If True and ``phi_mle`` exists, draw per-segment horizontal lines at
        ``log2(phi_mle)`` on the expression track. If False and ``phi_mle_roll``
        exists, draw the rolling estimate instead.
    theta_roll
        If True and ``theta_hat_roll`` exists, overlay two allele-imbalance curves
        around 0.5 on the allele track.
    dot_size
        Marker size for SNP scatter points.
    dot_alpha
        Alpha for level-1 markers (round). Level-2 markers (square) are drawn opaque.
    legend
        If True, this function prepares legend handles but does not place a legend
        by default when ``parent`` is used in a larger panel. Leave True when calling
        this function standalone; set False when it is embedded by a wrapper that
        adds a figure-level legend.
    exclude_gap
        If True and both ``gaps`` and ``acen`` are provided, shade those regions.
    genome
        Reserved for compatibility with upstream API. Not used by this function.
    text_size
        Base font size for titles and axis labels.
    raster
        If True, rasterize the scatter layers (useful for large point counts).
    parent
        A ``matplotlib.figure.Figure`` or ``SubFigure`` to draw into. If ``None``,
        a new Figure is created and returned.
    cnv_colors
        Mapping from state label (e.g., ``"amp_up"``) to color hex string.
        A sensible default is used when not provided.
    cnv_labels
        Mapping from state label to a human-readable legend label. Defaults to identity.
    gaps
        Genomic gap table with columns ``CHROM``, ``start``, ``end``; shaded if provided.
    acen
        Centromere table with columns ``CHROM``, ``start``, ``end``; shaded if provided.

    Returns
    -------
    (fig, axes)
        ``fig`` is the Figure that contains the panel (the provided ``parent`` when
        not None, otherwise a newly created Figure). ``axes`` is a 2D array-like
        of Axes with shape ``(n_tracks, n_chromosomes)``. ``n_tracks`` is 1 when
        ``allele_only=True`` and 2 otherwise.

    Raises
    ------
    KeyError
        If the chosen x marker (``POS`` or ``snp_index``) is missing, or if
        required columns for the selected tracks are missing (``pBAF``, ``DP``
        for allele; ``logFC``, ``mu`` for expression).

    Notes
    -----
    - The function will derive ``pHF`` as ``pBAF`` masked by ``DP >= min_depth``.
    - When ``p_up`` is available, it refines ``state_post`` into up/down sublabels
      per CNV state by combining with a detected theta level.
    - Natural chromosome ordering.

    Examples
    --------
    >>> fig, axes = plot_psbulk(bulk_df, use_pos=True, exp_limit=3.0)
    >>> fig.savefig("panel.png", dpi=300)
    """
    df = bulk.copy()

    # Ensure post columns exist
    if not {"state_post", "cnv_state_post"}.issubset(df.columns):
        if "state" in df.columns:
            df["state_post"] = df["state"]
        if "cnv_state" in df.columns:
            df["cnv_state_post"] = df["cnv_state"]

    # LLR filter to neutral
    if "LLR" in df.columns and min_LLR != 0:
        df["LLR"] = df["LLR"].fillna(0.0)
        neu_mask = df["LLR"] < min_LLR
        df.loc[neu_mask, "cnv_state_post"] = "neu"
        df.loc[neu_mask, "state_post"] = "neu"

    # Mark clonal LOH as deletions (as in the R code)
    if "loh" in df.columns:
        df.loc[df["loh"].astype(bool), "state_post"] = "del"

    # Which x variable?
    marker = "POS" if use_pos else "snp_index"
    if marker not in df.columns:
        raise KeyError(f"{marker} column is required for plotting")

    # Up/down retest handling when p_up present
    if "p_up" in df.columns:
        theta_level = np.where(df["state_post"].astype(str).str.contains("_2"), 2, 1)
        df["_theta_level"] = theta_level
        mask_target = df["cnv_state_post"].isin(["amp", "loh", "del"])
        up_mask = (df["p_up"] > 0.5) & mask_target
        down_mask = (~up_mask) & mask_target
        df.loc[up_mask, "state_post"] = (
            df.loc[up_mask, "cnv_state_post"].astype(str)
            + "_"
            + df.loc[up_mask, "_theta_level"].astype(str)
            + "_up"
        )
        df.loc[down_mask, "state_post"] = (
            df.loc[down_mask, "cnv_state_post"].astype(str)
            + "_"
            + df.loc[down_mask, "_theta_level"].astype(str)
            + "_down"
        )

    # Baseline correction for expression track
    if not allele_only and "logFC" in df.columns and "mu" in df.columns:
        #df["logFC"] = df["logFC"] - df["mu"]
        df["logFC"] = df["logFC"] - (df["mu"] / np.log(2.0))

    # Build long-form data (logFC, pHF)
    D = df.copy()
    if "pBAF" not in D.columns or "DP" not in D.columns:
        raise KeyError("Columns pBAF and DP are required to construct the pHF track.")
    if "logFC" in D.columns:
        D["logFC"] = D["logFC"].where(~((D["logFC"] > exp_limit) | (D["logFC"] < -exp_limit)), np.nan)
    D["pHF"] = D["pBAF"].where(D["DP"] >= min_depth, np.nan)

    variables = ["pHF"] if allele_only else ["logFC", "pHF"]

    long = D.melt(
        id_vars=[c for c in D.columns if c not in {"logFC", "pHF"}],
        value_vars=variables,
        var_name="variable",
        value_name="value",
    )

    # Natural chromosome ordering
    chroms = natsorted(map(str, long["CHROM"].astype(str).unique()))
    nrows, ncols = len(variables), len(chroms)

    # Compute width ratios from marker span per chromosome
    width_ratios = []
    for chrom in chroms:
        chx = long.loc[long["CHROM"].astype(str) == chrom, marker].to_numpy()
        span = float(np.nanmax(chx) - np.nanmin(chx) + 1.0) if chx.size else 1.0
        width_ratios.append(max(span, 1.0))

    # Create axes grid (in parent if provided; otherwise new Figure)
    if parent is None:
        fig = plt.figure(figsize=(1.5 * ncols, 1.2 * max(1, nrows)))
        gs = fig.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.08, hspace=0.4, width_ratios=width_ratios)
        axes = np.array([[fig.add_subplot(gs[ri, ci]) for ci in range(ncols)] for ri in range(nrows)])
    else:
        fig = parent  # might be a Figure or a SubFigure
        gs = parent.add_gridspec(nrows=nrows, ncols=ncols, wspace=0.08, hspace=0.4, width_ratios=width_ratios)
        axes = np.array([[parent.add_subplot(gs[ri, ci]) for ci in range(ncols)] for ri in range(nrows)])
    
    # Optional gap/acen shading (only with POS)
    segs_exclude = None
    if use_pos and exclude_gap and gaps is not None and acen is not None:
        se = pd.concat([gaps, acen], ignore_index=True)
        se = se.rename(columns={"start": "seg_start", "end": "seg_end"})
        se = se[se["CHROM"].astype(str).isin(chroms)].copy()
        segs_exclude = se

    # Colors / labels
    if cnv_colors is None:
        cnv_colors = {
            "neu": "#9e9e9e",
            "loh_up": "#4a90e2", "loh_down": "#357ABD",
            "del_up": "#d0021b", "del_down": "#a80012",
            "amp_up": "#7ed321", "amp_down": "#5fa015",
            "bamp": "#f5a623", "bdel": "#8b572a",
            "loh_2_up": "#4a90e2", "loh_2_down": "#357ABD",
            "del_2_up": "#d0021b", "del_2_down": "#a80012",
            "amp_2_up": "#7ed321", "amp_2_down": "#5fa015",
        }
    if cnv_labels is None:
        cnv_labels = {k: k for k in cnv_colors.keys()}

    def state_to_color(states: pd.Series) -> np.ndarray:
        return np.array([cnv_colors.get(s, "#666666") for s in states.astype(str)], dtype=object)

    # Draw facets
    for ci, chrom in enumerate(chroms):
        ch_mask = long["CHROM"].astype(str) == chrom
        se_chr = (
            segs_exclude[segs_exclude["CHROM"].astype(str) == chrom]
            if segs_exclude is not None else
            pd.DataFrame(columns=["seg_start", "seg_end"])
        )

        for ri, var in enumerate(variables):
            ax = axes[ri, ci]
            dat = long[ch_mask & (long["variable"] == var)]

            # Background shading for gap/acen
            if use_pos and not se_chr.empty:
                for _, row in se_chr.iterrows():
                    ax.axvspan(float(row["seg_start"]), float(row["seg_end"]), color="0.95", zorder=0)

            # Points: level-1 round, level-2 square
            has_level2 = dat["state_post"].astype(str).str.contains("_2")
            dat_lvl1 = dat[~has_level2]
            dat_lvl2 = dat[has_level2]
            c1 = state_to_color(dat_lvl1["state_post"])
            c2 = state_to_color(dat_lvl2["state_post"])
            
            # Only show y-axis info on the first axis of each row
            # Titles and labels
            if ri == 0:
                # add pad so the title sits a bit above the axes
                ax.set_title(f"chr{chrom}", fontsize=text_size, rotation=45)
            if ci == 0:
                ax.set_ylabel(var, fontsize=text_size)          # row label on the first column
                ax.tick_params(axis="y", labelleft=True)
            else:
                ax.set_ylabel("")                               # no y-axis title on other columns
                ax.tick_params(axis="y", labelleft=False)

            if not dat_lvl1.empty:
                ax.scatter(
                    dat_lvl1[marker].values, dat_lvl1["value"].values,
                    s=dot_size, c=c1, marker="o", alpha=dot_alpha,
                    rasterized=raster, linewidths=0, zorder=2
                )
            if not dat_lvl2.empty:
                ax.scatter(
                    dat_lvl2[marker].values, dat_lvl2["value"].values,
                    s=dot_size, c=c2, marker="s", alpha=1.0,
                    rasterized=raster, linewidths=0, zorder=3
                )

            # Axis ranges and guides
            if var == "logFC":
                ax.set_ylim(-exp_limit, exp_limit)
                ax.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8, zorder=1)
            else:  # pHF
                ax.set_ylim(-0.05, 1.05)

            # Segment overlays on logFC
            if var == "logFC" and not allele_only:
                if phi_mle and "phi_mle" in df.columns:
                    segs = (
                        df[["CHROM", "seg", "seg_start", "seg_start_index", "seg_end",
                            "seg_end_index", "phi_mle"]]
                        .drop_duplicates()
                        .copy()
                    )
                    segs = segs[segs["CHROM"].astype(str) == chrom]
                    segs = segs[np.log2(segs["phi_mle"]).values < exp_limit]
                    x1, x2 = ("seg_start", "seg_end") if use_pos else ("seg_start_index", "seg_end_index")
                    for _, row in segs.iterrows():
                        y = float(np.log2(row["phi_mle"]))
                        ax.hlines(y=y, xmin=float(row[x1]), xmax=float(row[x2]),
                                  color="darkred", linewidth=0.8, zorder=4)
                elif not phi_mle and "phi_mle_roll" in df.columns:
                    ch_df = df[df["CHROM"].astype(str) == chrom].copy()
                    ch_df = ch_df.assign(variable="logFC")
                    mask_ok = np.isfinite(np.log2(ch_df["phi_mle_roll"].astype(float)))
                    ch_df = ch_df[mask_ok]
                    if not ch_df.empty:
                        ax.plot(
                            ch_df[marker].values,
                            np.log2(ch_df["phi_mle_roll"].values),
                            color="darkred", linewidth=0.8, zorder=4
                        )
                    ax.axhline(0.0, color="0.4", linestyle="--", linewidth=0.8, zorder=1)

            # Theta rolling overlays on pHF
            if theta_roll and var == "pHF" and "theta_hat_roll" in df.columns:
                ch_df = df[df["CHROM"].astype(str) == chrom]
                if "snp_index" in ch_df.columns:
                    xvals = ch_df["snp_index"].values
                    thr = ch_df["theta_hat_roll"].values.astype(float)
                    # simple deterministic colors
                    base_state = str(ch_df["cnv_state_post"].astype(str).iloc[0]) if len(ch_df) else "neu"
                    down_color = cnv_colors.get(base_state + "_down", "#666666")
                    up_color   = cnv_colors.get(base_state + "_up", "#666666")
                    ax.plot(xvals, 0.5 - thr, color=down_color, linewidth=0.7, zorder=5)
                    ax.plot(xvals, 0.5 + thr, color=up_color, linewidth=0.7, zorder=5)

            ax.tick_params(axis="both", labelsize=max(8, text_size - 2))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

    return (fig if isinstance(fig, plt.Figure) else fig.figure), axes


def plot_bulks(
    bulks: pd.DataFrame,
    *,
    ncol: int = 1,
    title: bool = True,
    title_size: int = 8,
    panel_vspace: float = 0.25,   # space between sample rows
    panel_wspace: float = 0.0,   # space between sample columns
    **psbulk_kwargs: Any,
    ) -> plt.Figure:
    """
    Plot a grid of pseudobulk panels (one per sample) and return a single Figure.

    This is a small wrapper that:
    1) splits ``bulks`` by the column ``sample``,
    2) creates a grid of SubFigures,
    3) calls `plot_psbulk` once per sample into each SubFigure (with legends disabled),
    4) adds a single figure-level legend on the right.

    Parameters
    ----------
    bulks
        Long table with a ``sample`` column and all columns expected by
        `plot_psbulk`. Each sample will produce one panel.
        If ``sample`` is missing, a single sample named "1" is assumed.
    ncol
        Number of sample panels per row in the outer grid.
    title
        If True, put a title above each sample panel using the sample name and,
        when available, ``n_cells``.
    title_size
        Font size for per-panel titles.
    panel_vspace
        Vertical spacing between panel rows in the outer grid (``GridSpec.hspace`` units).
    panel_wspace
        Horizontal spacing between panel columns in the outer grid (``GridSpec.wspace`` units).
    **psbulk_kwargs
        Any additional keyword arguments forwarded to :func:`plot_psbulk`
        (for example: ``use_pos``, ``exp_limit``, ``theta_roll``, etc.). This
        wrapper forces ``legend=False`` for each subpanel to avoid duplicates.

    Returns
    -------
    fig
        The single figure that contains all sample panels and the shared legend.

    Examples
    --------
    >>> fig = plot_bulks(bulks_df, ncol=2, panel_vspace=0.15)
    >>> fig.savefig("all_samples.png", dpi=300)
    """
    df = bulks.copy()
    if "sample" not in df.columns:
        df["sample"] = "1"

    groups = [(k, g.copy()) for k, g in df.groupby("sample", observed=True, sort=False)]
    ns = len(groups)
    nrow = int(np.ceil(ns / ncol))

    # Pull the color/label mappings (or defaults) so we can build one legend later
    cnv_colors = psbulk_kwargs.get("cnv_colors") or {
        "neu": "#9e9e9e",
        "loh_up": "#4a90e2", "loh_down": "#357ABD",
        "del_up": "#d0021b", "del_down": "#a80012",
        "amp_up": "#7ed321", "amp_down": "#5fa015",
        "bamp": "#f5a623", "bdel": "#8b572a",
        "loh_2_up": "#4a90e2", "loh_2_down": "#357ABD",
        "del_2_up": "#d0021b", "del_2_down": "#a80012",
        "amp_2_up": "#7ed321", "amp_2_down": "#5fa015",
    }
    cnv_labels = psbulk_kwargs.get("cnv_labels") or {k: k for k in cnv_colors.keys()}

    fig = plt.figure(figsize=(18 * ncol, 5 * nrow))
    fig.subplots_adjust(left=0.05, right=0.85)
    gs_outer = fig.add_gridspec(nrows=nrow, ncols=ncol, hspace=panel_vspace, wspace=panel_wspace)
    subfigs = np.array([[fig.add_subfigure(gs_outer[r, c]) for c in range(ncol)] for r in range(nrow)])

    i = 0
    for r in range(nrow):
        for c in range(ncol):
            if i >= ns:
                subfigs[r, c].set_visible(False)
                continue

            sample, sub = groups[i]
            i += 1

            # Important: disable per-panel legend to avoid overlaps/duplicates
            kw = dict(psbulk_kwargs)
            kw["legend"] = False
            plot_psbulk(sub, parent=subfigs[r, c], **kw)
            
            if title:
                if "n_cells" in sub.columns:
                    vals = pd.unique(sub["n_cells"].dropna())
                    n_cells = vals[0] if len(vals) else None
                    ttl = f"{sample} (n={n_cells})" if n_cells is not None else f"{sample}"
                else:
                    ttl = f"{sample}"
                subfigs[r, c].suptitle(ttl, fontsize=title_size)
            
            # make subfigure background transparent
            subfigs[r, c].patch.set_alpha(0.0)

    # Optionally filter to only states present in bulks (keeps legend compact)
    present_states = set(df.get("state_post", pd.Series(dtype=str)).astype(str).unique())
    ordered = [s for s in cnv_colors if (not present_states) or (s in present_states)]
    if not ordered:  # fall back to all keys if we could not detect states
        ordered = list(cnv_colors.keys())

    handles = [
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor=cnv_colors.get(s, "#666666"),
               markeredgewidth=0, markersize=6,
               label=cnv_labels.get(s, s))
        for s in ordered
    ]
    # shape legend (level 1 vs level 2 markers)
    shape_items = [
        Line2D([0], [0], marker="o", color="k", label="level 1", linestyle="None",
               markerfacecolor="k", alpha=psbulk_kwargs.get("dot_alpha", 0.5), markersize=6),
        Line2D([0], [0], marker="s", color="k", label="level 2", linestyle="None",
               markerfacecolor="k", alpha=1.0, markersize=6),
    ]

    # Put it at the top-right of the overall figure; tweak as you like
    fig.legend(handles=handles + shape_items,
               loc="upper right",
               frameon=False,
               ncols=min(1, max(1, len(handles)//2 + 1)),
               fontsize=15,
               bbox_to_anchor=(1.0,1),
               )

    return fig



#### plot_exp_roll

@dataclass
class ExpRollPlotChromPanels:
    fig: plt.Figure
    ax_tree: Optional[plt.Axes]
    ax_chrom: List[plt.Axes]


def _normalize_chrom(x: object) -> str:
    s = str(x).strip()
    if s.lower().startswith("chr"):
        s = s[3:].strip()
    if s in {"M", "m", "Mt", "mt", "MT"}:
        return "MT"
    return s


def plot_exp_roll(
    gexp_roll_wide: ad.AnnData,
    hc: np.ndarray,
    k: int,
    gtf: pd.DataFrame,
    lim: float = 0.8,
    n_sample: int = 300,
    reverse: bool = True,
    plot_tree: bool = True,
    layer: str = "X_smooth",
    random_state: int = 0,
    hide_chrom_labels: Optional[Iterable[object]] = None,
    chrom_sizes: Optional[Dict[str, int]] = None,
    min_panel_width: float = 0.35,
    max_panel_width: float = 4.0,
    tree_width: float = 2.0,
    wspace: float = 0.08,
    sep_lw: float = 2.2,
    sep_alpha: float = 0.9,
    show_colorbar: bool = True,
    cbar_height: float = 0.035,
    cbar_gap: float = 0.020,
    cbar_width: float = 0.3,
    debug: bool = False,
    show: bool = False,
    close: bool = False,
    savepath: Optional[Union[str, Path]] = None,
    dpi: int = 200,
    ) -> ExpRollPlotChromPanels:
    """
    Plot a genome-ordered expression heatmap with one panel per chromosome.

    Genes are taken from `gexp_roll_wide.var_names`, joined to `gtf` (gene, CHROM, gene_start),
    and ordered by natsorted chromosome then `gene_start`. Cells are subsampled in the leaf
    order of `hc`; if `plot_tree=True` a Ward dendrogram is recomputed on the sampled matrix
    and its leaves define the heatmap row order. Chromosome panel widths are proportional to
    `chrom_sizes` (if provided) or to the genomic span of displayed genes on that chromosome.
    Dendrogram and heatmaps share a y-scale of `[0, 10*n_cells]` to match SciPy dendrogram
    leaf spacing.

    Parameters
    ----------
    gexp_roll_wide
        AnnData-like object with attributes:
        - `var_names`: gene identifiers (must match `gtf["gene"]`),
        - `obs_names`: cell identifiers,
        - `layers[layer]`: expression matrix of shape (n_cells, n_genes) (dense or sparse).
    hc
        SciPy linkage matrix for the full set of cells. Used only to obtain an initial
        dendrogram leaf order for subsampling (`leaves_list(hc)`).
    k
        Number of clusters used to compute horizontal separator positions via
        `scipy.cluster.hierarchy.fcluster` when `plot_tree=True`.
    gtf
        Gene annotation table with columns `gene`, `CHROM`, `gene_start`.
        Optional `gene_end` improves genomic-span estimation when `chrom_sizes` is not provided.
    lim
        Color limit; values are clipped to `[-lim, lim]` before plotting.
    n_sample
        Maximum number of cells to plot. Cells are sampled uniformly (without replacement)
        from the `hc` leaf order, then plotted in that order (or dendrogram order if recomputed).
    reverse
        If True, reverse the leaf order used for row ordering.
    plot_tree
        If True, recompute Ward linkage on the sampled matrix, plot the dendrogram, and reorder
        rows to match its leaves (keeps dendrogram/heatmaps aligned).
    layer
        Key in `gexp_roll_wide.layers` to plot.
    random_state
        Seed for the RNG used for cell subsampling.
    hide_chrom_labels
        Iterable of chromosome labels to hide after normalization (e.g. `18`, `"18"`, `"X"`, `"MT"`).
    chrom_sizes
        Optional mapping `{chrom -> length_bp}` to set chromosome panel width ratios. Keys may
        include a `"chr"` prefix; values are interpreted as base-pair lengths.
    min_panel_width, max_panel_width
        Clamp for chromosome width ratios (after normalization by the median) to keep the layout readable.
    tree_width
        Width ratio of the dendrogram column relative to chromosome panels.
    wspace
        Horizontal whitespace between panels (GridSpec `wspace`).
    sep_lw, sep_alpha
        Line width and alpha for vertical separators drawn between chromosome panels.
    show_colorbar
        If True, draw a centered horizontal colorbar above the panels.
    cbar_height
        Colorbar axis height in figure coordinates.
    cbar_gap
        Vertical gap between the colorbar and the top of the panel area (figure coordinates).
    cbar_width
        Fraction (0–1) of the main panel span used by the colorbar; the bar is centered.
    debug
        If True, print basic diagnostics (chromosome order, width ratios, number of plotted rows).

    show
        If True, call `plt.show()` before returning.
    close
        If True, call `plt.close(fig)` before returning (useful in pipelines to avoid rendering/memory).
    savepath
        If provided, save the figure to this path via `fig.savefig(...)` (parents created as needed).
    dpi
        DPI used for saving when `savepath` is provided.

    Returns
    -------
    ExpRollPlotChromPanels
        Container with:
        - `fig`: matplotlib Figure,
        - `ax_tree`: dendrogram axis (or None if `plot_tree=False`),
        - `ax_chrom`: list of chromosome heatmap axes (natsorted chromosome order).

    Raises
    ------
    ValueError
        If required `gtf` columns are missing, no gene overlap is found, or `var_names` contains duplicates.
    """
    req = {"gene", "CHROM", "gene_start"}
    if not req.issubset(gtf.columns):
        raise ValueError(f"gtf must contain columns: {sorted(req)}")

    hide_set: Set[str] = set()
    if hide_chrom_labels:
        hide_set = {_normalize_chrom(x) for x in hide_chrom_labels}

    # genes present in AnnData
    genes = list(map(str, list(getattr(gexp_roll_wide, "var_names"))))
    if len(set(genes)) != len(genes):
        raise ValueError("gexp_roll_wide.var_names contains duplicates; cannot map genes uniquely.")
    gene_pos = {g: i for i, g in enumerate(genes)}

    # filter & normalize gtf
    gtf2 = gtf.copy()
    gtf2["gene"] = gtf2["gene"].astype(str)
    gtf2["CHROM"] = gtf2["CHROM"].map(_normalize_chrom)
    gtf2["gene_start"] = pd.to_numeric(gtf2["gene_start"], errors="coerce")
    if "gene_end" in gtf2.columns:
        gtf2["gene_end"] = pd.to_numeric(gtf2["gene_end"], errors="coerce")
    gtf2 = gtf2.dropna(subset=["gene_start"]).copy()

    gtf2 = gtf2[gtf2["gene"].isin(genes)].copy()
    if gtf2.empty:
        raise ValueError("No overlap between gexp_roll_wide.var_names and gtf['gene'].")

    natkey = natsort_keygen()
    chrom_order0 = sorted(gtf2["CHROM"].unique().tolist(), key=natkey)
    chrom_rank = {c: i for i, c in enumerate(chrom_order0)}
    gtf2["_chrom_rank"] = gtf2["CHROM"].map(chrom_rank).astype(int)
    gtf2 = gtf2.sort_values(["_chrom_rank", "gene_start"], kind="mergesort")

    gtf2 = gtf2.drop_duplicates(subset=["gene"], keep="first").copy()
    chrom_order = sorted(gtf2["CHROM"].unique().tolist(), key=natkey)

    chrom_to_gene_idx: Dict[str, np.ndarray] = {}
    for c in chrom_order:
        g_chr = gtf2.loc[gtf2["CHROM"] == c, "gene"].tolist()
        if g_chr:
            chrom_to_gene_idx[c] = np.array([gene_pos[g] for g in g_chr], dtype=int)

    chrom_order = [c for c in chrom_order if c in chrom_to_gene_idx]
    if not chrom_order:
        raise ValueError("After filtering, no chromosomes have any genes to plot.")

    # sample cells using ORIGINAL hc leaf order
    cell_names = list(map(str, list(getattr(gexp_roll_wide, "obs_names"))))
    leaf_idx0 = leaves_list(hc)
    cell_order_all = [cell_names[i] for i in leaf_idx0]
    if reverse:
        cell_order_all = list(reversed(cell_order_all))

    rng = np.random.default_rng(random_state)
    n_take = min(int(n_sample), len(cell_order_all))
    if n_take <= 0:
        raise ValueError("n_sample must be >= 1")

    take = set(rng.choice(cell_order_all, size=n_take, replace=False).tolist())
    cell_order = [c for c in cell_order_all if c in take]

    cell_pos = {c: i for i, c in enumerate(cell_names)}
    row_idx = np.array([cell_pos[c] for c in cell_order], dtype=int)

    # extract sampled cell matrix (all genes)
    X = gexp_roll_wide.layers[layer]
    X_cell = X[row_idx, :]
    X_cell = X_cell.toarray() if hasattr(X_cell, "toarray") else np.asarray(X_cell)

    # recompute linkage on sampled cells & get leaf order (NO PLOT)
    if plot_tree:
        Z = linkage(pdist(X_cell, metric="euclidean"), method="ward")
        cl = fcluster(Z, t=k, criterion="maxclust")
        dd = dendrogram(Z, orientation="left", no_plot=True)
        leaves = dd["leaves"]
        if reverse:
            leaves = leaves[::-1]
        X_cell = X_cell[leaves, :]
        cl_plot = cl[leaves]
    else:
        Z = None
        cl_plot = None

    # width ratios
    chrom_sizes_norm = {_normalize_chrom(k): float(v) for k, v in (chrom_sizes or {}).items()}
    raw = []
    for c in chrom_order:
        if c in chrom_sizes_norm:
            raw.append(chrom_sizes_norm[c])
        else:
            sub = gtf2.loc[gtf2["CHROM"] == c]
            s0 = float(sub["gene_start"].min())
            if "gene_end" in sub.columns and sub["gene_end"].notna().any():
                e1 = float(sub["gene_end"].max())
            else:
                e1 = float(sub["gene_start"].max())
            raw.append(max(1.0, e1 - s0))

    raw = np.asarray(raw, dtype=float)
    scale = np.median(raw[raw > 0]) if np.any(raw > 0) else 1.0
    ratios = (raw / scale).tolist()
    ratios = [float(np.clip(r, min_panel_width, max_panel_width)) for r in ratios]

    # cluster separators
    if cl_plot is not None:
        uniq = list(dict.fromkeys(cl_plot.tolist()))
        rank = {u: i for i, u in enumerate(uniq)}
        cl_rank = np.array([len(uniq) - rank[c] for c in cl_plot], dtype=int)
        cluster_change = np.flatnonzero(cl_rank[1:] != cl_rank[:-1]) + 1
    else:
        cluster_change = np.array([], dtype=int)

    vmin, vmax = -lim, lim
    n_rows = X_cell.shape[0]

    if debug:
        print("chrom_order:", chrom_order)
        print("ratios:", ratios)
        print("n_rows:", n_rows)

    # FIGURE / GRID
    fig = plt.figure(figsize=(14, 6), constrained_layout=False)

    top = 0.96 - cbar_height - cbar_gap if show_colorbar else 0.96
    bottom = 0.20
    left = 0.08
    right = 0.99

    if plot_tree:
        gs = gridspec.GridSpec(
            1, 1 + len(chrom_order),
            width_ratios=[tree_width] + ratios,
            figure=fig,
            wspace=wspace,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        ax_tree = fig.add_subplot(gs[0, 0])
        ax_chrom = [fig.add_subplot(gs[0, j + 1]) for j in range(len(chrom_order))]
    else:
        ax_tree = None
        gs = gridspec.GridSpec(
            1, len(chrom_order),
            width_ratios=ratios,
            figure=fig,
            wspace=wspace,
            left=left,
            right=right,
            top=top,
            bottom=bottom,
        )
        ax_chrom = [fig.add_subplot(gs[0, j]) for j in range(len(chrom_order))]

    # DENDROGRAM (PLOT) aligned to heatmap y-extent
    if ax_tree is not None:
        dendrogram(
            Z,
            orientation="left",
            no_labels=True,
            color_threshold=0,
            above_threshold_color="black",
            ax=ax_tree,
        )
        ax_tree.set_ylim(0, 10 * n_rows)
        ax_tree.invert_yaxis()
        ax_tree.axis("off")

    # HEATMAP PANELS
    im = None
    for j, c in enumerate(chrom_order):
        ax = ax_chrom[j]
        idx = chrom_to_gene_idx[c]
        C = np.clip(X_cell[:, idx], vmin, vmax)

        im = ax.imshow(
            C,
            aspect="auto",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
            cmap="bwr",
            origin="upper",
            extent=(-0.5, C.shape[1] - 0.5, 0, 10 * n_rows),
        )

        ax.set_ylim(0, 10 * n_rows)
        ax.invert_yaxis()

        for r in cluster_change:
            ax.axhline(10 * r, linewidth=0.6, color="black", alpha=0.35)

        ax.set_yticks([])
        ax.set_xticks([])

        lab = "" if c in hide_set else c
        ax.set_xlabel(lab, fontsize=9, labelpad=6)

        for spine in ax.spines.values():
            spine.set_visible(False)

        if j > 0:
            ax.axvline(-0.5, linewidth=sep_lw, color="black", alpha=sep_alpha)

    # COLORBAR ON TOP (centered, narrow)
    if show_colorbar and im is not None:
        span = right - left
        cw = float(np.clip(cbar_width, 0.05, 1.0)) * span
        cl = left + 0.5 * (span - cw)

        cax_bottom = top + cbar_gap * 0.5
        cax = fig.add_axes([cl, cax_bottom, cw, cbar_height])

        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
        cbar.set_label("Expression magnitude", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")

    fig.supylabel("Cells (dendrogram order)", x=0.05)
    fig.supxlabel("Genomic position (per-chromosome panels; widths proportional)", y=0.13)

    out = ExpRollPlotChromPanels(fig=fig, ax_tree=ax_tree, ax_chrom=ax_chrom)

    # pipeline behavior
    if savepath is not None:
        savepath = Path(savepath)
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()

    if close:
        plt.close(fig)

    return out


