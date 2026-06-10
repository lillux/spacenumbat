#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 30 19:30:40 2025

@author: lillux
"""
from __future__ import annotations

from dataclasses import dataclass

from typing import Optional, Dict, Any, Iterable, Set, List, Union, Mapping, Sequence
from pathlib import Path

import re

import numpy as np
import pandas as pd

import anndata as ad

import matplotlib.pyplot as plt
from matplotlib.pyplot import Line2D
from matplotlib import gridspec
from matplotlib.figure import Figure, SubFigure


from natsort import natsort_keygen

from scipy.cluster.hierarchy import dendrogram, leaves_list, fcluster, linkage
from scipy.spatial.distance import pdist




_PLOT_STATE = "__plot_state__"
_PLOT_CNV_STATE = "__plot_cnv_state__"
_PLOT_PHF = "__plot_phf__"

DEFAULT_LEGEND_BREAKS = (
    "neu",
    "loh_up",
    "loh_down",
    "del_up",
    "del_down",
    "amp_up",
    "amp_down",
    "bamp",
    "bdel",
    )


@dataclass(frozen=True)
class PsbulkColumns:
    """Column names used by :func:`plot_psbulk` and :func:`plot_bulks`."""

    chrom: str = "CHROM"
    pos: str = "POS"
    snp_index: str = "snp_index"

    state_post: str = "state_post"
    state_fallback: str = "state"
    cnv_state_post: str | None = "cnv_state_post"
    cnv_state_fallback: str | None = "cnv_state"

    llr: str = "LLR"
    loh: str = "loh"
    p_up: str = "p_up"

    logfc: str = "logFC"
    mu: str = "mu"
    pbaf: str = "pBAF"
    depth: str = "DP"

    theta_roll: str = "theta_hat_roll"
    phi_mle: str = "phi_mle"
    phi_roll: str = "phi_mle_roll"

    seg_start: str = "seg_start"
    seg_end: str = "seg_end"
    seg_start_index: str = "seg_start_index"
    seg_end_index: str = "seg_end_index"

    sample: str = "sample"
    n_cells: str = "n_cells"

    region_chrom: str = "CHROM"
    region_start: str = "start"
    region_end: str = "end"


DEFAULT_COLUMNS = PsbulkColumns()


def default_cnv_colors() -> dict[str, str]:
    """Return the default state-to-color mapping."""
    return {
        "neu": "grey",
        "loh": "green",
        "del": "cornflowerblue",
        "amp": "red",
        "loh_up": "lime",
        "loh_down": "limegreen",
        "del_up": "navy",
        "del_down": "royalblue",
        "amp_up": "crimson",
        "amp_down": "deeppink",
        "loh_2_up": "green",
        "loh_2_down": "green",
        "del_2_up": "blue",
        "del_2_down": "blue",
        "amp_2_up": "red",
        "amp_2_down": "red",
        "bamp": "blueviolet",
        "bdel": "mediumblue",
    }


def _with_level1_aliases(colors: Mapping[str, str]) -> dict[str, str]:
    """Add ``*_1_up/down`` aliases without overriding user-supplied colors."""
    
    out = dict(colors)
    for state in ("loh", "del", "amp"):
        for direction in ("up", "down"):
            canonical = f"{state}_{direction}"
            if canonical in out:
                out.setdefault(f"{state}_1_{direction}", out[canonical])
    return out


def _require_columns(df: pd.DataFrame, columns: Sequence[str], context: str) -> None:
    
    missing = sorted(set(columns).difference(df.columns))
    if missing:
        raise KeyError(f"{context} requires columns: {missing}")


def _resolve_column(
    df: pd.DataFrame,
    preferred: str,
    fallback: str | None,
    description: str,
    ) -> str:
    
    if preferred in df.columns:
        return preferred
    if fallback is not None and fallback in df.columns:
        return fallback
    choices = [preferred] + ([fallback] if fallback is not None else [])
    raise KeyError(f"Missing {description} column. Tried: {choices}")


def _prepare_plot_states(
    df: pd.DataFrame,
    columns: PsbulkColumns,
    min_llr: float,
    transform_states: bool,
    neutral_state: str,
    loh_state: str,
    p_up_states: Sequence[str],
    p_up_threshold: float,
    level2_pattern: str | None,
    ) -> pd.DataFrame:
    
    """Prepare internal plotting states."""
    out = df.copy()

    state_source = _resolve_column(
        out,
        columns.state_post,
        columns.state_fallback,
        "state",
    )
    out[_PLOT_STATE] = out[state_source].astype("string")

    if columns.cnv_state_post is not None:
        cnv_source = _resolve_column(
            out,
            columns.cnv_state_post,
            columns.cnv_state_fallback,
            "CNV state",
        )
        out[_PLOT_CNV_STATE] = out[cnv_source].astype("string")

    if not transform_states:
        return out

    if min_llr != 0 and columns.llr in out.columns:
        llr = pd.to_numeric(out[columns.llr], errors="coerce").fillna(0.0)
        neutral = llr < min_llr
        out.loc[neutral, _PLOT_STATE] = neutral_state
        if _PLOT_CNV_STATE in out.columns:
            out.loc[neutral, _PLOT_CNV_STATE] = neutral_state

    if columns.loh in out.columns:
        loh = out[columns.loh].astype("boolean")
        out.loc[loh.eq(True), _PLOT_STATE] = loh_state
        out.loc[loh.isna(), _PLOT_STATE] = pd.NA

    if columns.p_up not in out.columns or _PLOT_CNV_STATE not in out.columns:
        return out

    state = out[_PLOT_STATE].astype("string")
    is_level2 = (
        pd.Series(False, index=out.index)
        if level2_pattern is None
        else state.str.contains(level2_pattern, regex=False, na=False)
    )

    theta_level = pd.Series("1", index=out.index, dtype="string")
    theta_level.loc[is_level2] = "2"
    theta_level.loc[state.isna()] = pd.NA

    p_up = pd.to_numeric(out[columns.p_up], errors="coerce")
    direction = pd.Series(pd.NA, index=out.index, dtype="string")
    direction.loc[p_up > p_up_threshold] = "up"
    direction.loc[p_up <= p_up_threshold] = "down"

    base_state = out[_PLOT_CNV_STATE].astype("string")
    target = base_state.isin(tuple(map(str, p_up_states)))
    refined = base_state + "_" + theta_level + "_" + direction
    out.loc[target, _PLOT_STATE] = refined.loc[target]

    return out


def _ordered_present_states(
    states: pd.Series,
    colors: Mapping[str, str],
    ) -> list[str]:
    present = list(pd.unique(states.dropna().astype(str)))
    mapped = [state for state in colors if state in present]
    return mapped + [state for state in present if state not in colors]


def _legend_states(
    states: pd.Series,
    colors: Mapping[str, str],
    legend_breaks: Sequence[str] | None,
    present_only: bool,
    ) -> list[str]:
    if legend_breaks is None:
        return _ordered_present_states(states, colors)

    ordered = [state for state in legend_breaks if state in colors]
    if not present_only:
        return ordered

    present = set(states.dropna().astype(str))
    canonical_present = present | {state.replace("_1_", "_") for state in present}
    return [state for state in ordered if state in canonical_present]


def _add_legend(
    fig: Figure,
    states: pd.Series,
    colors: Mapping[str, str],
    labels: Mapping[str, str],
    legend_breaks: Sequence[str] | None,
    present_only: bool,
    fontsize: float,
    legend_kwargs: Mapping[str, Any] | None,
    ) -> None:
    ordered = _legend_states(states, colors, legend_breaks, present_only)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            color="none",
            markerfacecolor=colors[state],
            markeredgewidth=0,
            markersize=6,
            label=labels.get(state, state),
        )
        for state in ordered
    ]

    kwargs: dict[str, Any] = {
        "loc": "upper right",
        "bbox_to_anchor": (1.0, 1.0),
        "frameon": False,
        "fontsize": fontsize,
        "title": "CNV state",
    }
    if legend_kwargs is not None:
        kwargs.update(legend_kwargs)
    fig.legend(handles=handles, **kwargs)


def _natural_key(value: str) -> list[object]:
    """Return a dependency-free natural-sort key."""
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def _chromosomes(
    chrom: pd.Series,
    order: Sequence[str] | None,
    ) -> list[str]:
    
    present = list(pd.unique(chrom.dropna().astype(str)))
    if order is None:
        return sorted(present, key=_natural_key)

    requested = [str(value) for value in order]
    requested_set = set(requested)
    remaining = [value for value in present if value not in requested_set]
    return [value for value in requested if value in present] + sorted(remaining, key=_natural_key)


def _span(values: pd.Series) -> float:
    
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size < 2:
        return 1.0
    return max(float(numeric.max() - numeric.min()), 1.0)


def _excluded_regions(
    gaps: pd.DataFrame | None,
    acen: pd.DataFrame | None,
    columns: PsbulkColumns,
    ) -> pd.DataFrame | None:
    
    frames = [frame for frame in (gaps, acen) if frame is not None and not frame.empty]
    if not frames:
        return None

    required = [columns.region_chrom, columns.region_start, columns.region_end]
    for frame in frames:
        _require_columns(frame, required, "Excluded-region shading")

    regions = pd.concat(frames, ignore_index=True)
    return regions.rename(
        columns={
            columns.region_chrom: "chrom",
            columns.region_start: "start",
            columns.region_end: "end",
        }
    )[["chrom", "start", "end"]]


def _plot_state_colored_line(
    ax: plt.Axes,
    x: pd.Series,
    y: pd.Series,
    states: pd.Series,
    colors: Mapping[str, str],
    unknown_color: str,
    linewidth: float,
    zorder: float,
    ) -> None:
    
    data = pd.DataFrame(
        {
            "x": pd.to_numeric(x, errors="coerce"),
            "y": pd.to_numeric(y, errors="coerce"),
            "state": states.astype("string"),
        }
    ).sort_values("x", kind="stable")

    valid = np.isfinite(data["x"]) & np.isfinite(data["y"]) & data["state"].notna()
    state_changed = data["state"].ne(data["state"].shift()).fillna(True)
    new_run = (~valid) | (~valid.shift(fill_value=False)) | state_changed
    data["run"] = new_run.cumsum()

    for _, run in data.loc[valid].groupby("run", sort=False):
        state = str(run["state"].iloc[0])
        ax.plot(
            run["x"],
            run["y"],
            color=colors.get(state, unknown_color),
            linewidth=linewidth,
            zorder=zorder,
        )


def plot_psbulk(
    bulk: pd.DataFrame,
    columns: PsbulkColumns = DEFAULT_COLUMNS,
    use_pos: bool = True,
    allele_only: bool = False,
    min_LLR: float = 5.0,
    min_depth: int = 8,
    exp_limit: float = 2.0,
    phi_mle: bool = True,
    theta_roll: bool = False,
    transform_states: bool = True,
    neutral_state: str = "neu",
    loh_state: str = "loh",
    p_up_states: Sequence[str] = ("amp", "loh", "del"),
    p_up_threshold: float = 0.5,
    level2_pattern: str | None = "_2",
    cnv_colors: Mapping[str, str] | None = None,
    cnv_labels: Mapping[str, str] | None = None,
    legend: bool = True,
    legend_breaks: Sequence[str] | None = DEFAULT_LEGEND_BREAKS,
    legend_present_only: bool = False,
    legend_kwargs: Mapping[str, Any] | None = None,
    unknown_color: str = "#666666",
    dot_size: float = 8.0,
    dot_alpha: float = 0.5,
    level1_marker: str = "o",
    level2_marker: str = "s",
    mu_scale: float = 1.0,
    exclude_gap: bool = True,
    gaps: pd.DataFrame | None = None,
    acen: pd.DataFrame | None = None,
    chrom_order: Sequence[str] | None = None,
    text_size: float = 10,
    raster: bool = False,
    show_x_ticks: bool = False,
    figsize: tuple[float, float] | None = None,
    chrom_width: float = 1.5,
    track_height: float = 1.2,
    parent: Figure | SubFigure | None = None,
    close: bool = True,
    ) -> tuple[Figure, np.ndarray]:
    """Plot one pseudobulk HMM profile.

    ``columns`` controls all input column names. A closed returned figure remains usable and saveable.
    """
    if bulk.empty:
        raise ValueError("bulk must contain at least one row")
    if min_depth < 0:
        raise ValueError("min_depth must be non-negative")
    if exp_limit <= 0:
        raise ValueError("exp_limit must be positive")
    if not 0 <= dot_alpha <= 1:
        raise ValueError("dot_alpha must be between 0 and 1")

    marker_col = columns.pos if use_pos else columns.snp_index
    required = [columns.chrom, marker_col, columns.pbaf, columns.depth]
    if not allele_only:
        required.extend([columns.logfc, columns.mu])
    _require_columns(bulk, required, "Pseudobulk plotting")

    df = _prepare_plot_states(
        bulk,
        columns=columns,
        min_llr=min_LLR,
        transform_states=transform_states,
        neutral_state=neutral_state,
        loh_state=loh_state,
        p_up_states=p_up_states,
        p_up_threshold=p_up_threshold,
        level2_pattern=level2_pattern,
    )

    if not allele_only:
        logfc = pd.to_numeric(df[columns.logfc], errors="coerce")
        mu = pd.to_numeric(df[columns.mu], errors="coerce")
        df[columns.logfc] = (logfc - mu * mu_scale).where(
            lambda values: values.between(-exp_limit, exp_limit)
        )

    pbaf = pd.to_numeric(df[columns.pbaf], errors="coerce")
    depth = pd.to_numeric(df[columns.depth], errors="coerce")
    df[_PLOT_PHF] = pbaf.where(depth >= min_depth)

    tracks = (
        [("pHF", _PLOT_PHF)]
        if allele_only
        else [("logFC", columns.logfc), ("pHF", _PLOT_PHF)]
    )
    chroms = _chromosomes(df[columns.chrom], chrom_order)
    if not chroms:
        raise ValueError("No non-missing chromosome values were found")

    width_ratios = [
        _span(df.loc[df[columns.chrom].astype(str).eq(chrom), marker_col])
        for chrom in chroms
    ]
    nrows, ncols = len(tracks), len(chroms)

    if parent is None:
        if figsize is None:
            figsize = (chrom_width * ncols, track_height * nrows)
        container: Figure | SubFigure = plt.figure(figsize=figsize)
    else:
        container = parent

    gs = container.add_gridspec(
        nrows=nrows,
        ncols=ncols,
        width_ratios=width_ratios,
        wspace=0.08,
        hspace=0.4,
    )
    axes = np.array(
        [
            [container.add_subplot(gs[row, col]) for col in range(ncols)]
            for row in range(nrows)
        ],
        dtype=object,
    )

    colors = _with_level1_aliases(
        default_cnv_colors() if cnv_colors is None else cnv_colors
    )
    labels = {state: state for state in colors} if cnv_labels is None else dict(cnv_labels)
    regions = _excluded_regions(gaps, acen, columns) if exclude_gap and use_pos else None

    chrom_as_string = df[columns.chrom].astype(str)
    for col_index, chrom in enumerate(chroms):
        chrom_data = df.loc[chrom_as_string.eq(chrom)]
        x = pd.to_numeric(chrom_data[marker_col], errors="coerce")

        chrom_regions = None
        if regions is not None:
            chrom_regions = regions.loc[regions["chrom"].astype(str).eq(chrom)]

        states = chrom_data[_PLOT_STATE].astype("string")
        level2 = (
            pd.Series(False, index=chrom_data.index)
            if level2_pattern is None
            else states.str.contains(level2_pattern, regex=False, na=False)
        )

        for row_index, (track_label, value_col) in enumerate(tracks):
            ax = axes[row_index, col_index]
            y = pd.to_numeric(chrom_data[value_col], errors="coerce")
            valid = np.isfinite(x) & np.isfinite(y) & states.notna()

            if chrom_regions is not None:
                for region in chrom_regions.itertuples(index=False):
                    ax.axvspan(float(region.start), float(region.end), color="0.95", zorder=0)

            for mask, marker, alpha, zorder in (
                (valid & ~level2, level1_marker, dot_alpha, 2),
                (valid & level2, level2_marker, 1.0, 3),
            ):
                if mask.any():
                    ax.scatter(
                        x.loc[mask],
                        y.loc[mask],
                        s=dot_size,
                        c=[colors.get(str(state), unknown_color) for state in states.loc[mask]],
                        marker=marker,
                        alpha=alpha,
                        rasterized=raster,
                        linewidths=0,
                        zorder=zorder,
                    )

            if row_index == 0:
                ax.set_title(f"chr{chrom}", fontsize=text_size, rotation=45)
            if col_index == 0:
                ax.set_ylabel(track_label, fontsize=text_size)
            else:
                ax.tick_params(axis="y", labelleft=False)

            if not show_x_ticks:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)

            if track_label == "logFC":
                ax.set_ylim(-exp_limit, exp_limit)
                ax.axhline(0, color="0.4", linestyle="--", linewidth=0.8, zorder=1)
            else:
                ax.set_ylim(-0.05, 1.05)

            ax.tick_params(axis="both", labelsize=max(8, text_size - 2))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        if not allele_only:
            expression_ax = axes[0, col_index]
            if phi_mle:
                start_col = columns.seg_start if use_pos else columns.seg_start_index
                end_col = columns.seg_end if use_pos else columns.seg_end_index
                _require_columns(
                    df,
                    [columns.chrom, start_col, end_col, columns.phi_mle],
                    "Segment expression overlay",
                )
                segments = chrom_data[[start_col, end_col, columns.phi_mle]].drop_duplicates()
                phi = pd.to_numeric(segments[columns.phi_mle], errors="coerce")
                segment_y = np.log2(phi.where(phi > 0))
                keep = segment_y.between(-exp_limit, exp_limit)
                for index in segments.index[keep]:
                    expression_ax.hlines(
                        y=float(segment_y.loc[index]),
                        xmin=float(segments.loc[index, start_col]),
                        xmax=float(segments.loc[index, end_col]),
                        color="darkred",
                        linewidth=0.8,
                        zorder=4,
                    )
            else:
                _require_columns(df, [columns.phi_roll], "Rolling expression overlay")
                phi_roll = pd.to_numeric(chrom_data[columns.phi_roll], errors="coerce")
                rolling_y = np.log2(phi_roll.where(phi_roll > 0))
                keep = rolling_y.between(-exp_limit, exp_limit)
                expression_ax.plot(
                    x.loc[keep],
                    rolling_y.loc[keep],
                    color="darkred",
                    linewidth=0.8,
                    zorder=4,
                )

        if theta_roll:
            _require_columns(df, [columns.theta_roll], "Theta rolling overlay")
            theta = pd.to_numeric(chrom_data[columns.theta_roll], errors="coerce")
            base_states = (
                chrom_data[_PLOT_CNV_STATE]
                if _PLOT_CNV_STATE in chrom_data.columns
                else chrom_data[_PLOT_STATE]
            ).astype("string")
            allele_ax = axes[-1, col_index]
            for suffix, y in (
                ("down", 0.5 - theta),
                ("up", 0.5 + theta),
            ):
                _plot_state_colored_line(
                    allele_ax,
                    x=x,
                    y=y,
                    states=base_states + f"_{suffix}",
                    colors=colors,
                    unknown_color=unknown_color,
                    linewidth=0.7,
                    zorder=5,
                )

    output_fig = container if isinstance(container, Figure) else container.figure
    if legend and parent is None:
        _add_legend(
            output_fig,
            states=df[_PLOT_STATE],
            colors=colors,
            labels=labels,
            legend_breaks=legend_breaks,
            present_only=legend_present_only,
            fontsize=text_size,
            legend_kwargs=legend_kwargs,
        )

    if close and parent is None:
        plt.close(output_fig)

    return output_fig, axes


def plot_bulks(
    bulks: pd.DataFrame,
    columns: PsbulkColumns = DEFAULT_COLUMNS,
    ncol: int = 1,
    title: bool = True,
    title_size: float = 8,
    panel_vspace: float = 0.25,
    panel_wspace: float = 0.0,
    panel_size: tuple[float, float] = (18.0, 5.0),
    figsize: tuple[float, float] | None = None,
    legend: bool = True,
    legend_fontsize: float = 15,
    legend_kwargs: Mapping[str, Any] | None = None,
    close: bool = True,
    **plot_kwargs: Any,
    ) -> Figure:
    """Plot one pseudobulk panel per sample with a shared legend."""
    if bulks.empty:
        raise ValueError("bulks must contain at least one row")
    if ncol < 1:
        raise ValueError("ncol must be at least 1")

    df = bulks.copy()
    if columns.sample not in df.columns:
        df[columns.sample] = "1"

    groups = list(df.groupby(columns.sample, observed=True, sort=False))
    nrow = int(np.ceil(len(groups) / ncol))
    if figsize is None:
        figsize = (panel_size[0] * ncol, panel_size[1] * nrow)

    colors = _with_level1_aliases(
        default_cnv_colors()
        if plot_kwargs.get("cnv_colors") is None
        else plot_kwargs["cnv_colors"]
    )
    labels = (
        {state: state for state in colors}
        if plot_kwargs.get("cnv_labels") is None
        else dict(plot_kwargs["cnv_labels"])
    )

    fig = plt.figure(figsize=figsize)
    outer = fig.add_gridspec(
        nrows=nrow,
        ncols=ncol,
        hspace=panel_vspace,
        wspace=panel_wspace,
    )

    for index, (sample, sample_data) in enumerate(groups):
        row, col = divmod(index, ncol)
        subfig = fig.add_subfigure(outer[row, col])

        child_kwargs = dict(plot_kwargs)
        child_kwargs.update(
            {
                "columns": columns,
                "legend": False,
                "parent": subfig,
                "close": False,
            }
        )
        plot_psbulk(sample_data, **child_kwargs)

        if title:
            title_text = str(sample)
            if columns.n_cells in sample_data.columns:
                values = pd.unique(sample_data[columns.n_cells].dropna())
                if len(values):
                    title_text = f"{sample} (n={values[0]})"
            subfig.suptitle(title_text, fontsize=title_size)
        subfig.patch.set_alpha(0)

    for index in range(len(groups), nrow * ncol):
        row, col = divmod(index, ncol)
        empty = fig.add_subfigure(outer[row, col])
        empty.set_visible(False)

    if legend:
        prepared = _prepare_plot_states(
            df,
            columns=columns,
            min_llr=plot_kwargs.get("min_LLR", 5.0),
            transform_states=plot_kwargs.get("transform_states", True),
            neutral_state=plot_kwargs.get("neutral_state", "neu"),
            loh_state=plot_kwargs.get("loh_state", "loh"),
            p_up_states=plot_kwargs.get("p_up_states", ("amp", "loh", "del")),
            p_up_threshold=plot_kwargs.get("p_up_threshold", 0.5),
            level2_pattern=plot_kwargs.get("level2_pattern", "_2"),
        )
        _add_legend(
            fig,
            states=prepared[_PLOT_STATE],
            colors=colors,
            labels=labels,
            legend_breaks=plot_kwargs.get("legend_breaks", DEFAULT_LEGEND_BREAKS),
            present_only=plot_kwargs.get("legend_present_only", False),
            fontsize=legend_fontsize,
            legend_kwargs=legend_kwargs,
        )

    if close:
        plt.close(fig)

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


