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


def plot_bulks_circos(
    bulks: pd.DataFrame,
    columns: PsbulkColumns = DEFAULT_COLUMNS,
    *,
    track: str = "logFC",
    sample_order: Sequence | None = None,
    chrom_order: Sequence | None = None,
    chrom_sizes: Mapping | pd.Series | None = None,
    cytobands: bool = False,
    use_pos: bool = True,
    min_LLR: float = 5.0,
    min_depth: int = 8,
    exp_limit: float = 2.0,
    mu_scale: float = 1.0,
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
    unknown_color: str = "#666666",
    dot_size: float = 2.0,
    dot_alpha: float = 0.6,
    raster: bool = True,
    exclude_gap: bool = True,
    gaps: pd.DataFrame | None = None,
    acen: pd.DataFrame | None = None,
    r_lim: tuple[float, float] = (20.0, 88.0),
    track_gap: float = 1.5,
    chrom_gap: float = 1.0,
    label_gap: float = 36.0,
    chrom_label_r: float = 106.0,
    show_x_ticks: bool = False,
    tick_interval: float | None = None,
    text_size: float = 8.0,
    show_center_text: bool = True,
    title: str | None = None,
    figsize: tuple[float, float] = (10.0, 10.0),
    dpi: int = 100,
    legend: bool = True,
    legend_fontsize: float = 8.0,
    legend_kwargs: Mapping | None = None,
    savepath: str | Path | None = None,
    close: bool = True,
) -> Figure:
    """Plot clone pseudobulks as aligned concentric tracks using pycirclize.

    Accept the concatenated pseudobulk table used by ``plot_bulks`` and draw
    one ring, or one pair of rings, per clone. Chromosomes occupy sectors
    shared by all clones. Signal values increase from the inner edge of each
    ring to its outer edge, using the same value scale for every clone.

    Parameters
    ----------
    bulks : pandas.DataFrame
        Concatenated clone pseudobulk observations. With default ``columns``,
        the following input columns are used:

        * Always required: ``CHROM`` and either ``POS`` or ``snp_index``,
          selected by ``use_pos``.
        * State source: ``state_post``, falling back to ``state``. A CNV state
          source, ``cnv_state_post`` falling back to ``cnv_state``, is also
          required unless ``columns.cnv_state_post`` is None. Fallbacks apply
          when a column is absent, rather than to individual missing values.
        * Expression tracks: ``logFC`` and ``mu``. If ``phi_mle=True``, also
          require ``phi_mle`` and ``seg_start``/``seg_end`` in position mode,
          or ``seg_start_index``/``seg_end_index`` in SNP-index mode. If
          ``phi_mle=False``, require ``phi_mle_roll`` instead.
        * BAF tracks: ``pBAF`` and ``DP``; also require ``theta_hat_roll``
          when ``theta_roll=True``.
        * Optional: ``sample`` for clone grouping, ``n_cells`` for clone
          labels, and ``LLR``, ``loh``, and ``p_up`` for state preparation.

        Rows with missing chromosomes are discarded. If ``sample`` is absent,
        all retained rows form a single clone labeled ``"1"``. If present,
        sample identifiers must not be missing. The input is not modified.
    columns : spacenumbat.plot.PsbulkColumns, default DEFAULT_COLUMNS
        Mapping of logical fields to input column names, shared with
        ``plot_bulks``. It also maps the chromosome, start, and end fields
        used by ``gaps`` and ``acen``.
    track : {"logFC", "pBAF", "both"}, default "logFC"
        Signal to display. ``"logFC"`` draws corrected expression,
        ``logFC - mu * mu_scale``. ``"pBAF"`` draws phased B-allele frequency.
        ``"both"`` assigns each clone a band containing an outer expression
        ring and an inner BAF ring.
    sample_order : sequence, optional
        Distinct clone identifiers to select and arrange from outermost to
        innermost. Identifiers must match the input values and their types.
        By default, include all clones in their order of first appearance.
    chrom_order : sequence, optional
        Distinct chromosomes to select and arrange clockwise. Each requested
        chromosome must occur in the selected clones. By default, include
        all observed chromosomes in natural order, such as 1, 2, ..., 10.
        A leading ``chr`` prefix is removed before matching names.
    chrom_sizes : mapping or pandas.Series, optional
        Physical chromosome lengths in the same coordinate units as ``POS``.
        Keys are chromosome names, for example ``{"chr1": 248956422}``.
        A Series must be indexed by chromosome. A nonempty mapping must cover
        every selected chromosome; additional chromosomes do not add sectors.
        Lengths must be positive and finite, and contain all retained marker
        and available segment coordinates. Only valid with ``use_pos=True``.
        If omitted, sector endpoints are inferred from the observed data,
        except when ``cytobands=True`` uses full hg38 chromosome lengths.
        With cytobands enabled, supplied lengths must match hg38 exactly.
    cytobands : bool, default False
        Draw hg38 cytobands on the outer chromosome track using pycirclize's
        example dataset and standard stain colors. This also uses the full
        hg38 chromosome lengths so bands and pseudobulk coordinates align.
        Only valid with ``use_pos=True`` and positions in base pairs.
        Chromosome names such as ``1`` and ``chr1`` are both accepted; each
        selected chromosome must exist in the hg38 dataset. The first use
        downloads the dataset; pycirclize caches it for subsequent calls.
    use_pos : bool, default True
        Use physical ``POS`` coordinates and position-based segment endpoints.
        If False, use ``snp_index`` and index-based segment endpoints instead.
        Coordinate ranges are shared across clones within each chromosome.
    min_LLR : float, default 5.0
        With state transformation enabled, assign rows with ``LLR < min_LLR``
        to ``neutral_state`` before subsequent LOH and direction handling.
        Missing or nonnumeric LLR values are treated as zero by the upstream
        helper. Setting this to zero disables the LLR step; an absent LLR
        column also skips it. This changes plotting states, not row inclusion.
    min_depth : int, default 8
        Minimum ``DP`` for displaying a BAF point, inclusive. Must be
        nonnegative. Expression points and theta overlays are not filtered
        by this threshold.
    exp_limit : float, default 2.0
        Positive finite magnitude defining the expression range
        ``[-exp_limit, exp_limit]``. Finite expression points outside this
        range are clipped to the nearest boundary and remain visible.
        Out-of-range fitted overlays are omitted.
    mu_scale : float, default 1.0
        Multiplier for the expression baseline correction
        ``logFC - mu * mu_scale``. It does not rescale fitted phi overlays.
    phi_mle : bool, default True
        On expression tracks, draw segment-level horizontal arcs at
        ``log2(phi_mle)``. If False, draw the rolling-expression curve at
        ``log2(phi_mle_roll)``. Nonpositive phi values are omitted. Ignored
        when expression tracks are not requested.
    theta_roll : bool, default False
        On BAF tracks, overlay curves at ``0.5 - theta_hat_roll`` and
        ``0.5 + theta_hat_roll``, colored by CNV state and allele direction.
        Ignored when BAF tracks are not requested.
    transform_states : bool, default True
        Reuse the upstream state-preparation helper to apply LLR masking,
        LOH overrides, and allele-direction refinement, in that order.
        If False, retain the selected source state labels for plotting.
    neutral_state : str, default "neu"
        Label used by LLR neutralization. Its default point color is white
        with a black outline; override the fill through ``cnv_colors``.
    loh_state : str, default "loh"
        Plotting state assigned to rows where the optional ``loh`` field is
        True. Missing values in an existing ``loh`` field produce a missing
        plotting state in the upstream helper. Used during transformation.
    p_up_states : sequence of str, default ("amp", "loh", "del")
        Base CNV states refined into allele-direction and level labels when
        ``p_up`` and a CNV state source are available. Used during
        transformation.
    p_up_threshold : float, default 0.5
        Direction threshold: ``p_up > p_up_threshold`` gives ``up``;
        values at or below the threshold give ``down``. Used during
        transformation when allele-direction refinement is available.
    level2_pattern : str or None, default "_2"
        Literal substring identifying level-2 states. It controls level
        assignment during transformation and square point markers. Other
        points use circles. None disables level-2 detection.
    cnv_colors : mapping of str to color, optional
        Overrides for individual entries in the default CNA palette. Keys
        are plotting state labels and values are Matplotlib colors.
        Level-1 ``*_1_up/down`` aliases are added for directional colors.
    cnv_labels : mapping of str to str, optional
        Display labels for legend entries. Both exact state labels and
        canonical directional labels such as ``amp_up`` are supported.
    unknown_color : color, default "#666666"
        Fallback color for nonmissing state labels absent from the palette.
    dot_size : float, default 2.0
        Nonnegative scatter-marker area in points squared.
    dot_alpha : float, default 0.6
        Opacity of ordinary points, between zero and one. Level-2 square
        markers are drawn fully opaque.
    raster : bool, default True
        Rasterize scatter points when saving vector formats such as PDF or
        SVG. Text, track borders, and fitted curves remain vector elements.
    exclude_gap : bool, default True
        Shade supplied gap and centromere intervals in light gray when
        ``use_pos=True``. Despite the name, this does not remove observations.
        Ignored in SNP-index mode.
    gaps : pandas.DataFrame, optional
        Gap intervals with chromosome, start, and end columns, defaulting to
        ``CHROM``, ``start``, and ``end``. Column names follow the
        ``region_*`` fields in ``columns``. Coordinates must match the marker
        coordinate convention. Shading is restricted to the sector range.
    acen : pandas.DataFrame, optional
        Centromere intervals in the same format and units as ``gaps``.
    r_lim : tuple of float, default (20.0, 88.0)
        Inner and outer radii allocated to clone bands in pycirclize radius
        units, where the nominal outer radius is 100. Must satisfy
        ``0 <= inner < outer < 94``. The chromosome track occupies radii
        94 through 97. Clone bands share the available radial space equally.
    track_gap : float, default 1.5
        Nonnegative radial gap between consecutive clone bands. Paired
        expression/BAF rings additionally have a fixed internal gap of 0.6.
    chrom_gap : float, default 1.0
        Nonnegative angular spacing between chromosome sectors, in degrees.
        Total spacing must leave positive angular space for chromosome data.
    label_gap : float, default 36.0
        Angular span reserved around the top of the circle for clone labels,
        in degrees. Must be strictly between zero and 180.
    chrom_label_r : float, default 106.0
        Radial position of chromosome labels in pycirclize radius units.
        Increase it to move the labels farther from the chromosome track.
    show_x_ticks : bool, default False
        Draw coordinate ticks on the chromosome track. Physical-position
        labels are shown in megabases; SNP-index labels use the original
        index units.
    tick_interval : float, optional
        Positive finite tick spacing in input coordinate units, rather than
        displayed megabases. Defaults to 50,000,000 in position mode. In
        SNP-index mode, each sector uses its span divided by four, rounded
        upward and bounded below by one. Used only if ``show_x_ticks=True``.
    text_size : float, default 8.0
        Base font size in points for clone labels and other plot annotations.
    show_center_text : bool, default True
        Show signal ranges and, when coordinate ticks are enabled, their
        units in the center of the plot. Set False to leave the center blank.
        Clone labels, chromosome labels, the title, and legend are unaffected.
    title : str, optional
        Figure title. None or an empty string omits the title.
    figsize : tuple of float, default (10.0, 10.0)
        Figure width and height in inches.
    dpi : int, default 100
        Resolution in dots per inch for the figure and automatic file export.
        Must be positive and finite. For vector output such as PDF or SVG,
        this controls the resolution of rasterized elements, including
        scatter points when ``raster=True``.
    legend : bool, default True
        Add a shared legend for nonmissing plotting states present in the
        selected data. State presence is assessed before depth and signal
        filtering, so an entry can remain even if no corresponding point is
        visible.
    legend_fontsize : float, default 8.0
        Legend-entry font size in points.
    legend_kwargs : mapping, optional
        Keyword overrides passed to ``Figure.legend``, such as ``loc``,
        ``bbox_to_anchor``, ``title``, or ``frameon``. The default position is
        just outside the right edge of the figure.
    savepath : str or pathlib.Path, optional
        Output path for automatically saving the completed figure. The file
        format follows the extension, such as ``.png``, ``.pdf``, or ``.svg``.
        Expand a leading ``~`` and create missing parent directories. Save
        with ``dpi=dpi`` and ``bbox_inches="tight"`` to include the legend
        and outer labels. An existing output file is overwritten. If None,
        return the figure without saving it.
    close : bool, default True
        Close the figure in pyplot before returning it to suppress automatic
        display. The returned object remains usable with ``fig.savefig``.
        Automatic saving, when requested, occurs before closing. Set False
        for interactive display.

    Returns
    -------
    matplotlib.figure.Figure
        Circular pseudobulk figure, matching the return type of
        ``plot_bulks``. Use ``bbox_inches="tight"`` when saving to include
        the external legend and chromosome labels.

    Raises
    ------
    ImportError
        If the optional pycirclize dependency is missing.
    KeyError
        If a required coordinate, signal, overlay, state-source, or
        supplied-region column is absent.
    ValueError
        If the table or selection is empty, identifiers or requested orders
        are invalid, coordinate ranges are unusable, chromosome lengths are
        invalid or incomplete, or plotting parameters leave no room for
        sectors or clone tracks. Also raised for invalid track choices,
        numeric limits, opacity, gaps, tick intervals, or DPI. Also raised
        if cytobands are requested in SNP-index mode, a selected chromosome
        is unavailable in hg38, or supplied lengths disagree with hg38.
    OSError
        If an output directory cannot be created or the figure cannot be
        written to ``savepath``, or downloading the cytoband dataset fails.

    See Also
    --------
    spacenumbat.plot.plot_bulks : Plot pseudobulks in rectangular panels.
    spacenumbat.plot.plot_psbulk : Plot an individual pseudobulk profile.

    Notes
    -----
    This function uses the optional ``circos`` extra. From a source checkout,
    install it with ``python -m pip install '.[circos]'``.

    No zero-based/one-based coordinate conversion is performed. Without
    ``chrom_sizes`` or cytobands, physical sectors start at zero and end at
    the largest finite marker or available segment coordinate across
    selected clones, with a minimum length of one. Unobserved chromosome
    tails are therefore absent. With cytobands enabled, sectors instead span
    the full hg38 chromosomes. In SNP-index mode, each sector spans the
    minimum through maximum observed index or available segment endpoint,
    with minimum span one.

    Nonfinite coordinates or signals and missing plotting states are omitted
    from scatter points. Finite expression values are clipped to the
    expression range after baseline correction; this does not alter the
    input table or fitted overlays. BAF points are limited to the displayed
    range ``[-0.05, 1.05]``. Rolling lines break at missing or out-of-range values
    and state changes; no fitted line joins separate chromosomes or clones.
    A clone with no rows on a particular chromosome retains an empty aligned
    track, provided it has rows on at least one selected chromosome.

    Examples
    --------
    Draw one expression ring per clone and save a 300-DPI PNG:

    >>> fig = plot_bulks_circos(
    ...     bulks, dpi=300, savepath="figures/clone_circos.png",
    ... )

    Add hg38 cytobands and use the reference chromosome lengths:

    >>> fig = plot_bulks_circos(
    ...     bulks, cytobands=True, track="both", exp_limit=1.5,
    ...     show_center_text=False,
    ...     dpi=300, savepath="figures/clone_circos_hg38.pdf",
    ... )

    Draw paired rings with physical chromosome lengths read from a TSV:

    >>> lengths = pd.read_table("chrom_sizes.tsv").set_index("CHROM")["length"]
    >>> fig = plot_bulks_circos(
    ...     bulks, track="both", chrom_sizes=lengths,
    ...     min_depth=10, show_x_ticks=True, close=False,
    ... )

    Use a custom clone column while preserving all other field mappings:

    >>> from dataclasses import replace
    >>> custom_columns = replace(DEFAULT_COLUMNS, sample="clone_id")
    >>> renamed = bulks.rename(columns={"sample": "clone_id"})
    >>> fig = plot_bulks_circos(
    ...     renamed, columns=custom_columns, track="pBAF",
    ... )
    """
    try:
        from pycirclize import Circos
    except ModuleNotFoundError as exc:
        if exc.name != "pycirclize":
            raise
        raise ImportError(
            "plot_bulks_circos requires pycirclize. Install it with "
            "`python -m pip install 'spacenumbat[circos]'` or, from a source "
            "checkout, `python -m pip install '.[circos]'`."
        ) from exc

    if bulks.empty:
        raise ValueError("bulks must contain at least one row")
    if not np.isfinite(dpi) or dpi <= 0:
        raise ValueError("dpi must be positive and finite")
    if track not in {"logFC", "pBAF", "both"}:
        raise ValueError("track must be 'logFC', 'pBAF', or 'both'")
    if min_depth < 0 or not np.isfinite(exp_limit) or exp_limit <= 0:
        raise ValueError("min_depth must be non-negative and exp_limit positive")
    if not 0 <= dot_alpha <= 1 or dot_size < 0:
        raise ValueError("dot_alpha must be in [0, 1] and dot_size non-negative")
    if not 0 <= r_lim[0] < r_lim[1] < 94:
        raise ValueError("r_lim must satisfy 0 <= inner < outer < 94")
    if track_gap < 0 or chrom_gap < 0 or not 0 < label_gap < 180:
        raise ValueError("Gaps must be non-negative; label_gap must be in (0, 180)")
    if not use_pos and chrom_sizes is not None:
        raise ValueError("chrom_sizes is only meaningful with use_pos=True")
    if cytobands and not use_pos:
        raise ValueError("cytobands requires use_pos=True and hg38 base-pair coordinates")
    if tick_interval is not None and (not np.isfinite(tick_interval) or tick_interval <= 0):
        raise ValueError("tick_interval must be positive and finite")

    def chrom_name(value):
        name = str(value)
        return name[3:] if name.lower().startswith("chr") else name

    marker = columns.pos if use_pos else columns.snp_index
    start = columns.seg_start if use_pos else columns.seg_start_index
    end = columns.seg_end if use_pos else columns.seg_end_index
    metrics = ["logFC", "pBAF"] if track == "both" else [track]
    required = [columns.chrom, marker]
    if "logFC" in metrics:
        required += [columns.logfc, columns.mu]
        required += [start, end, columns.phi_mle] if phi_mle else [columns.phi_roll]
    if "pBAF" in metrics:
        required += [columns.pbaf, columns.depth]
        if theta_roll:
            required += [columns.theta_roll]
    _require_columns(bulks, required, "Circos pseudobulk plotting")

    # Reset potentially repeated indices from concatenated clone tables.
    df = bulks.loc[bulks[columns.chrom].notna()].copy().reset_index(drop=True)
    if columns.sample not in df:
        df[columns.sample] = "1"
    if df[columns.sample].isna().any():
        raise ValueError("Clone/sample identifiers must not be missing")
    available = list(pd.unique(df[columns.sample]))
    samples = available if sample_order is None else list(sample_order)
    if not samples or len(set(samples)) != len(samples) or any(s not in available for s in samples):
        raise ValueError("sample_order must contain distinct, existing sample identifiers")
    df = df.loc[df[columns.sample].isin(samples)].copy()
    df[columns.chrom] = df[columns.chrom].map(chrom_name)
    present = list(pd.unique(df[columns.chrom]))
    chroms = sorted(present, key=_natural_key) if chrom_order is None else list(map(chrom_name, chrom_order))
    if not chroms or len(set(chroms)) != len(chroms) or any(c not in present for c in chroms):
        raise ValueError("chrom_order must contain distinct, existing chromosome names")
    df = df.loc[df[columns.chrom].isin(chroms)].copy()
    if any(not df[columns.sample].eq(s).any() for s in samples):
        raise ValueError("A selected clone has no rows on the selected chromosomes")
    if chrom_gap * len(chroms) >= 360 - label_gap:
        raise ValueError("chrom_gap is too large for this number of chromosomes")

    numeric_cols = {marker, start, end, columns.logfc, columns.mu, columns.phi_mle,
                    columns.phi_roll, columns.pbaf, columns.depth, columns.theta_roll}
    for name in numeric_cols.intersection(df.columns):
        df[name] = pd.to_numeric(df[name], errors="coerce").astype(float)
    df = _prepare_plot_states(
        df, columns, min_LLR, transform_states, neutral_state, loh_state,
        p_up_states, p_up_threshold, level2_pattern,
    )

    sizes = {}
    for chrom, length in ({} if chrom_sizes is None else dict(chrom_sizes)).items():
        name, length = chrom_name(chrom), float(length)
        if name in sizes or not np.isfinite(length) or length <= 0:
            raise ValueError("chrom_sizes must have unique chromosomes and positive, finite lengths")
        sizes[name] = length
    cytoband_file = None
    if cytobands:
        from pycirclize.parser import Bed
        from pycirclize.utils import load_eukaryote_example_dataset

        bed_file, cytoband_file, _ = load_eukaryote_example_dataset("hg38")
        hg38_sizes = {chrom_name(rec.chr): rec.end for rec in Bed(bed_file).records}
        for chrom in chroms:
            if chrom not in hg38_sizes:
                raise ValueError(f"Chromosome {chrom} is not available in the hg38 cytoband dataset")
            if sizes and chrom not in sizes:
                raise ValueError(f"chrom_sizes is missing chromosome {chrom}")
            if chrom in sizes and sizes[chrom] != hg38_sizes[chrom]:
                raise ValueError(f"chrom_sizes for chromosome {chrom} must match hg38: {hg38_sizes[chrom]}")
        sizes = {chrom: hg38_sizes[chrom] for chrom in chroms}
    coord_cols = [c for c in (marker, start, end) if c in df]
    sectors = {}
    for chrom in chroms:
        coords = df.loc[df[columns.chrom].eq(chrom), coord_cols].to_numpy().ravel()
        coords = coords[np.isfinite(coords)]
        if not len(coords):
            raise ValueError(f"Chromosome {chrom} has no finite coordinates")
        lo = 0.0 if use_pos else float(coords.min())
        hi = sizes.get(chrom, max(float(coords.max()), lo + 1.0))
        if sizes and chrom not in sizes:
            raise ValueError(f"chrom_sizes is missing chromosome {chrom}")
        if coords.min() < lo or coords.max() > hi:
            raise ValueError(f"Coordinates on chromosome {chrom} fall outside its supplied range")
        sectors[f"chr{chrom}" if cytobands else chrom] = (lo, hi)

    colors = default_cnv_colors()
    colors[neutral_state] = "white"
    if cnv_colors is not None:
        colors.update(cnv_colors)
    colors = _with_level1_aliases(colors)
    for state in df[_PLOT_STATE].dropna().unique():
        colors.setdefault(str(state), unknown_color)
    custom_labels = {} if cnv_labels is None else cnv_labels
    labels = {
        state: custom_labels.get(state, custom_labels.get(state.replace("_1_", "_"), state.replace("_1_", "_")))
        for state in colors
    }
    regions = _excluded_regions(gaps, acen, columns) if exclude_gap and use_pos else None
    if regions is not None:
        regions = regions.copy()
        regions["chrom"] = regions["chrom"].map(chrom_name)

    width = (r_lim[1] - r_lim[0] - track_gap * (len(samples) - 1)) / len(samples)
    inner_gap = 0.6 if track == "both" else 0.0
    subwidth = (width - inner_gap) / len(metrics)
    if subwidth <= 0:
        raise ValueError("Too many clones for r_lim and track_gap; decrease track_gap")
    circos = Circos(sectors, start=label_gap / 2, end=360 - label_gap / 2, space=chrom_gap)
    if cytobands:
        circos.add_cytoband_tracks((94, 97), cytoband_file)

    def rolling_line(ring, x, values, states, limits, color=None):
        # Split at missing values and state changes; preserve breaks after sorting.
        data = pd.DataFrame({"x": x, "y": values, "state": states}).sort_values("x", kind="stable")
        valid = (np.isfinite(data.x) & np.isfinite(data.y)
                 & data.y.between(*limits) & data.state.notna())
        changed = data.state.ne(data.state.shift()).fillna(True)
        runs = ((~valid) | (~valid.shift(fill_value=False)) | changed).cumsum()
        for _, run in data.loc[valid].groupby(runs.loc[valid], sort=False):
            ring.line(run.x.to_numpy(), run.y.to_numpy(), vmin=limits[0], vmax=limits[1],
                      color=color or colors.get(str(run.state.iloc[0]), unknown_color), lw=0.7, zorder=4)

    grouped = {s: df.loc[df[columns.sample].eq(s)] for s in samples}
    for i, sample in enumerate(samples):
        top = r_lim[1] - i * (width + track_gap)
        label = str(sample)
        if columns.n_cells in df:
            counts = grouped[sample][columns.n_cells].dropna().unique()
            if len(counts) == 1:
                label += f" (n={counts[0]})"
        circos.text(label, r=top - width / 2, deg=0, size=text_size, ha="center", va="center")

    for ci, sector in enumerate(circos.sectors):
        chrom = chrom_name(sector.name) if cytobands else sector.name
        if cytobands:
            chromosome = sector.get_track("cytoband")
        else:
            chromosome = sector.add_track((94, 97))
            chromosome.axis(fc="#eeeeee" if ci % 2 == 0 else "#cccccc", ec="black", lw=0.5)
        sector.text(f"chr{chrom}", r=chrom_label_r, size=text_size + 1)
        if show_x_ticks:
            interval = tick_interval or (50_000_000 if use_pos else max(1, np.ceil(sector.size / 4)))
            formatter = (lambda x: f"{x / 1e6:g}") if use_pos else (lambda x: f"{x:g}")
            chromosome.xticks_by_interval(interval, label_formatter=formatter, label_size=text_size - 1,
                                          tick_length=1, show_endlabel=False)
        for i, sample in enumerate(samples):
            data = grouped[sample].loc[grouped[sample][columns.chrom].eq(chrom)].sort_values(marker, kind="stable")
            top = r_lim[1] - i * (width + track_gap)
            states = data[_PLOT_STATE].astype("string")
            level2 = (states.str.contains(level2_pattern, regex=False, na=False)
                      if level2_pattern is not None else pd.Series(False, index=data.index))
            x = data[marker]
            for j, metric in enumerate(metrics):
                high = top - j * (subwidth + inner_gap)
                ring = sector.add_track((high - subwidth, high), r_pad_ratio=0.05)
                ring.axis(fc="white", ec="0.75", lw=0.35)
                if regions is not None:
                    for region in regions.loc[regions.chrom.eq(chrom)].itertuples(index=False):
                        a, b = float(region.start), float(region.end)
                        if not np.isfinite([a, b]).all():
                            continue
                        a, b = max(sector.start, a), min(sector.end, b)
                        if a < b:
                            ring.rect(a, b, fc="0.94", ec="none", zorder=0.5)
                if metric == "logFC":
                    limits, baseline = (-exp_limit, exp_limit), 0.0
                    y = data[columns.logfc] - data[columns.mu] * mu_scale
                    y = y.where(np.isfinite(y)).clip(*limits)
                else:
                    limits, baseline = (-0.05, 1.05), 0.5
                    y = data[columns.pbaf].where(data[columns.depth] >= min_depth)
                ring.line([sector.start, sector.end], [baseline, baseline],
                          vmin=limits[0], vmax=limits[1], color="0.55", lw=0.5, ls="--", zorder=1)
                valid = np.isfinite(x) & np.isfinite(y) & y.between(*limits) & states.notna()
                for mask, symbol, alpha in ((valid & ~level2, "o", dot_alpha), (valid & level2, "s", 1.0)):
                    if mask.any():
                        ring.scatter(x.loc[mask].to_numpy(), y.loc[mask].to_numpy(),
                                     vmin=limits[0], vmax=limits[1], s=dot_size, marker=symbol,
                                     c=[colors.get(str(s), unknown_color) for s in states.loc[mask]],
                                     ec="black", lw=0.15, alpha=alpha, rasterized=raster, zorder=2)
                if metric == "logFC" and phi_mle:
                    segments = data[[start, end, columns.phi_mle]].drop_duplicates()
                    for a, b, phi in segments.itertuples(index=False, name=None):
                        if not np.isfinite([a, b, phi]).all() or b < a or phi <= 0:
                            continue
                        value = np.log2(phi)
                        if limits[0] <= value <= limits[1]:
                            ring.line([a, b], [value, value], vmin=limits[0], vmax=limits[1],
                                      color="darkred", lw=0.8, zorder=3)
                elif metric == "logFC":
                    values = np.log2(data[columns.phi_roll].where(data[columns.phi_roll] > 0))
                    rolling_line(ring, x, values, states, limits, color="darkred")
                elif theta_roll:
                    base = data[_PLOT_CNV_STATE] if _PLOT_CNV_STATE in data else states
                    for suffix, sign in (("down", -1), ("up", 1)):
                        values = 0.5 + sign * data[columns.theta_roll]
                        rolling_line(ring, x, values, base.astype("string") + f"_{suffix}", limits)

    if show_center_text:
        scale_text = []
        if "logFC" in metrics:
            scale_text.append(f"logFC: {-exp_limit:g} to {exp_limit:g}")
        if "pBAF" in metrics:
            scale_text.append("pBAF: 0 to 1")
        if show_x_ticks:
            scale_text.append("positions: Mb" if use_pos else "positions: SNP index")
        circos.text("\n".join(scale_text), r=0, size=max(5, text_size - 1), color="0.35", ha="center", va="center")
    fig = circos.plotfig(figsize=figsize, dpi=dpi)
    if title:
        fig.suptitle(title, fontsize=text_size + 3)
    if legend:
        kwargs = {"loc": "center left", "bbox_to_anchor": (0.98, 0.5)}
        if legend_kwargs is not None:
            kwargs.update(legend_kwargs)
        _add_legend(fig, df[_PLOT_STATE], colors, labels, None, True, legend_fontsize, kwargs)
        # White neutral markers need an outline in the legend as well.
        for handle in fig.legends[-1].get_lines():
            handle.set_markeredgecolor("black")
            handle.set_markeredgewidth(0.4)
    if savepath is not None:
        savepath = Path(savepath).expanduser()
        savepath.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=dpi, bbox_inches="tight")
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


