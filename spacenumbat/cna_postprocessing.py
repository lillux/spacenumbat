#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 10:32:16 2026

@author: ccarlino
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd



ACROCENTRIC_P_ARMS = {
    "13p",
    "14p",
    "15p",
    "21p",
    "22p",
}

CNA_STATES = (
    "neu",
    "loh",
    "amp",
    "del",
    "bamp",
    "bdel",
)

# Canonical output and internal column names.
CNA_STATE_COLS = tuple(f"p_{state}" for state in CNA_STATES)

ALTERED_STATES = CNA_STATES[1:]


def _resolve_cna_state_columns(
    state_cols: Mapping[str, str] | Sequence[str] | None,
    ) -> dict[str, str]:
    """
    Resolve source probability columns for the canonical CNA states.

    Parameters
    ----------
    state_cols
        CNA posterior-column specification.

        A mapping is recommended. Keys can be either state names:

            {
                "neu": "posterior_neutral",
                "loh": "posterior_loh",
                ...
            }

        or canonical probability names:

            {
                "p_neu": "posterior_neutral",
                "p_loh": "posterior_loh",
                ...
            }

        A sequence is also accepted for backward compatibility. Its
        columns must follow the order defined by ``CNA_STATES``:

            neu, loh, amp, del, bamp, bdel

        If None, the canonical ``CNA_STATE_COLS`` names are used.

    Returns
    -------
    dict[str, str]
        Mapping from semantic state name to source column name.
    """
    if state_cols is None:
        resolved = dict(
            zip(
                CNA_STATES,
                CNA_STATE_COLS,
                strict=True,
            )
        )

    elif isinstance(state_cols, Mapping):
        resolved: dict[str, str] = {}

        for raw_state, column in state_cols.items():
            state = str(raw_state).strip().lower()

            if state.startswith("p_"):
                state = state.removeprefix("p_")

            if state not in CNA_STATES:
                raise KeyError(
                    f"Unknown CNA state {raw_state!r}. "
                    f"Expected one of: {list(CNA_STATES)}"
                )

            if state in resolved:
                raise ValueError(
                    f"CNA state {state!r} was specified more than once."
                )

            resolved[state] = column

        missing_states = set(CNA_STATES).difference(resolved)

        if missing_states:
            raise KeyError(
                "Missing CNA probability-column mappings for states: "
                f"{sorted(missing_states)}"
            )

    else:
        if isinstance(state_cols, (str, bytes)):
            raise TypeError(
                "state_cols must be a mapping or a sequence of six "
                "column names, not a single string."
            )

        source_columns = list(state_cols)

        if len(source_columns) != len(CNA_STATES):
            raise ValueError(
                f"state_cols must contain exactly {len(CNA_STATES)} "
                "columns in this order: "
                f"{list(CNA_STATES)}"
            )

        resolved = dict(
            zip(
                CNA_STATES,
                source_columns,
                strict=True,
            )
        )

    invalid_columns = {
        state: column
        for state, column in resolved.items()
        if not isinstance(column, str) or not column
    }

    if invalid_columns:
        raise TypeError(
            "Every CNA probability column name must be a non-empty "
            f"string. Invalid entries: {invalid_columns}"
        )

    source_columns = list(resolved.values())

    duplicated = pd.Index(source_columns).duplicated(
        keep=False
    )

    if duplicated.any():
        duplicated_columns = (
            pd.Index(source_columns)[duplicated]
            .unique()
            .tolist()
        )

        raise ValueError(
            "Each CNA state must use a distinct source column. "
            f"Duplicated columns: {duplicated_columns}"
        )

    # Return the mapping in the canonical state order.
    return {
        state: resolved[state]
        for state in CNA_STATES
    }



def _normalize_chromosome(series: pd.Series) -> pd.Series:
    """
    Normalize chromosome labels to:
        1, 2, ..., 22, X, Y
    """
    return (
        series.astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.replace(r"^chr", "", regex=True)
        .str.upper()
    )



def prepare_arm_reference(
    cytoband_arms: pd.DataFrame,
    chrom_col: str = "CHROM",
    start_col: str = "seg_start",
    end_col: str = "seg_end",
    arm_col: str = "seg_id",
    exclude_sex_chromosomes: bool = True,
    exclude_acrocentric_p_arms: bool = False,
    ) -> pd.DataFrame:
    """
    Prepare a common hg38 chromosome-arm reference.

    Parameters
    ----------
    cytoband_arms
        Table with one row per chromosome arm.

    exclude_sex_chromosomes
        Exclude Xp, Xq, Yp and Yq.

    exclude_acrocentric_p_arms
        Independently exclude 13p, 14p, 15p, 21p and 22p.

    Notes
    -----
    No coordinate conversion is performed. The input coordinates are
    assumed to be compatible 0-based coordinates.

    Returns
    -------
    pd.DataFrame
        Standardized chromosome-arm reference.
    """
    arms = cytoband_arms.rename(
        columns={
            chrom_col: "CHROM",
            start_col: "arm_start",
            end_col: "arm_end",
            arm_col: "arm_id",
        }).copy()

    required = {
        "CHROM",
        "arm_start",
        "arm_end",
        "arm_id",
    }

    missing = required.difference(arms.columns)

    if missing:
        raise KeyError(f"Missing chromosome-arm columns: {sorted(missing)}")

    arms["CHROM"] = _normalize_chromosome(arms["CHROM"])
    arms["arm_id"] = arms["arm_id"].astype(str)

    for column in ("arm_start", "arm_end"):
        values = pd.to_numeric(
            arms[column],
            errors="raise",
        )

        if not np.allclose(values, np.round(values)):
            raise ValueError(f"{column!r} contains non-integer coordinates.")

        arms[column] = np.round(values).astype(np.int64)

    if (arms["arm_start"] < 0).any():
        raise ValueError("Chromosome-arm starts cannot be negative.")

    if (arms["arm_end"] <= arms["arm_start"]).any():
        raise ValueError("Every chromosome arm must have arm_end > arm_start.")

    if arms["arm_id"].duplicated().any():
        duplicated = (
            arms.loc[
                arms["arm_id"].duplicated(),
                "arm_id",
                ].unique().tolist()
        )

        raise ValueError(f"arm_id must be unique. Duplicates: {duplicated}")

    arms["is_sex_chromosome"] = arms["CHROM"].isin(["X", "Y"])

    arms["is_acrocentric_p"] = arms["arm_id"].isin(ACROCENTRIC_P_ARMS)

    # independent exclusions of chromosome.
    if exclude_sex_chromosomes:
        arms = arms.loc[~arms["is_sex_chromosome"]].copy()

    if exclude_acrocentric_p_arms:
        arms = arms.loc[~arms["is_acrocentric_p"]].copy()

    arms["arm_length"] = (arms["arm_end"] - arms["arm_start"])

    chromosome_order = {str(chromosome): chromosome for chromosome in range(1, 23)}

    chromosome_order.update({"X": 23,"Y": 24,})

    arms["_chromosome_order"] = (arms["CHROM"].map(chromosome_order))

    arms["_arm_order"] = (arms["arm_id"].str[-1].map({"p": 0,
                                                      "q": 1,
                                                     }))

    if arms[["_chromosome_order", "_arm_order"]].isna().any().any():
        raise ValueError("The arm reference contains non-canonical chromosomes "
                         "or malformed arm identifiers."
                        )

    return (arms.sort_values(["_chromosome_order",
                              "_arm_order",
                             ]
                            ).drop(columns=["_chromosome_order",
                                              "_arm_order",
                                             ]
                                  ).reset_index(drop=True))


def _prepare_clone_post_barcodes(
    clone_post: pd.DataFrame,
    output_cell_col: str,
    clone_post_barcode_col: str | None,
    ) -> pd.DataFrame:
    """
    Return a copy of clone_post with a canonical barcode column.

    Parameters
    ----------
    clone_post
        Clone-posterior table.

    output_cell_col
        Canonical barcode column name used in joint_post and in the output.

    clone_post_barcode_col
        Barcode column in clone_post.

        If None, clone_post.index is used.

    Returns
    -------
    pd.DataFrame
        Copy of clone_post containing `output_cell_col`.
    """
    clone_table = clone_post.copy()

    if clone_post_barcode_col is None:
        barcode_values = clone_post.index.to_numpy(copy=True)

    else:
        if clone_post_barcode_col not in clone_post.columns:
            raise KeyError("clone_post is missing barcode column "
                           f"{clone_post_barcode_col!r}.")

        barcode_values = clone_post[clone_post_barcode_col].to_numpy(copy=True)

    if pd.isna(barcode_values).any():
        source = (
            "clone_post.index"
            if clone_post_barcode_col is None
            else f"clone_post[{clone_post_barcode_col!r}]"
        )

        raise ValueError(f"{source} contains missing barcode identifiers.")

    duplicated = pd.Index(barcode_values).duplicated()

    if duplicated.any():
        duplicated_barcodes = (
            pd.Index(barcode_values[duplicated])
            .unique()
            .tolist()
        )

        raise ValueError("clone_post must contain one row per barcode. "
                         f"Duplicated barcodes include: {duplicated_barcodes[:10]}")

    # Reset the index without retaining it because barcode_values have
    # already been extracted explicitly.
    clone_table = clone_table.reset_index(drop=True)

    # The canonical barcode is always taken from the requested source.
    # Drop a pre-existing column with the same output name to avoid
    # ambiguity when the index is selected as the barcode source.
    if output_cell_col in clone_table.columns:
        clone_table = clone_table.drop(
            columns=output_cell_col
        )

    clone_table.insert(
        0,
        output_cell_col,
        barcode_values,
    )

    return clone_table


def _arm_probability_column(
    state: str,
    posterior_name: str,
    ) -> str:
    """
    Return the canonical output column for one posterior set.

    Examples
    --------
    posterior_name=""       -> p_amp
    posterior_name="local"  -> p_amp_local
    """
    return (
        f"p_{state}"
        if not posterior_name
        else f"p_{state}_{posterior_name}"
    )


def _arm_derived_column(
    name: str,
    posterior_name: str,
    ) -> str:
    """
    Return the output name for a derived posterior quantity.

    Examples
    --------
    posterior_name=""       -> p_cnv
    posterior_name="local"  -> p_cnv_local
    """
    return (
        name
        if not posterior_name
        else f"{name}_{posterior_name}"
    )


def _resolve_probability_sets(
    state_cols: Mapping[str, str] | Sequence[str] | None,
    local_state_cols: Mapping[str, str] | Sequence[str] | None,
    ) -> dict[str, dict[str, str]]:
    """
    Resolve primary and optional local CNA posterior columns.

    The primary set is stored under the empty name ``""`` and produces
    canonical output columns such as ``p_amp``.

    The optional local set is stored under ``"local"`` and produces
    columns such as ``p_amp_local``.
    """
    resolved = {
        "": _resolve_cna_state_columns(state_cols),
    }

    if local_state_cols is not None:
        resolved["local"] = _resolve_cna_state_columns(
            local_state_cols
        )

    return resolved


def _harmonize_probability_sets(
    table: pd.DataFrame,
    state_cols: Mapping[str, str] | Sequence[str] | None,
    local_state_cols: Mapping[str, str] | Sequence[str] | None,
    table_name: str,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, str]]]:
    """
    Copy custom posterior columns to canonical internal/output names.

    Source columns are retained.

    For example:

        posterior_gain       -> p_amp
        posterior_gain_local -> p_amp_local
    """
    probability_sets = _resolve_probability_sets(
        state_cols=state_cols,
        local_state_cols=local_state_cols,
    )

    out = table.copy()

    required_source_columns = {
        source_column
        for state_map in probability_sets.values()
        for source_column in state_map.values()
    }

    missing = required_source_columns.difference(
        out.columns
    )

    if missing:
        raise KeyError(
            f"{table_name} is missing CNA posterior columns: "
            f"{sorted(missing)}"
        )

    for posterior_name, state_map in probability_sets.items():
        for state in CNA_STATES:
            source_column = state_map[state]

            output_column = _arm_probability_column(
                state,
                posterior_name,
            )

            source_values = pd.to_numeric(
                out[source_column],
                errors="raise",
            )

            # Do not silently overwrite an existing canonical column
            # that disagrees with the explicitly selected source.
            if (
                output_column in out.columns
                and output_column != source_column
            ):
                existing_values = pd.to_numeric(
                    out[output_column],
                    errors="raise",
                )

                if not np.allclose(
                    existing_values.to_numpy(dtype=float),
                    source_values.to_numpy(dtype=float),
                    atol=0.0,
                    rtol=0.0,
                    equal_nan=True,
                ):
                    raise ValueError(
                        f"{table_name} already contains "
                        f"{output_column!r}, but it disagrees with "
                        f"the selected source column "
                        f"{source_column!r}."
                    )

            out[output_column] = source_values

    return out, probability_sets


def _coerce_interval_columns(
    table: pd.DataFrame,
    start_col: str,
    end_col: str,
    *,
    table_name: str,
    ) -> pd.DataFrame:
    """Validate and convert genomic interval coordinates."""
    out = table.copy()

    for column in (start_col, end_col):
        values = pd.to_numeric(
            out[column],
            errors="raise",
        )

        numeric = values.to_numpy(dtype=float)

        if not np.isfinite(numeric).all():
            raise ValueError(
                f"{table_name}[{column!r}] contains "
                "non-finite coordinates."
            )

        rounded = np.round(numeric)

        if not np.allclose(
            numeric,
            rounded,
            atol=1e-8,
            rtol=0,
        ):
            raise ValueError(
                f"{table_name}[{column!r}] contains "
                "non-integer coordinates."
            )

        out[column] = rounded.astype(np.int64)

    if (out[start_col] < 0).any():
        raise ValueError(
            f"{table_name}[{start_col!r}] contains "
            "negative coordinates."
        )

    if (out[end_col] <= out[start_col]).any():
        raise ValueError(
            f"Every interval in {table_name} must have "
            f"{end_col} > {start_col}."
        )

    return out


def _validate_probability_sets(
    table: pd.DataFrame,
    probability_sets: Mapping[str, Mapping[str, str]],
    *,
    probability_tolerance: float,
    table_name: str,
    ) -> pd.DataFrame:
    """
    Validate all primary/local posterior vectors.

    Only negligible floating-point deviations are clipped and
    renormalized.
    """
    out = table.copy()

    for posterior_name in probability_sets:
        columns = [
            _arm_probability_column(
                state,
                posterior_name,
            )
            for state in CNA_STATES
        ]

        probabilities = out[
            columns
        ].to_numpy(dtype=float)

        if not np.isfinite(probabilities).all():
            raise ValueError(
                f"{table_name} contains non-finite probabilities "
                f"in posterior set "
                f"{posterior_name or 'primary'!r}."
            )

        outside_range = (
            (
                probabilities
                < -probability_tolerance
            ).any()
            or
            (
                probabilities
                > 1.0 + probability_tolerance
            ).any()
        )

        if outside_range:
            raise ValueError(
                f"{table_name} probabilities in posterior set "
                f"{posterior_name or 'primary'!r} must lie "
                "within [0, 1]."
            )

        row_sums = probabilities.sum(axis=1)

        if not np.allclose(
            row_sums,
            1.0,
            atol=probability_tolerance,
            rtol=0,
        ):
            raise ValueError(
                f"{table_name} probabilities in posterior set "
                f"{posterior_name or 'primary'!r} do not sum "
                "to one. Observed range: "
                f"{row_sums.min():.6g}–"
                f"{row_sums.max():.6g}"
            )

        clipped = np.clip(
            probabilities,
            0.0,
            1.0,
        )

        clipped_sums = clipped.sum(
            axis=1,
            keepdims=True,
        )

        if (clipped_sums <= 0).any():
            raise ValueError(
                f"{table_name} contains a zero-mass posterior "
                f"vector in set "
                f"{posterior_name or 'primary'!r}."
            )

        out[columns] = clipped / clipped_sums

    return out


def _canonical_cna_state(value: object) -> str | None:
    """Normalize a CNA-state label.""" 
    if pd.isna(value): 
        return None 
    state = str(value).strip().lower() 
    aliases = {"neutral": "neu", 
               "normal": "neu",
               "diploid": "neu", 
               "gain": "amp", 
               "loss": "del",
               } 
    return aliases.get(state, state)


def _parse_consensus_states(
    value: object,
    ) -> list[str]:
    """
    Parse and deduplicate a SpaceNumbat consensus-state string.
    """
    if pd.isna(value):
        return []

    states = [
        _canonical_cna_state(state)
        for state in str(value).split(",")
    ]

    states = [
        state
        for state in states
        if state in CNA_STATES
    ]

    return list(dict.fromkeys(states))



def resolve_multistate_joint_segments(
    joint_post: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    cell_col: str = "cell",
    chrom_col: str = "CHROM",
    start_col: str = "seg_start",
    end_col: str = "seg_end",
    segment_col: str = "seg",
    expanded_state_col: str = "cnv_state",
    consensus_segment_col: str = "seg_cons",
    consensus_states_col: str = "cnv_states",
    state_cols: Mapping[str, str] | Sequence[str] | None = None,
    local_state_cols: (
        Mapping[str, str]
        | Sequence[str]
        | None
    ) = None,
    probability_tolerance: float = 1e-8,
    validate_expanded_states: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse rows created by SpaceNumbat ``expand_states()``.

    The complete primary and optional local posterior vectors are
    preserved unchanged. Existing SpaceNumbat MAP and MLE calls are
    retained rather than recalculated.

    Parameters
    ----------
    joint_post
        SpaceNumbat joint posterior table. It may be the expanded table
        produced by ``operations.expand_states()``.

    segs_consensus
        SpaceNumbat consensus-segment table.

    state_cols
        Primary posterior columns.

        For HMRF output, these are normally the spatially regularized
        columns:

            p_neu, p_loh, p_amp, p_del, p_bamp, p_bdel

        Custom source names are accepted through a mapping.

    local_state_cols
        Optional non-spatial posterior columns. For native HMRF output:

            p_neu_local
            p_loh_local
            p_amp_local
            p_del_local
            p_bamp_local
            p_bdel_local

        When supplied, the returned table contains canonical local
        columns with these names.

    Returns
    -------
    resolved_joint_post
        One row per barcode and physical genomic interval.

    resolution_log
        One audit row per expanded group that was collapsed.

    Notes
    -----
    ``p_cnv`` and ``p_n`` are recomputed because ``expand_states()``
    replaces them with state-specific quantities.

    ``cnv_state_map``, ``cnv_state_map_local``, and
    ``cnv_state_mle`` are preserved.
    """
    if probability_tolerance < 0:
        raise ValueError(
            "probability_tolerance must be non-negative."
        )

    joint_required = {
        cell_col,
        chrom_col,
        start_col,
        end_col,
        segment_col,
    }

    missing = joint_required.difference(
        joint_post.columns
    )

    if missing:
        raise KeyError(
            f"Missing joint_post columns: {sorted(missing)}"
        )

    consensus_required = {
        chrom_col,
        start_col,
        end_col,
        consensus_segment_col,
    }

    missing = consensus_required.difference(
        segs_consensus.columns
    )

    if missing:
        raise KeyError(
            f"Missing segs_consensus columns: "
            f"{sorted(missing)}"
        )

    joint, probability_sets = (
        _harmonize_probability_sets(
            joint_post,
            state_cols=state_cols,
            local_state_cols=local_state_cols,
            table_name="joint_post",
        )
    )

    consensus = segs_consensus.copy()

    joint[chrom_col] = _normalize_chromosome(
        joint[chrom_col]
    )

    consensus[chrom_col] = _normalize_chromosome(
        consensus[chrom_col]
    )

    if joint[chrom_col].isna().any():
        raise ValueError(
            f"joint_post[{chrom_col!r}] contains "
            "missing chromosome labels."
        )

    if consensus[chrom_col].isna().any():
        raise ValueError(
            f"segs_consensus[{chrom_col!r}] contains "
            "missing chromosome labels."
        )

    joint = _coerce_interval_columns(
        joint,
        start_col=start_col,
        end_col=end_col,
        table_name="joint_post",
    )

    consensus = _coerce_interval_columns(
        consensus,
        start_col=start_col,
        end_col=end_col,
        table_name="segs_consensus",
    )

    joint = _validate_probability_sets(
        joint,
        probability_sets,
        probability_tolerance=probability_tolerance,
        table_name="joint_post",
    )

    genomic_key = [
        chrom_col,
        start_col,
        end_col,
    ]

    duplicated_consensus = consensus.duplicated(
        genomic_key,
        keep=False,
    )

    if duplicated_consensus.any():
        examples = consensus.loc[
            duplicated_consensus,
            [
                *genomic_key,
                consensus_segment_col,
            ],
        ].head()

        raise ValueError(
            "segs_consensus must contain one row per "
            "physical genomic interval. Duplicates were "
            f"found:\n{examples}"
        )

    consensus_lookup = {
        tuple(
            row[column]
            for column in genomic_key
        ): row
        for _, row in consensus.iterrows()
    }

    grouping_key = [
        cell_col,
        chrom_col,
        start_col,
        end_col,
    ]

    # These values are computed before expand_states() and should be
    # identical across all expanded copies.
    consistent_summary_columns = [
        column
        for column in (
            "cnv_state_mle",
            "cnv_state_map",
            "cnv_state_map_local",
            "hmrf_iterations",
            "hmrf_converged",
        )
        if column in joint.columns
    ]

    resolved_rows: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []

    grouped = joint.groupby(
        grouping_key,
        sort=False,
        observed=True,
        dropna=False,
    )

    for group_key, group in grouped:
        (
            barcode,
            chromosome,
            segment_start,
            segment_end,
        ) = group_key

        consensus_key = (
            chromosome,
            segment_start,
            segment_end,
        )

        consensus_row = consensus_lookup.get(
            consensus_key
        )

        if consensus_row is None and len(group) > 1:
            raise ValueError(
                "A duplicated joint_post interval could not "
                "be matched to segs_consensus:\n"
                f"cell={barcode!r}, "
                f"interval={consensus_key}"
            )

        resolved = group.iloc[0].copy()

        if len(group) == 1:
            state = (
                _canonical_cna_state(
                    resolved[expanded_state_col]
                )
                if expanded_state_col in resolved.index
                else None
            )

            resolved["resolved_cnv_states"] = (
                state
                if state in CNA_STATES
                else ""
            )

            resolved["n_joint_rows_collapsed"] = 1
            resolved_rows.append(resolved)
            continue

        allowed_states: list[str] = []

        if (
            consensus_row is not None
            and consensus_states_col
            in consensus.columns
        ):
            allowed_states = _parse_consensus_states(
                consensus_row[
                    consensus_states_col
                ]
            )

        observed_states: list[str] = []

        if expanded_state_col in group.columns:
            observed_states = [
                state
                for state in (
                    _canonical_cna_state(value)
                    for value
                    in group[expanded_state_col]
                )
                if state in CNA_STATES
            ]

            observed_states = list(
                dict.fromkeys(observed_states)
            )

        if not allowed_states:
            allowed_states = observed_states

        if (
            validate_expanded_states
            and allowed_states
            and set(observed_states)
            != set(allowed_states)
        ):
            raise ValueError(
                "Expanded SpaceNumbat rows do not match "
                "the consensus state list:\n"
                f"cell={barcode!r}, "
                f"interval={consensus_key}, "
                f"observed={observed_states}, "
                f"consensus={allowed_states}"
            )

        # expand_states() copies the complete posterior vector.
        # Verify that property and retain one copy unchanged.
        for posterior_name in probability_sets:
            probability_columns = [
                _arm_probability_column(
                    state,
                    posterior_name,
                )
                for state in CNA_STATES
            ]

            probability_matrix = group[
                probability_columns
            ].to_numpy(dtype=float)

            reference_vector = probability_matrix[0]

            if not np.allclose(
                probability_matrix,
                reference_vector[None, :],
                atol=probability_tolerance,
                rtol=0,
            ):
                raise ValueError(
                    "Rows representing the same physical "
                    "segment contain different complete posterior "
                    "vectors. This is not expected from "
                    "SpaceNumbat expand_states():\n"
                    f"cell={barcode!r}, "
                    f"interval={consensus_key}, "
                    f"posterior_set="
                    f"{posterior_name or 'primary'!r}"
                )

            resolved[
                probability_columns
            ] = reference_vector

        # Preserve existing MAP/MLE and HMRF diagnostics.
        for column in consistent_summary_columns:
            non_missing = group[column].dropna()
            unique_values = pd.unique(non_missing)

            if len(unique_values) > 1:
                raise ValueError(
                    f"Expanded rows disagree in {column!r} "
                    f"for cell={barcode!r}, "
                    f"interval={consensus_key}: "
                    f"{unique_values.tolist()}"
                )

            resolved[column] = (
                unique_values[0]
                if len(unique_values) == 1
                else pd.NA
            )

        physical_segment = (
            consensus_row[consensus_segment_col]
            if consensus_row is not None
            else None
        )

        if pd.notna(physical_segment):
            resolved[segment_col] = physical_segment

        # These fields are state-specific after expand_states() and
        # do not represent the collapsed physical segment.
        if expanded_state_col in resolved.index:
            resolved[expanded_state_col] = pd.NA

        if "Z_cnv" in resolved.index:
            resolved["Z_cnv"] = np.nan

        if "seg_label" in resolved.index:
            resolved["seg_label"] = str(
                resolved[segment_col]
            )

        resolved["resolved_cnv_states"] = ",".join(
            allowed_states
        )

        resolved["n_joint_rows_collapsed"] = len(
            group
        )

        resolved_rows.append(resolved)

        audit_rows.append(
            {
                cell_col: barcode,
                chrom_col: chromosome,
                start_col: segment_start,
                end_col: segment_end,
                "input_rows": len(group),
                "input_segment_labels": ",".join(
                    group[segment_col]
                    .astype(str)
                    .drop_duplicates()
                ),
                "input_states": ",".join(
                    observed_states
                ),
                "consensus_segment": physical_segment,
                "consensus_states": ",".join(
                    allowed_states
                ),
                "primary_posterior_validated": True,
                "local_posterior_validated": (
                    "local" in probability_sets
                ),
            }
        )

    resolved_joint_post = pd.DataFrame(
        resolved_rows
    ).reset_index(drop=True)

    duplicated_after = (
        resolved_joint_post.duplicated(
            grouping_key,
            keep=False,
        )
    )

    if duplicated_after.any():
        raise RuntimeError(
            "Multistate resolution did not produce "
            "unique barcode–interval rows."
        )

    resolved_joint_post = _validate_probability_sets(
        resolved_joint_post,
        probability_sets,
        probability_tolerance=probability_tolerance,
        table_name="resolved_joint_post",
    )

    # Recompute only quantities that expand_states() converts into
    # state-specific values.
    for posterior_name in probability_sets:
        neutral_column = _arm_probability_column(
            "neu",
            posterior_name,
        )

        p_n_column = _arm_derived_column(
            "p_n",
            posterior_name,
        )

        p_cnv_column = _arm_derived_column(
            "p_cnv",
            posterior_name,
        )

        resolved_joint_post[p_n_column] = (
            resolved_joint_post[neutral_column]
        )

        resolved_joint_post[p_cnv_column] = (
            1.0
            - resolved_joint_post[neutral_column]
        )

    resolution_log = pd.DataFrame(
        audit_rows
    )

    return resolved_joint_post, resolution_log




def build_barcode_arm_posteriors(
    joint_post: pd.DataFrame,
    clone_post: pd.DataFrame,
    arm_reference: pd.DataFrame,
    cell_col: str = "cell",
    clone_post_barcode_col: str | None = None,
    chrom_col: str = "CHROM",
    start_col: str = "seg_start",
    end_col: str = "seg_end",
    state_cols: Mapping[str, str] | Sequence[str] | None = None,
    local_state_cols: (
        Mapping[str, str]
        | Sequence[str]
        | None
    ) = None,
    probability_tolerance: float = 1e-5,
    validate_nonoverlap: bool = True,
    clone_metadata_cols: Sequence[str] = (
        "clone_opt",
        "GT_opt",
        "p_opt",
        "p_cnv",
        "compartment_opt",
    ),
    ) -> pd.DataFrame:
    """
    Project SpaceNumbat segment posteriors onto chromosome arms.

    Primary and optional local posterior vectors are projected
    independently.

    The primary set produces:

        p_neu, p_loh, p_amp, p_del, p_bamp, p_bdel
        p_gain, p_loss, p_cnv, signed_cna
        expected_altered_bp
        state_entropy
        dominant_state
        dominant_probability

    The local set produces the corresponding ``*_local`` columns.

    Notes
    -----
    These are length-weighted expected arm occupancies. They are not
    posterior probabilities that the entire chromosome arm is in one
    specific state.

    Unreported genomic fractions are treated as neutral.
    """
    if probability_tolerance < 0:
        raise ValueError(
            "probability_tolerance must be non-negative."
        )

    coordinate_columns = [
        cell_col,
        chrom_col,
        start_col,
        end_col,
    ]

    if len(set(coordinate_columns)) != len(
        coordinate_columns
    ):
        raise ValueError(
            "cell_col, chrom_col, start_col, and end_col "
            "must be distinct."
        )

    required_joint_columns = set(
        coordinate_columns
    )

    missing = required_joint_columns.difference(
        joint_post.columns
    )

    if missing:
        raise KeyError(
            f"Missing joint_post columns: {sorted(missing)}"
        )

    required_arm_columns = {
        "CHROM",
        "arm_id",
        "arm_start",
        "arm_end",
        "arm_length",
        "is_sex_chromosome",
        "is_acrocentric_p",
    }

    missing = required_arm_columns.difference(
        arm_reference.columns
    )

    if missing:
        raise KeyError(
            f"Missing arm-reference columns: "
            f"{sorted(missing)}"
        )

    clone_table = _prepare_clone_post_barcodes(
        clone_post,
        output_cell_col=cell_col,
        clone_post_barcode_col=clone_post_barcode_col,
    )

    barcodes = clone_table[
        [cell_col]
    ].copy()

    harmonized, probability_sets = (
        _harmonize_probability_sets(
            joint_post,
            state_cols=state_cols,
            local_state_cols=local_state_cols,
            table_name="joint_post",
        )
    )

    probability_columns = [
        _arm_probability_column(
            state,
            posterior_name,
        )
        for posterior_name in probability_sets
        for state in CNA_STATES
    ]

    segments = (
        harmonized[
            [
                cell_col,
                chrom_col,
                start_col,
                end_col,
                *probability_columns,
            ]
        ]
        .rename(
            columns={
                chrom_col: "CHROM",
                start_col: "seg_start",
                end_col: "seg_end",
            }
        )
        .copy()
    )

    if segments[cell_col].isna().any():
        raise ValueError(
            f"joint_post[{cell_col!r}] contains "
            "missing barcodes."
        )

    known_barcodes = pd.Index(
        barcodes[cell_col]
    )

    unknown_barcodes = pd.Index(
        segments[cell_col].drop_duplicates()
    ).difference(known_barcodes)

    if len(unknown_barcodes) > 0:
        raise ValueError(
            "joint_post contains barcodes absent from "
            "clone_post. Examples: "
            f"{unknown_barcodes[:10].tolist()}"
        )

    segments["CHROM"] = _normalize_chromosome(
        segments["CHROM"]
    )

    if segments["CHROM"].isna().any():
        raise ValueError(
            "joint_post contains missing chromosome labels."
        )

    segments = _coerce_interval_columns(
        segments,
        start_col="seg_start",
        end_col="seg_end",
        table_name="joint_post",
    )

    segments = _validate_probability_sets(
        segments,
        probability_sets,
        probability_tolerance=probability_tolerance,
        table_name="joint_post",
    )

    arms = arm_reference.copy()

    arms["CHROM"] = _normalize_chromosome(
        arms["CHROM"]
    )

    arms = _coerce_interval_columns(
        arms,
        start_col="arm_start",
        end_col="arm_end",
        table_name="arm_reference",
    )

    if arms["arm_id"].duplicated().any():
        duplicates = (
            arms.loc[
                arms["arm_id"].duplicated(
                    keep=False
                ),
                "arm_id",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "arm_reference contains duplicated arm_id "
            f"values: {duplicates}"
        )

    expected_arm_length = (
        arms["arm_end"]
        - arms["arm_start"]
    )

    supplied_arm_length = pd.to_numeric(
        arms["arm_length"],
        errors="raise",
    ).to_numpy(dtype=np.int64)

    if not np.array_equal(
        expected_arm_length.to_numpy(
            dtype=np.int64
        ),
        supplied_arm_length,
    ):
        raise ValueError(
            "arm_reference.arm_length must equal "
            "arm_end - arm_start."
        )

    arms["arm_length"] = expected_arm_length.astype(
        np.int64
    )

    # Ignore chromosomes that were intentionally excluded from
    # the selected arm reference.
    segments = segments.loc[
        segments["CHROM"].isin(
            arms["CHROM"]
        )
    ].copy()

    if validate_nonoverlap and not segments.empty:
        ordered = segments.sort_values(
            [
                cell_col,
                "CHROM",
                "seg_start",
                "seg_end",
            ],
            kind="mergesort",
        )

        previous_end = (
            ordered.groupby(
                [
                    cell_col,
                    "CHROM",
                ],
                sort=False,
                observed=True,
            )["seg_end"]
            .shift()
        )

        overlaps_previous = (
            ordered["seg_start"]
            < previous_end
        )

        if overlaps_previous.any():
            examples = ordered.loc[
                overlaps_previous,
                [
                    cell_col,
                    "CHROM",
                    "seg_start",
                    "seg_end",
                ],
            ].head()

            raise ValueError(
                "Overlapping physical CNA segments were "
                "found within a barcode and chromosome:\n"
                f"{examples}"
            )

    arm_columns = [
        "CHROM",
        "arm_id",
        "arm_start",
        "arm_end",
        "arm_length",
        "is_sex_chromosome",
        "is_acrocentric_p",
    ]

    background = barcodes.merge(
        arms[arm_columns],
        how="cross",
    )

    # Every unreported arm fraction starts as neutral.
    for posterior_name in probability_sets:
        for state in CNA_STATES:
            column = _arm_probability_column(
                state,
                posterior_name,
            )

            background[column] = (
                1.0
                if state == "neu"
                else 0.0
            )

    background["reported_segment_fraction"] = 0.0

    if not segments.empty:
        overlaps = segments.merge(
            arms[arm_columns],
            on="CHROM",
            how="inner",
            validate="many_to_many",
        )

        overlaps["overlap_start"] = np.maximum(
            overlaps["seg_start"],
            overlaps["arm_start"],
        )

        overlaps["overlap_end"] = np.minimum(
            overlaps["seg_end"],
            overlaps["arm_end"],
        )

        overlaps["overlap_bp"] = (
            overlaps["overlap_end"]
            - overlaps["overlap_start"]
        ).clip(lower=0)

        overlaps = overlaps.loc[
            overlaps["overlap_bp"] > 0
        ].copy()

        overlaps["_overlap_fraction"] = (
            overlaps["overlap_bp"]
            / overlaps["arm_length"]
        )

        aggregation_spec: dict[
            str,
            tuple[str, str],
        ] = {
            "_reported_segment_fraction": (
                "_overlap_fraction",
                "sum",
            )
        }

        temporary_columns: list[str] = []

        for posterior_name in probability_sets:
            tag = (
                posterior_name
                if posterior_name
                else "primary"
            )

            neutral_column = _arm_probability_column(
                "neu",
                posterior_name,
            )

            neutral_delta_column = (
                f"__neutral_delta_{tag}"
            )

            overlaps[neutral_delta_column] = (
                overlaps["_overlap_fraction"]
                * (
                    overlaps[neutral_column]
                    - 1.0
                )
            )

            aggregation_spec[
                neutral_delta_column
            ] = (
                neutral_delta_column,
                "sum",
            )

            temporary_columns.append(
                neutral_delta_column
            )

            for state in ALTERED_STATES:
                source_column = (
                    _arm_probability_column(
                        state,
                        posterior_name,
                    )
                )

                mass_column = (
                    f"__{state}_mass_{tag}"
                )

                overlaps[mass_column] = (
                    overlaps["_overlap_fraction"]
                    * overlaps[source_column]
                )

                aggregation_spec[mass_column] = (
                    mass_column,
                    "sum",
                )

                temporary_columns.append(
                    mass_column
                )

        arm_mass = (
            overlaps.groupby(
                [
                    cell_col,
                    "arm_id",
                ],
                observed=True,
                as_index=False,
            )
            .agg(**aggregation_spec)
        )

        background = background.merge(
            arm_mass,
            on=[
                cell_col,
                "arm_id",
            ],
            how="left",
            validate="one_to_one",
        )

        background["reported_segment_fraction"] = (
            background[
                "_reported_segment_fraction"
            ]
            .fillna(0.0)
        )

        for posterior_name in probability_sets:
            tag = (
                posterior_name
                if posterior_name
                else "primary"
            )

            neutral_column = _arm_probability_column(
                "neu",
                posterior_name,
            )

            background[neutral_column] += (
                background[
                    f"__neutral_delta_{tag}"
                ]
                .fillna(0.0)
            )

            for state in ALTERED_STATES:
                output_column = (
                    _arm_probability_column(
                        state,
                        posterior_name,
                    )
                )

                background[output_column] += (
                    background[
                        f"__{state}_mass_{tag}"
                    ]
                    .fillna(0.0)
                )

        background = background.drop(
            columns=[
                "_reported_segment_fraction",
                *temporary_columns,
            ]
        )

    if (
        background["reported_segment_fraction"]
        > 1.0 + probability_tolerance
    ).any():
        raise ValueError(
            "Reported segments cover an arm more than "
            "once for at least one barcode."
        )

    # Validate each arm posterior set and derive arm summaries.
    for posterior_name in probability_sets:
        state_probability_columns = [
            _arm_probability_column(
                state,
                posterior_name,
            )
            for state in CNA_STATES
        ]

        arm_probabilities = background[
            state_probability_columns
        ].to_numpy(dtype=float)

        outside_range = (
            (
                arm_probabilities
                < -probability_tolerance
            ).any()
            or
            (
                arm_probabilities
                > 1.0 + probability_tolerance
            ).any()
        )

        if outside_range:
            raise ValueError(
                "Arm projection produced probabilities "
                "outside [0, 1] for posterior set "
                f"{posterior_name or 'primary'!r}."
            )

        if not np.allclose(
            arm_probabilities.sum(axis=1),
            1.0,
            atol=probability_tolerance,
            rtol=0,
        ):
            raise ValueError(
                "Projected chromosome-arm probabilities "
                "do not sum to one for posterior set "
                f"{posterior_name or 'primary'!r}."
            )

        clipped = np.clip(
            arm_probabilities,
            0.0,
            1.0,
        )

        clipped = (
            clipped
            / clipped.sum(
                axis=1,
                keepdims=True,
            )
        )

        background[
            state_probability_columns
        ] = clipped

        p_neu = _arm_probability_column(
            "neu",
            posterior_name,
        )

        p_amp = _arm_probability_column(
            "amp",
            posterior_name,
        )

        p_bamp = _arm_probability_column(
            "bamp",
            posterior_name,
        )

        p_del = _arm_probability_column(
            "del",
            posterior_name,
        )

        p_bdel = _arm_probability_column(
            "bdel",
            posterior_name,
        )

        p_gain = _arm_derived_column(
            "p_gain",
            posterior_name,
        )

        p_loss = _arm_derived_column(
            "p_loss",
            posterior_name,
        )

        p_cnv = _arm_derived_column(
            "p_cnv",
            posterior_name,
        )

        signed_cna = _arm_derived_column(
            "signed_cna",
            posterior_name,
        )

        expected_altered_bp = (
            _arm_derived_column(
                "expected_altered_bp",
                posterior_name,
            )
        )

        entropy_column = _arm_derived_column(
            "state_entropy",
            posterior_name,
        )

        dominant_state_column = (
            _arm_derived_column(
                "dominant_state",
                posterior_name,
            )
        )

        dominant_probability_column = (
            _arm_derived_column(
                "dominant_probability",
                posterior_name,
            )
        )

        background[p_gain] = (
            background[p_amp]
            + background[p_bamp]
        )

        background[p_loss] = (
            background[p_del]
            + background[p_bdel]
        )

        background[p_cnv] = (
            1.0
            - background[p_neu]
        )

        background[signed_cna] = (
            background[p_gain]
            - background[p_loss]
        )

        background[expected_altered_bp] = (
            background[p_cnv]
            * background["arm_length"]
        )

        probabilities = background[
            state_probability_columns
        ].to_numpy(dtype=float)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            entropy = -np.where(
                probabilities > 0,
                probabilities
                * np.log(probabilities),
                0.0,
            ).sum(axis=1)

        background[entropy_column] = (
            entropy
            / np.log(len(CNA_STATES))
        )

        dominant_index = probabilities.argmax(
            axis=1
        )

        state_names = np.asarray(
            CNA_STATES,
            dtype=object,
        )

        background[dominant_state_column] = (
            state_names[dominant_index]
        )

        background[
            dominant_probability_column
        ] = probabilities[
            np.arange(len(background)),
            dominant_index,
        ]

    metadata_columns = [
        column
        for column in clone_metadata_cols
        if column in clone_table.columns
    ]

    metadata = clone_table[
        [
            cell_col,
            *metadata_columns,
        ]
    ].copy()

    # Avoid collision between clone-level and arm-level p_cnv.
    if "p_cnv" in metadata.columns:
        metadata = metadata.rename(
            columns={
                "p_cnv": "clone_p_cnv",
            }
        )

    if "p_cnv_local" in metadata.columns:
        metadata = metadata.rename(
            columns={
                "p_cnv_local": "clone_p_cnv_local",
            }
        )

    background = background.merge(
        metadata,
        on=cell_col,
        how="left",
        validate="many_to_one",
    )

    arm_order = {
        arm_id: position
        for position, arm_id
        in enumerate(arms["arm_id"])
    }

    barcode_order = {
        barcode: position
        for position, barcode
        in enumerate(barcodes[cell_col])
    }

    background["_arm_order"] = (
        background["arm_id"].map(
            arm_order
        )
    )

    background["_barcode_order"] = (
        background[cell_col].map(
            barcode_order
        )
    )

    return (
        background.sort_values(
            [
                "_barcode_order",
                "_arm_order",
            ],
            kind="mergesort",
        )
        .drop(
            columns=[
                "_barcode_order",
                "_arm_order",
            ]
        )
        .reset_index(drop=True)
    )



def build_spacenumbat_arm_posteriors(
    joint_post: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    clone_post: pd.DataFrame,
    arm_reference: pd.DataFrame,
    state_cols: Mapping[str, str] | Sequence[str] | None = None,
    local_state_cols: (
        Mapping[str, str]
        | Sequence[str]
        | None
    ) = None,
    cell_col: str = "cell",
    clone_post_barcode_col: str | None = None,
    chrom_col: str = "CHROM",
    start_col: str = "seg_start",
    end_col: str = "seg_end",
    segment_col: str = "seg",
    expanded_state_col: str = "cnv_state",
    consensus_segment_col: str = "seg_cons",
    consensus_states_col: str = "cnv_states",
    probability_tolerance: float = 1e-8,
    validate_expanded_states: bool = True,
    validate_nonoverlap: bool = True,
    clone_metadata_cols: Sequence[str] = (
        "clone_opt",
        "GT_opt",
        "p_opt",
        "p_cnv",
        "compartment_opt",
    ),
    ) -> tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ]:
    """
    Collapse SpaceNumbat expanded rows and build arm posteriors.

    Returns
    -------
    arm_posteriors
        One row per barcode and chromosome arm.

    resolved_joint_post
        One row per barcode and physical consensus segment.

    resolution_log
        Audit log for collapsed multistate rows.
    """
    resolved_joint_post, resolution_log = (
        resolve_multistate_joint_segments(
            joint_post=joint_post,
            segs_consensus=segs_consensus,
            cell_col=cell_col,
            chrom_col=chrom_col,
            start_col=start_col,
            end_col=end_col,
            segment_col=segment_col,
            expanded_state_col=expanded_state_col,
            consensus_segment_col=consensus_segment_col,
            consensus_states_col=consensus_states_col,
            state_cols=state_cols,
            local_state_cols=local_state_cols,
            probability_tolerance=probability_tolerance,
            validate_expanded_states=validate_expanded_states,
        )
    )

    arm_posteriors = build_barcode_arm_posteriors(
        joint_post=resolved_joint_post,
        clone_post=clone_post,
        arm_reference=arm_reference,
        cell_col=cell_col,
        clone_post_barcode_col=clone_post_barcode_col,
        chrom_col=chrom_col,
        start_col=start_col,
        end_col=end_col,
        state_cols=state_cols,
        local_state_cols=local_state_cols,
        probability_tolerance=probability_tolerance,
        validate_nonoverlap=validate_nonoverlap,
        clone_metadata_cols=clone_metadata_cols,
    )

    return {
        "arm_post":arm_posteriors,
        "arm_joint_post":resolved_joint_post,
        "log":resolution_log,
    }



DEFAULT_ARM_MATRIX_VALUE_COLS = (
    *CNA_STATE_COLS,
    "p_gain",
    "p_loss",
    "p_cnv",
    "signed_cna",
)

# Arm-level quantities for which build_barcode_arm_posteriors()
# produces a corresponding *_local column when local posteriors
# are requested.
LOCALIZABLE_ARM_VALUE_COLS = frozenset(
    (
        *CNA_STATE_COLS,
        "p_gain",
        "p_loss",
        "p_cnv",
        "signed_cna",
        "expected_altered_bp",
        "state_entropy",
        "dominant_probability",
    )
)


def arm_probability_matrices(
    barcode_arm_posteriors: pd.DataFrame,
    cell_col: str = "cell",
    arm_col: str = "arm_id",
    arm_order: Sequence[str] | None = None,
    value_cols: Sequence[str] | None = None,
    include_local: bool = False,
    validate_complete: bool = True,
    ) -> dict[str, pd.DataFrame]:
    """
    Create one barcode × chromosome-arm matrix per CNA quantity.

    Primary arm-level quantities are returned by default. HMRF-local
    quantities, such as ``p_amp_local``, are returned only when
    ``include_local=True``.

    Parameters
    ----------
    barcode_arm_posteriors
        Long-format barcode × chromosome-arm table returned as
        ``result["arm_post"]`` by
        :func:`build_spacenumbat_arm_posteriors`.

    cell_col
        Barcode identifier column.

    arm_col
        Chromosome-arm identifier column.

    arm_order
        Optional output arm order.

        If None, arm order is taken from the first occurrence of each arm
        in ``barcode_arm_posteriors``. This preserves the ordering created
        from ``arm_reference``.

        A supplied order may contain a subset of available arms, but it
        cannot contain duplicated or unknown arm identifiers.

    value_cols
        Primary arm-level quantities to convert into matrices.

        If None, the following are used:

            p_neu, p_loh, p_amp, p_del, p_bamp, p_bdel,
            p_gain, p_loss, p_cnv, signed_cna

        Columns ending in ``"_local"`` should not be supplied here.
        Use ``include_local=True`` to add their local counterparts.

    include_local
        If True, add ``*_local`` matrices for every requested quantity
        having a local counterpart in ``LOCALIZABLE_ARM_VALUE_COLS``.

        The local arm quantities must already have been generated by
        ``build_spacenumbat_arm_posteriors`` using ``local_state_cols``.

    validate_complete
        If True, require every requested barcode × arm combination to
        contain a finite value.

        The standard SpaceNumbat arm-level output is complete because
        unreported arm fractions are represented as neutral.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary keyed by arm-level quantity. Each value is a DataFrame
        with barcodes as rows and chromosome arms as columns.

        With ``include_local=False``:

            matrices["p_amp"]
            matrices["p_cnv"]

        With ``include_local=True``:

            matrices["p_amp"]
            matrices["p_amp_local"]
            matrices["p_cnv"]
            matrices["p_cnv_local"]

    Raises
    ------
    KeyError
        If identifier columns, requested primary columns, or requested
        local columns are missing.

    ValueError
        If barcode–arm rows are duplicated, identifiers are missing,
        column specifications contain duplicates, values are non-numeric
        or non-finite, or the requested arm order is invalid.
    """
    if not isinstance(include_local, bool):
        raise TypeError(
            "include_local must be a boolean."
        )

    if not isinstance(validate_complete, bool):
        raise TypeError(
            "validate_complete must be a boolean."
        )

    required_identifier_columns = {
        cell_col,
        arm_col,
    }

    missing = required_identifier_columns.difference(
        barcode_arm_posteriors.columns
    )

    if missing:
        raise KeyError(
            "barcode_arm_posteriors is missing identifier columns: "
            f"{sorted(missing)}"
        )

    if barcode_arm_posteriors[cell_col].isna().any():
        raise ValueError(
            f"barcode_arm_posteriors[{cell_col!r}] contains "
            "missing barcode identifiers."
        )

    if barcode_arm_posteriors[arm_col].isna().any():
        raise ValueError(
            f"barcode_arm_posteriors[{arm_col!r}] contains "
            "missing chromosome-arm identifiers."
        )

    duplicated_pairs = barcode_arm_posteriors.duplicated(
        [cell_col, arm_col],
        keep=False,
    )

    if duplicated_pairs.any():
        examples = barcode_arm_posteriors.loc[
            duplicated_pairs,
            [cell_col, arm_col],
        ].drop_duplicates().head(10)

        raise ValueError(
            "barcode_arm_posteriors must contain at most one row "
            "per barcode and chromosome arm. Duplicated pairs "
            f"include:\n{examples}"
        )

    # Resolve primary quantities
    if value_cols is None:
        primary_value_cols = list(
            DEFAULT_ARM_MATRIX_VALUE_COLS
        )

    else:
        if isinstance(value_cols, (str, bytes)):
            raise TypeError(
                "value_cols must be a sequence of column names, "
                "not a single string."
            )

        primary_value_cols = list(value_cols)

    invalid_value_cols = [
        column
        for column in primary_value_cols
        if not isinstance(column, str)
        or not column.strip()
    ]

    if invalid_value_cols:
        raise TypeError(
            "Every entry in value_cols must be a non-empty string. "
            f"Invalid entries: {invalid_value_cols}"
        )

    duplicated_value_cols = (
        pd.Index(primary_value_cols)[
            pd.Index(primary_value_cols).duplicated(
                keep=False
            )
        ]
        .unique()
        .tolist()
    )

    if duplicated_value_cols:
        raise ValueError(
            "value_cols contains duplicated column names: "
            f"{duplicated_value_cols}"
        )

    explicitly_local = [
        column
        for column in primary_value_cols
        if column.endswith("_local")
    ]

    if explicitly_local:
        raise ValueError(
            "value_cols must contain primary arm quantities only. "
            "Use include_local=True to add local counterparts. "
            f"Local columns found: {explicitly_local}"
        )

    missing_primary = set(
        primary_value_cols
    ).difference(
        barcode_arm_posteriors.columns
    )

    if missing_primary:
        raise KeyError(
            "barcode_arm_posteriors is missing requested primary "
            f"arm quantities: {sorted(missing_primary)}"
        )

    # Add optional local prediction
    local_value_cols: list[str] = []

    if include_local:
        local_value_cols = [
            f"{column}_local"
            for column in primary_value_cols
            if column in LOCALIZABLE_ARM_VALUE_COLS
        ]

        if not local_value_cols:
            raise ValueError(
                "None of the requested value_cols has a recognized "
                "HMRF-local counterpart."
            )

        missing_local = set(
            local_value_cols
        ).difference(
            barcode_arm_posteriors.columns
        )

        if missing_local:
            raise KeyError(
                "Local arm quantities were requested but are absent "
                "from barcode_arm_posteriors: "
                f"{sorted(missing_local)}. Generate them upstream by "
                "passing local_state_cols to "
                "build_spacenumbat_arm_posteriors()."
            )

    requested_value_cols = [
        *primary_value_cols,
        *local_value_cols,
    ]

    # Resolve barcode and arm ordering
    cell_order = (
        barcode_arm_posteriors[cell_col]
        .drop_duplicates()
        .tolist()
    )

    available_arm_order = (
        barcode_arm_posteriors[arm_col]
        .drop_duplicates()
        .tolist()
    )

    if arm_order is None:
        resolved_arm_order = available_arm_order

    else:
        if isinstance(arm_order, (str, bytes)):
            raise TypeError(
                "arm_order must be a sequence of arm identifiers, "
                "not a single string."
            )

        resolved_arm_order = list(arm_order)

        if pd.isna(resolved_arm_order).any():
            raise ValueError(
                "arm_order contains missing arm identifiers."
            )

        duplicated_arms = (
            pd.Index(resolved_arm_order)[
                pd.Index(resolved_arm_order).duplicated(
                    keep=False
                )
            ]
            .unique()
            .tolist()
        )

        if duplicated_arms:
            raise ValueError(
                "arm_order contains duplicated arm identifiers: "
                f"{duplicated_arms}"
            )

        unknown_arms = pd.Index(
            resolved_arm_order
        ).difference(
            pd.Index(available_arm_order)
        )

        if len(unknown_arms) > 0:
            raise ValueError(
                "arm_order contains arms absent from "
                "barcode_arm_posteriors: "
                f"{unknown_arms.tolist()}"
            )

    # Validate values and construct matrices
    selected = barcode_arm_posteriors[
        [
            cell_col,
            arm_col,
            *requested_value_cols,
        ]
    ].copy()

    for value_col in requested_value_cols:
        selected[value_col] = pd.to_numeric(
            selected[value_col],
            errors="raise",
        )

        values = selected[
            value_col
        ].to_numpy(dtype=float)

        if not np.isfinite(values).all():
            bad_rows = np.flatnonzero(
                ~np.isfinite(values)
            )[:10].tolist()

            raise ValueError(
                f"Arm quantity {value_col!r} contains non-finite "
                f"values. Example row positions: {bad_rows}"
            )

    matrices: dict[str, pd.DataFrame] = {}

    for value_col in requested_value_cols:
        matrix = (
            selected.pivot(
                index=cell_col,
                columns=arm_col,
                values=value_col,
            )
            .reindex(
                index=cell_order,
                columns=resolved_arm_order,
            )
        )

        if validate_complete and matrix.isna().any().any():
            missing_positions = (
                matrix.isna()
                .stack()
            )

            missing_positions = (
                missing_positions[
                    missing_positions
                ]
                .index
                .tolist()[:10]
            )

            raise ValueError(
                f"The {value_col!r} arm matrix is incomplete. "
                "Missing barcode–arm combinations include: "
                f"{missing_positions}"
            )

        matrices[value_col] = matrix

    return matrices
    
