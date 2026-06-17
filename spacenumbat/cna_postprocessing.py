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

CNA_STATE_COLS = tuple(f"p_{state}" for state in CNA_STATES)
ALTERED_STATES = CNA_STATES[1:]

DEFAULT_ARM_MATRIX_VALUE_COLS = (
    *CNA_STATE_COLS,
    "p_gain",
    "p_loss",
    "p_cnv",
    "signed_cna",
)


# Generic validation helpers

def _resolve_cna_state_columns(state_cols: Mapping[str, str] | Sequence[str] | None) -> dict[str, str]:
    """Resolve source probability columns for the canonical CNA states."""
    if state_cols is None:
        resolved = dict(zip(CNA_STATES, CNA_STATE_COLS, strict=True))

    elif isinstance(state_cols, Mapping):
        resolved: dict[str, str] = {}

        for raw_state, raw_column in state_cols.items():
            state = str(raw_state).strip().lower()
            if state.startswith("p_"):
                state = state.removeprefix("p_")

            if state not in CNA_STATES:
                raise KeyError(f"Unknown CNA state {raw_state!r}. "
                               f"Expected one of: {list(CNA_STATES)}")
            if state in resolved:
                raise ValueError(f"CNA state {state!r} was specified more than once.")

            resolved[state] = raw_column

        missing_states = set(CNA_STATES).difference(resolved)
        if missing_states:
            raise KeyError("Missing CNA probability-column mappings for states: "
                           f"{sorted(missing_states)}")

    else:
        if isinstance(state_cols, (str, bytes)):
            raise TypeError("state_cols must be a mapping or a sequence of six "
                            "column names, not a single string.")

        source_columns = list(state_cols)
        if len(source_columns) != len(CNA_STATES):
            raise ValueError(f"state_cols must contain exactly {len(CNA_STATES)} "
                             f"columns in this order: {list(CNA_STATES)}")

        resolved = dict(zip(CNA_STATES, source_columns, strict=True))

    normalized: dict[str, str] = {}
    for state in CNA_STATES:
        column = resolved[state]
        if not isinstance(column, str) or not column.strip():
            raise TypeError("Every CNA probability column name must be a non-empty "
                            f"string. Invalid entry for {state!r}: {column!r}")
        normalized[state] = column.strip()

    source_columns = list(normalized.values())
    duplicated = pd.Index(source_columns).duplicated(keep=False)
    if duplicated.any():
        duplicated_columns = (
            pd.Index(source_columns)[duplicated].unique().tolist()
        )
        raise ValueError("Each CNA state must use a distinct source column. "
                         f"Duplicated columns: {duplicated_columns}")

    return normalized


def _normalize_chromosome(series: pd.Series) -> pd.Series:
    """Normalize chromosome labels to 1..22, X, or Y-like strings."""
    missing = series.isna()
    normalized = (
        series.astype("string")
        .str.strip()
        .str.upper()
        .str.replace(r"^CHR", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )
    return normalized.mask(missing, pd.NA)


def _coerce_interval_columns(
    table: pd.DataFrame,
    start_col: str,
    end_col: str,
    table_name: str,
    ) -> pd.DataFrame:
    """Validate and convert genomic interval coordinates."""
    out = table.copy()

    for column in (start_col, end_col):
        if column not in out.columns:
            raise KeyError(f"{table_name} is missing {column!r}.")

        numeric = pd.to_numeric(out[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            raise ValueError(f"{table_name}[{column!r}] contains non-finite coordinates.")

        rounded = np.round(numeric)
        if not np.allclose(numeric, rounded, atol=1e-8, rtol=0):
            raise ValueError(f"{table_name}[{column!r}] contains non-integer coordinates.")

        out[column] = rounded.astype(np.int64)

    if (out[start_col] < 0).any():
        raise ValueError(f"{table_name}[{start_col!r}] contains negative coordinates.")
    if (out[end_col] <= out[start_col]).any():
        raise ValueError(f"Every interval in {table_name} must have {end_col} > {start_col}.")

    return out


def _harmonize_state_columns(
    table: pd.DataFrame,
    state_cols: Mapping[str, str] | Sequence[str] | None,
    table_name: str,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Copy selected source posterior columns to canonical ``p_<state>`` names.

    Source columns are retained. This lets callers use custom input names while
    every downstream operation works on one stable internal schema.
    """
    state_column_map = _resolve_cna_state_columns(state_cols)
    source_columns = set(state_column_map.values())

    missing = source_columns.difference(table.columns)
    if missing:
        raise KeyError(f"{table_name} is missing CNA posterior columns: {sorted(missing)}")

    out = table.copy()

    for state in CNA_STATES:
        source_column = state_column_map[state]
        canonical_column = f"p_{state}"
        source_values = pd.to_numeric(out[source_column], errors="raise")

        if canonical_column in out.columns and canonical_column != source_column:
            existing_values = pd.to_numeric(out[canonical_column], errors="raise")
            if not np.allclose(
                existing_values.to_numpy(dtype=float),
                source_values.to_numpy(dtype=float),
                atol=0.0,
                rtol=0.0,
                equal_nan=True,
            ):
                raise ValueError(f"{table_name} already contains {canonical_column!r}, but "
                                 f"it disagrees with selected source column {source_column!r}.")

        out[canonical_column] = source_values

    return out, state_column_map


def _validate_state_probabilities(
    table: pd.DataFrame,
    probability_tolerance: float,
    table_name: str,
    ) -> pd.DataFrame:
    """Validate and numerically normalize the canonical CNA posterior vector."""
    if probability_tolerance < 0:
        raise ValueError("probability_tolerance must be non-negative.")

    missing = set(CNA_STATE_COLS).difference(table.columns)
    if missing:
        raise KeyError(f"{table_name} is missing canonical posterior columns: {sorted(missing)}")

    out = table.copy()
    if out.empty:
        return out

    for column in CNA_STATE_COLS:
        out[column] = pd.to_numeric(out[column], errors="raise")

    probabilities = out.loc[:, CNA_STATE_COLS].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(f"{table_name} contains non-finite CNA posterior probabilities.")

    if ((probabilities < -probability_tolerance).any() or (probabilities > 1.0 + probability_tolerance).any()):
        raise ValueError(f"{table_name} CNA posterior probabilities must lie in [0, 1].")

    row_sums = probabilities.sum(axis=1)
    
    if not np.allclose(row_sums, 1.0, atol=probability_tolerance, rtol=0):
        raise ValueError(f"{table_name} CNA posterior probabilities do not sum to one. "
                         f"Observed range: {row_sums.min():.6g}–{row_sums.max():.6g}")

    probabilities = np.clip(probabilities, 0.0, 1.0)
    clipped_sums = probabilities.sum(axis=1, keepdims=True)
    if (clipped_sums <= 0).any():
        raise ValueError(f"{table_name} contains a zero-mass posterior vector after clipping.")

    out.loc[:, CNA_STATE_COLS] = probabilities / clipped_sums
    return out


def _canonical_cna_state(value: object) -> str | None:
    """Normalize a CNA-state label."""
    if pd.isna(value):
        return None

    state = str(value).strip().lower()
    aliases = {
        "neutral": "neu",
        "normal": "neu",
        "diploid": "neu",
        "gain": "amp",
        "loss": "del",
    }
    return aliases.get(state, state)


def _parse_consensus_states(value: object) -> list[str]:
    """Parse and deduplicate a comma-separated consensus-state field."""
    if pd.isna(value):
        return []

    states = [
        _canonical_cna_state(item)
        for item in str(value).split(",")
    ]
    states = [state for state in states if state in CNA_STATES]
    return list(dict.fromkeys(states))


# Reference and barcode preparation

def prepare_arm_reference(
    cytoband_arms: pd.DataFrame,
    chrom_col: str = "CHROM",
    start_col: str = "seg_start",
    end_col: str = "seg_end",
    arm_col: str = "seg_id",
    exclude_sex_chromosomes: bool = True,
    exclude_acrocentric_p_arms: bool = False,
    ) -> pd.DataFrame:
    """Prepare and validate a chromosome-arm reference table."""
    
    role_columns = [chrom_col, start_col, end_col, arm_col]
    if len(set(role_columns)) != len(role_columns):
        raise ValueError("chrom_col, start_col, end_col, and arm_col must be distinct.")

    missing = set(role_columns).difference(cytoband_arms.columns)
    if missing:
        raise KeyError(f"Missing chromosome-arm columns: {sorted(missing)}")

    arms = (
        cytoband_arms.loc[:, role_columns]
        .rename(
            columns={
                chrom_col: "CHROM",
                start_col: "arm_start",
                end_col: "arm_end",
                arm_col: "arm_id",
            }).copy()
        )
    
    arms["CHROM"] = _normalize_chromosome(arms["CHROM"])
    if arms["CHROM"].isna().any():
        raise ValueError("The arm reference contains missing chromosome labels.")

    arms["arm_id"] = arms["arm_id"].astype("string").str.strip()
    if arms["arm_id"].isna().any() or (arms["arm_id"] == "").any():
        raise ValueError("The arm reference contains missing arm identifiers.")

    arms = _coerce_interval_columns(
        arms,
        start_col="arm_start",
        end_col="arm_end",
        table_name="cytoband_arms",
    )

    if arms["arm_id"].duplicated().any():
        duplicated = (
            arms.loc[arms["arm_id"].duplicated(keep=False), "arm_id"]
            .drop_duplicates()
            .tolist()
        )
        raise ValueError(f"arm_id must be unique. Duplicates: {duplicated}")

    canonical_chromosomes = {str(chromosome) for chromosome in range(1, 23)} | {"X","Y"}
    invalid_chromosomes = sorted(set(arms["CHROM"].astype(str)).difference(canonical_chromosomes))
    if invalid_chromosomes:
        raise ValueError("The arm reference contains non-canonical chromosomes: "
                         f"{invalid_chromosomes}")

    arm_pattern = r"^(?:[1-9]|1[0-9]|2[0-2]|X|Y)[pq]$"
    malformed = ~arms["arm_id"].str.match(arm_pattern, na=False)
    if malformed.any():
        raise ValueError("The arm reference contains malformed arm identifiers: "
                         f"{arms.loc[malformed, 'arm_id'].tolist()}")

    arm_chromosome = arms["arm_id"].str[:-1]
    inconsistent = arm_chromosome != arms["CHROM"].astype("string")
    if inconsistent.any():
        examples = arms.loc[inconsistent, ["CHROM", "arm_id"]].head(10)
        raise ValueError("arm_id chromosome labels disagree with CHROM:\n"
                         f"{examples}")

    arms["is_sex_chromosome"] = arms["CHROM"].isin(["X", "Y"])
    arms["is_acrocentric_p"] = arms["arm_id"].isin(ACROCENTRIC_P_ARMS)

    if exclude_sex_chromosomes:
        arms = arms.loc[~arms["is_sex_chromosome"]].copy()
    if exclude_acrocentric_p_arms:
        arms = arms.loc[~arms["is_acrocentric_p"]].copy()

    arms["arm_length"] = arms["arm_end"] - arms["arm_start"]

    chromosome_order = {str(chromosome): chromosome for chromosome in range(1, 23)}
    chromosome_order.update({"X": 23, "Y": 24})

    arms["_chromosome_order"] = arms["CHROM"].map(chromosome_order)
    arms["_arm_order"] = arms["arm_id"].str[-1].map({"p": 0, "q": 1})

    return (
        arms.sort_values(
            ["_chromosome_order", "_arm_order"],
            kind="mergesort",
        )
        .drop(columns=["_chromosome_order", "_arm_order"])
        .reset_index(drop=True)
    )


def _prepare_clone_post_barcodes(
    clone_post: pd.DataFrame,
    output_cell_col: str,
    clone_post_barcode_col: str | None,
    ) -> pd.DataFrame:
    """Return a clone table with one canonical barcode column."""
    clone_table = clone_post.copy()

    # Native SpaceNumbat clone_post contains a ``cell`` column. Prefer it
    # automatically when the caller does not provide an explicit source.
    if clone_post_barcode_col is None and output_cell_col in clone_post.columns:
        clone_post_barcode_col = output_cell_col

    if clone_post_barcode_col is None:
        barcode_values = clone_post.index.to_numpy(copy=True)
        barcode_source = "clone_post.index"
    else:
        if clone_post_barcode_col not in clone_post.columns:
            raise KeyError("clone_post is missing barcode column "
                           f"{clone_post_barcode_col!r}.")
        barcode_values = clone_post[clone_post_barcode_col].to_numpy(copy=True)
        barcode_source = f"clone_post[{clone_post_barcode_col!r}]"

    if pd.isna(barcode_values).any():
        raise ValueError(f"{barcode_source} contains missing barcode identifiers.")

    duplicated = pd.Index(barcode_values).duplicated(keep=False)
    if duplicated.any():
        duplicated_barcodes = (pd.Index(barcode_values)[duplicated].unique().tolist())
        raise ValueError("clone_post must contain one row per barcode. "
                         f"Duplicated barcodes include: {duplicated_barcodes[:10]}")

    clone_table = clone_table.reset_index(drop=True)

    if output_cell_col in clone_table.columns:
        existing = clone_table[output_cell_col].to_numpy(copy=False)
        if clone_post_barcode_col != output_cell_col and not np.array_equal(existing, barcode_values):
            raise ValueError(f"clone_post[{output_cell_col!r}] disagrees with "
                             f"the selected barcode source {barcode_source}.")
        clone_table = clone_table.drop(columns=output_cell_col)

    clone_table.insert(0, output_cell_col, barcode_values)
    return clone_table


# SpaceNumbat expanded-segment resolution

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
    probability_tolerance: float = 1e-8,
    validate_expanded_states: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse rows generated by ``operations.expand_states()``.

    SpaceNumbat duplicates a physical multi-state segment into one row per
    candidate event. The complete ``p_*`` posterior vector is copied to every
    expanded row, while ``seg``, ``cnv_state``, ``p_cnv``, ``p_n``, and
    ``Z_cnv`` become event-specific. This function verifies the copied full
    posterior vectors, retains one vector unchanged, restores the physical
    segment identifier, and recomputes only ``p_cnv`` and ``p_n``.

    Existing ``cnv_state_map`` and ``cnv_state_mle`` values are preserved.
    """
    required_joint = {
        cell_col,
        chrom_col,
        start_col,
        end_col,
        segment_col,
    }
    missing = required_joint.difference(joint_post.columns)
    if missing:
        raise KeyError(f"Missing joint_post columns: {sorted(missing)}")

    required_consensus = {
        chrom_col,
        start_col,
        end_col,
        consensus_segment_col,
    }
    missing = required_consensus.difference(segs_consensus.columns)
    if missing:
        raise KeyError(f"Missing segs_consensus columns: {sorted(missing)}")

    joint, _ = _harmonize_state_columns(
        joint_post,
        state_cols,
        table_name="joint_post",
    )
    consensus = segs_consensus.copy()

    joint[chrom_col] = _normalize_chromosome(joint[chrom_col])
    consensus[chrom_col] = _normalize_chromosome(consensus[chrom_col])

    if joint[chrom_col].isna().any():
        raise ValueError(f"joint_post[{chrom_col!r}] contains missing chromosome labels.")
    if consensus[chrom_col].isna().any():
        raise ValueError(f"segs_consensus[{chrom_col!r}] contains missing chromosome labels.")

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
    joint = _validate_state_probabilities(
        joint,
        probability_tolerance=probability_tolerance,
        table_name="joint_post",
    )

    genomic_key = [chrom_col, start_col, end_col]
    duplicated_consensus = consensus.duplicated(genomic_key, keep=False)
    if duplicated_consensus.any():
        examples = consensus.loc[
            duplicated_consensus,
            [*genomic_key, consensus_segment_col],
        ].head(10)
        raise ValueError("segs_consensus must contain one row per physical genomic "
                         f"interval. Duplicates were found:\n{examples}")

    consensus_lookup = {
        tuple(row[column] for column in genomic_key): row
        for _, row in consensus.iterrows()
    }

    grouping_key = [cell_col, chrom_col, start_col, end_col]
    consistent_columns = [
        column
        for column in (
            "cnv_state_mle",
            "cnv_state_map",
            "hmrf_iterations",
            "hmrf_converged",
        )
        if column in joint.columns
    ]

    resolved_rows: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []

    for group_key, group in joint.groupby(grouping_key,
                                          sort=False,
                                          observed=True,
                                          dropna=False):
        barcode, chromosome, segment_start, segment_end = group_key
        consensus_key = (chromosome, segment_start, segment_end)
        consensus_row = consensus_lookup.get(consensus_key)
        resolved = group.iloc[0].copy()

        if len(group) == 1:
            state = (
                _canonical_cna_state(resolved[expanded_state_col])
                if expanded_state_col in resolved.index
                else None
            )
            resolved["resolved_cnv_states"] = (state if state in CNA_STATES else "")
            resolved["n_joint_rows_collapsed"] = 1
            resolved_rows.append(resolved)
            continue

        if consensus_row is None:
            raise ValueError("A duplicated joint_post interval could not be matched "
                             "to segs_consensus:\n"
                             f"cell={barcode!r}, interval={consensus_key}")

        allowed_states = (
            _parse_consensus_states(consensus_row[consensus_states_col])
            if consensus_states_col in consensus.columns
            else []
        )

        observed_states: list[str] = []
        if expanded_state_col in group.columns:
            observed_states = [
                state
                for state in (
                    _canonical_cna_state(value)
                    for value in group[expanded_state_col]
                )
                if state in CNA_STATES
            ]
            observed_states = list(dict.fromkeys(observed_states))

        if not allowed_states:
            allowed_states = observed_states

        if (
            validate_expanded_states
            and allowed_states
            and set(observed_states) != set(allowed_states)
        ):
            raise ValueError("Expanded SpaceNumbat rows do not match the consensus "
                             "state list:\n"
                             f"cell={barcode!r}, interval={consensus_key}, "
                             f"observed={observed_states}, consensus={allowed_states}")

        probability_matrix = group.loc[:, CNA_STATE_COLS].to_numpy(dtype=float)
        reference_vector = probability_matrix[0]
        if not np.allclose(probability_matrix,
                           reference_vector[None, :],
                           atol=probability_tolerance,
                           rtol=0):
            raise ValueError("Rows representing the same physical segment contain "
                             "different complete posterior vectors. This is not expected "
                             "from SpaceNumbat operations.expand_states():\n"
                             f"cell={barcode!r}, interval={consensus_key}")
        resolved.loc[list(CNA_STATE_COLS)] = reference_vector

        for column in consistent_columns:
            values = pd.unique(group[column].dropna())
            if len(values) > 1:
                raise ValueError(f"Expanded rows disagree in {column!r} for "
                                 f"cell={barcode!r}, interval={consensus_key}: "
                                 f"{values.tolist()}")
            resolved[column] = values[0] if len(values) == 1 else pd.NA

        physical_segment = consensus_row[consensus_segment_col]
        if pd.notna(physical_segment):
            resolved[segment_col] = physical_segment

        # These fields are event-specific after expand_states().
        if expanded_state_col in resolved.index:
            resolved[expanded_state_col] = pd.NA
        if "Z_cnv" in resolved.index:
            resolved["Z_cnv"] = np.nan
        if "seg_label" in resolved.index:
            resolved["seg_label"] = str(resolved[segment_col])

        resolved["resolved_cnv_states"] = ",".join(allowed_states)
        resolved["n_joint_rows_collapsed"] = len(group)
        resolved_rows.append(resolved)

        audit_rows.append(
            {
                cell_col: barcode,
                chrom_col: chromosome,
                start_col: segment_start,
                end_col: segment_end,
                "input_rows": len(group),
                "input_segment_labels": ",".join(group[segment_col].astype(str).drop_duplicates()),
                "input_states": ",".join(observed_states),
                "consensus_segment": physical_segment,
                "consensus_states": ",".join(allowed_states),
                "posterior_validated": True,
            }
        )

    resolved_joint_post = pd.DataFrame(resolved_rows).reset_index(drop=True)

    if resolved_joint_post.duplicated(grouping_key, keep=False).any():
        raise RuntimeError("Multistate resolution did not produce unique "
                           "barcode–interval rows.")

    resolved_joint_post = _validate_state_probabilities(resolved_joint_post,
                                                        probability_tolerance=probability_tolerance,
                                                        table_name="resolved_joint_post")

    resolved_joint_post["p_n"] = resolved_joint_post["p_neu"]
    resolved_joint_post["p_cnv"] = 1.0 - resolved_joint_post["p_neu"]

    return resolved_joint_post, pd.DataFrame(audit_rows)


# Chromosome-arm projection

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
    Project one physical-segment posterior vector per barcode onto arms.

    Returned ``p_*`` values are arm-length-weighted expected state
    occupancies. Unreported arm fractions are treated as neutral.
    """
    coordinate_columns = [cell_col, chrom_col, start_col, end_col]
    if len(set(coordinate_columns)) != len(coordinate_columns):
        raise ValueError("cell_col, chrom_col, start_col, and end_col must be distinct.")

    missing = set(coordinate_columns).difference(joint_post.columns)
    if missing:
        raise KeyError(f"Missing joint_post columns: {sorted(missing)}")

    required_arm_columns = {
        "CHROM",
        "arm_id",
        "arm_start",
        "arm_end",
        "arm_length",
        "is_sex_chromosome",
        "is_acrocentric_p",
    }
    missing = required_arm_columns.difference(arm_reference.columns)
    if missing:
        raise KeyError(f"Missing arm-reference columns: {sorted(missing)}")

    clone_table = _prepare_clone_post_barcodes(clone_post,
                                               output_cell_col=cell_col,
                                               clone_post_barcode_col=clone_post_barcode_col)
    barcodes = clone_table.loc[:, [cell_col]].copy()

    harmonized, _ = _harmonize_state_columns(joint_post,
                                             state_cols,
                                             table_name="joint_post")

    segments = (harmonized.loc[:,
                               [cell_col,
                                chrom_col,
                                start_col,
                                end_col, 
                                *CNA_STATE_COLS],].rename(columns={chrom_col: "CHROM",
                                                                   start_col: "seg_start",
                                                                   end_col: "seg_end"}).copy())

    if segments[cell_col].isna().any():
        raise ValueError(f"joint_post[{cell_col!r}] contains missing barcodes.")

    known_barcodes = pd.Index(barcodes[cell_col])
    unknown_barcodes = pd.Index(
        segments[cell_col].drop_duplicates()
    ).difference(known_barcodes)
    if len(unknown_barcodes) > 0:
        raise ValueError("joint_post contains barcodes absent from clone_post. Examples: "
                         f"{unknown_barcodes[:10].tolist()}")

    segments["CHROM"] = _normalize_chromosome(segments["CHROM"])
    if segments["CHROM"].isna().any():
        raise ValueError("joint_post contains missing chromosome labels.")

    segments = _coerce_interval_columns(segments,
                                        start_col="seg_start",
                                        end_col="seg_end",
                                        table_name="joint_post")
    segments = _validate_state_probabilities(segments,
                                             probability_tolerance=probability_tolerance,
                                             table_name="joint_post")

    arms = arm_reference.copy()
    arms["CHROM"] = _normalize_chromosome(arms["CHROM"])
    if arms["CHROM"].isna().any():
        raise ValueError("arm_reference contains missing chromosome labels.")

    arms = _coerce_interval_columns(arms,
                                    start_col="arm_start",
                                    end_col="arm_end",
                                    table_name="arm_reference")

    if arms["arm_id"].duplicated().any():
        duplicated = arms.loc[arms["arm_id"].duplicated(keep=False), "arm_id"].drop_duplicates().tolist()
        
        raise ValueError(f"arm_reference contains duplicated arm_id values: {duplicated}")

    expected_arm_length = arms["arm_end"] - arms["arm_start"]
    supplied_arm_length = pd.to_numeric(arms["arm_length"], errors="raise").to_numpy(dtype=np.int64)
    if not np.array_equal(expected_arm_length.to_numpy(dtype=np.int64), supplied_arm_length):
        raise ValueError("arm_reference.arm_length must equal arm_end - arm_start.")
    arms["arm_length"] = expected_arm_length.astype(np.int64)

    selected_chromosomes = set(arms["CHROM"])
    segments = segments.loc[segments["CHROM"].isin(selected_chromosomes)].copy()

    if validate_nonoverlap and not segments.empty:
        ordered = segments.sort_values([cell_col, "CHROM", "seg_start", "seg_end"],
                                       kind="mergesort")
        previous_end = ordered.groupby([cell_col, "CHROM"],
                                       sort=False,
                                       observed=True)["seg_end"].shift()
        overlaps_previous = ordered["seg_start"] < previous_end

        if overlaps_previous.any():
            examples = ordered.loc[
                overlaps_previous,
                [cell_col, "CHROM", "seg_start", "seg_end"],
            ].head(10)
            raise ValueError("Overlapping physical CNA segments were found within a "
                             f"barcode and chromosome:\n{examples}")

    arm_columns = [
        "CHROM",
        "arm_id",
        "arm_start",
        "arm_end",
        "arm_length",
        "is_sex_chromosome",
        "is_acrocentric_p",
    ]

    background = barcodes.merge(arms.loc[:, arm_columns], how="cross")
    background["p_neu"] = 1.0
    for state in ALTERED_STATES:
        background[f"p_{state}"] = 0.0
    background["reported_segment_fraction"] = 0.0

    if not segments.empty:
        overlaps = segments.merge(arms.loc[:, arm_columns],
                                  on="CHROM",
                                  how="inner",
                                  validate="many_to_many")

        overlaps["overlap_start"] = np.maximum(overlaps["seg_start"], overlaps["arm_start"],)
        overlaps["overlap_end"] = np.minimum(overlaps["seg_end"], overlaps["arm_end"])
        overlaps["overlap_bp"] = (overlaps["overlap_end"] - overlaps["overlap_start"]).clip(lower=0)
        overlaps = overlaps.loc[overlaps["overlap_bp"] > 0].copy()
        overlaps["overlap_fraction"] = overlaps["overlap_bp"] / overlaps["arm_length"]

        overlaps["_neutral_delta"] = overlaps["overlap_fraction"] * (overlaps["p_neu"] - 1.0)

        mass_columns: list[str] = []
        for state in ALTERED_STATES:
            mass_column = f"_p_{state}_mass"
            overlaps[mass_column] = overlaps["overlap_fraction"] * overlaps[f"p_{state}"]
            mass_columns.append(mass_column)

        aggregation_spec: dict[str, tuple[str, str]] = {"_reported_segment_fraction": ("overlap_fraction", "sum"),
                                                        "_neutral_delta": ("_neutral_delta", "sum"),}
        aggregation_spec.update({mass_column: (mass_column, "sum") for mass_column in mass_columns})

        arm_mass = overlaps.groupby([cell_col, "arm_id"], observed=True, as_index=False, sort=False).agg(**aggregation_spec)

        background = background.merge(arm_mass, on=[cell_col, "arm_id"], how="left", validate="one_to_one")
        background["reported_segment_fraction"] = background["_reported_segment_fraction"].fillna(0.0)
        background["p_neu"] += background["_neutral_delta"].fillna(0.0)

        for state, mass_column in zip(ALTERED_STATES, mass_columns, strict=True):
            background[f"p_{state}"] += background[mass_column].fillna(0.0)

        background = background.drop(columns=["_reported_segment_fraction", "_neutral_delta", *mass_columns])

    if (background["reported_segment_fraction"] > 1.0 + probability_tolerance).any():
        raise ValueError("Reported segments cover an arm more than once for at least "
                         "one barcode.")

    arm_probabilities = background.loc[:, CNA_STATE_COLS].to_numpy(dtype=float)
    if ((arm_probabilities < -probability_tolerance).any() or (arm_probabilities > 1.0 + probability_tolerance).any()):
        raise ValueError("Arm projection produced CNA probabilities outside [0, 1].")
        
    if not np.allclose(arm_probabilities.sum(axis=1), 1.0, atol=probability_tolerance, rtol=0):
        raise ValueError("Projected chromosome-arm state probabilities do not sum to one.")

    arm_probabilities = np.clip(arm_probabilities, 0.0, 1.0)
    arm_probabilities /= arm_probabilities.sum(axis=1, keepdims=True)
    background.loc[:, CNA_STATE_COLS] = arm_probabilities

    background["p_gain"] = background["p_amp"] + background["p_bamp"]
    background["p_loss"] = background["p_del"] + background["p_bdel"]
    background["p_cnv"] = 1.0 - background["p_neu"]
    background["signed_cna"] = background["p_gain"] - background["p_loss"]
    background["expected_altered_bp"] = background["p_cnv"] * background["arm_length"]

    probabilities = background.loc[:, CNA_STATE_COLS].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.where(probabilities > 0, probabilities * np.log(probabilities), 0.0).sum(axis=1)
    background["state_entropy"] = entropy / np.log(len(CNA_STATES))

    dominant_index = probabilities.argmax(axis=1)
    state_names = np.asarray(CNA_STATES, dtype=object)
    background["dominant_state"] = state_names[dominant_index]
    background["dominant_probability"] = probabilities[np.arange(len(background)), dominant_index]

    metadata_columns = [column for column in clone_metadata_cols if column in clone_table.columns and column != cell_col]
    metadata = clone_table.loc[:, [cell_col, *metadata_columns]].copy()
    if "p_cnv" in metadata.columns:
        metadata = metadata.rename(columns={"p_cnv": "clone_p_cnv"})

    background = background.merge(metadata, on=cell_col, how="left", validate="many_to_one")

    arm_order = {arm_id: position for position, arm_id in enumerate(arms["arm_id"])}
    barcode_order = {barcode: position for position, barcode in enumerate(barcodes[cell_col])}
    background["_arm_order"] = background["arm_id"].map(arm_order)
    background["_barcode_order"] = background[cell_col].map(barcode_order)

    return (background.sort_values(["_barcode_order","_arm_order"],
                                   kind="mergesort").drop(columns=["_barcode_order", 
                                                                   "_arm_order"]).reset_index(drop=True))


def build_spacenumbat_arm_posteriors(
    joint_post: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    clone_post: pd.DataFrame,
    arm_reference: pd.DataFrame,
    state_cols: Mapping[str, str] | Sequence[str] | None = None,
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
    ) -> dict[str, pd.DataFrame]:
    """Collapse expanded SpaceNumbat rows and build arm-level posteriors."""
    resolved_joint_post, resolution_log = resolve_multistate_joint_segments(
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
        probability_tolerance=probability_tolerance,
        validate_expanded_states=validate_expanded_states,
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
        # The resolver has already harmonized custom source names.
        state_cols=CNA_STATE_COLS,
        probability_tolerance=probability_tolerance,
        validate_nonoverlap=validate_nonoverlap,
        clone_metadata_cols=clone_metadata_cols,
    )

    return {
        "arm_post": arm_posteriors,
        "arm_joint_post": resolved_joint_post,
        "log": resolution_log,
    }


# Matrix export

def arm_probability_matrices(
    barcode_arm_posteriors: pd.DataFrame,
    cell_col: str = "cell",
    arm_col: str = "arm_id",
    arm_order: Sequence[str] | None = None,
    value_cols: Sequence[str] | None = None,
    validate_complete: bool = True,
    ) -> dict[str, pd.DataFrame]:
    """Create one barcode × chromosome-arm matrix per requested quantity."""
    required_identifiers = {cell_col, arm_col}
    missing = required_identifiers.difference(barcode_arm_posteriors.columns)
    if missing:
        raise KeyError(
            "barcode_arm_posteriors is missing identifier columns: "
            f"{sorted(missing)}"
        )

    if barcode_arm_posteriors[cell_col].isna().any():
        raise ValueError(
            f"barcode_arm_posteriors[{cell_col!r}] contains missing barcodes."
        )
    if barcode_arm_posteriors[arm_col].isna().any():
        raise ValueError(
            f"barcode_arm_posteriors[{arm_col!r}] contains missing arms."
        )

    duplicated_pairs = barcode_arm_posteriors.duplicated(
        [cell_col, arm_col],
        keep=False,
    )
    if duplicated_pairs.any():
        examples = (
            barcode_arm_posteriors.loc[
                duplicated_pairs,
                [cell_col, arm_col],
            ]
            .drop_duplicates()
            .head(10)
        )
        raise ValueError(
            "barcode_arm_posteriors must contain one row per barcode "
            f"and arm. Duplicated pairs include:\n{examples}"
        )

    if value_cols is None:
        resolved_value_cols = list(DEFAULT_ARM_MATRIX_VALUE_COLS)
    else:
        if isinstance(value_cols, (str, bytes)):
            raise TypeError(
                "value_cols must be a sequence of column names, not a string."
            )
        resolved_value_cols = list(value_cols)

    invalid_columns = [
        column
        for column in resolved_value_cols
        if not isinstance(column, str) or not column.strip()
    ]
    if invalid_columns:
        raise TypeError(
            "Every value_cols entry must be a non-empty string. "
            f"Invalid entries: {invalid_columns}"
        )

    duplicated_values = pd.Index(resolved_value_cols).duplicated(keep=False)
    if duplicated_values.any():
        duplicates = (
            pd.Index(resolved_value_cols)[duplicated_values]
            .unique()
            .tolist()
        )
        raise ValueError(
            f"value_cols contains duplicated column names: {duplicates}"
        )

    missing = set(resolved_value_cols).difference(
        barcode_arm_posteriors.columns
    )
    if missing:
        raise KeyError(
            "barcode_arm_posteriors is missing requested arm quantities: "
            f"{sorted(missing)}"
        )

    cell_order = (
        barcode_arm_posteriors[cell_col].drop_duplicates().tolist()
    )
    available_arm_order = (
        barcode_arm_posteriors[arm_col].drop_duplicates().tolist()
    )

    if arm_order is None:
        resolved_arm_order = available_arm_order
    else:
        if isinstance(arm_order, (str, bytes)):
            raise TypeError(
                "arm_order must be a sequence of arm identifiers, not a string."
            )
        resolved_arm_order = list(arm_order)

        if pd.isna(resolved_arm_order).any():
            raise ValueError("arm_order contains missing arm identifiers.")

        duplicated_arms = pd.Index(resolved_arm_order).duplicated(keep=False)
        if duplicated_arms.any():
            duplicates = (
                pd.Index(resolved_arm_order)[duplicated_arms]
                .unique()
                .tolist()
            )
            raise ValueError(
                f"arm_order contains duplicated arm identifiers: {duplicates}"
            )

        unknown_arms = pd.Index(resolved_arm_order).difference(
            pd.Index(available_arm_order)
        )
        if len(unknown_arms) > 0:
            raise ValueError(
                "arm_order contains arms absent from the input table: "
                f"{unknown_arms.tolist()}"
            )

    selected = barcode_arm_posteriors.loc[
        :,
        [cell_col, arm_col, *resolved_value_cols],
    ].copy()

    for value_col in resolved_value_cols:
        selected[value_col] = pd.to_numeric(
            selected[value_col],
            errors="raise",
        )
        values = selected[value_col].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            bad_rows = np.flatnonzero(~np.isfinite(values))[:10].tolist()
            raise ValueError(
                f"Arm quantity {value_col!r} contains non-finite values. "
                f"Example row positions: {bad_rows}"
            )

    matrices: dict[str, pd.DataFrame] = {}
    for value_col in resolved_value_cols:
        matrix = (
            selected.pivot(
                index=cell_col,
                columns=arm_col,
                values=value_col,
            )
            .reindex(index=cell_order, columns=resolved_arm_order)
        )

        if validate_complete and matrix.isna().any().any():
            missing_positions = (
                matrix.isna().stack().loc[lambda values: values]
                .index.tolist()[:10]
            )
            raise ValueError(
                f"The {value_col!r} arm matrix is incomplete. Missing "
                f"barcode–arm combinations include: {missing_positions}"
            )

        matrices[value_col] = matrix

    return matrices
    
