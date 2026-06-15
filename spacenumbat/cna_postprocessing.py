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

import natsort


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



def _harmonize_cna_probability_columns(
    table: pd.DataFrame,
    state_cols: Mapping[str, str] | Sequence[str] | None,
    table_name: str,
    ) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Rename source CNA posterior columns to canonical internal names.

    The returned table always contains:

        p_neu, p_loh, p_amp, p_del, p_bamp, p_bdel
    """
    state_column_map = _resolve_cna_state_columns(state_cols)

    source_columns = list(state_column_map.values())

    missing = set(source_columns).difference(table.columns)

    if missing:
        raise KeyError(
            f"{table_name} is missing CNA probability columns: "
            f"{sorted(missing)}"
        )

    source_column_set = set(source_columns)
    rename_map: dict[str, str] = {}

    for state, source_column in state_column_map.items():
        canonical_column = f"p_{state}"

        # Avoid silently creating duplicate canonical columns.
        if (
            source_column != canonical_column
            and canonical_column in table.columns
            and canonical_column not in source_column_set
        ):
            raise ValueError(
                f"Cannot rename {source_column!r} to "
                f"{canonical_column!r} in {table_name}: the canonical "
                "column already exists and is not part of state_cols."
            )

        rename_map[source_column] = canonical_column

    harmonized = table.rename(columns=rename_map).copy()

    duplicated_columns = (
        harmonized.columns[
            harmonized.columns.duplicated(keep=False)
        ]
        .unique()
        .tolist()
    )

    if duplicated_columns:
        raise ValueError(
            f"Column harmonization created duplicate columns in "
            f"{table_name}: {duplicated_columns}"
        )

    return harmonized, state_column_map



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
    """
    Parse strings such as:
        'bdel,del'
        'amp'
        'loh'
    """
    if pd.isna(value):
        return []

    states = [
        _canonical_cna_state(state)
        for state in str(value).split(",")
    ]

    return [
        state
        for state in states
        if state in {*ALTERED_STATES, "neu"}
    ]


def resolve_multistate_joint_segments(
    joint_post: pd.DataFrame,
    segs_consensus: pd.DataFrame,
    cell_col: str = "cell",
    chrom_col: str = "CHROM",
    start_col: str = "seg_start",
    end_col: str = "seg_end",
    segment_col: str = "seg",
    consensus_segment_col: str = "seg_cons",
    consensus_states_col: str = "cnv_states",
    consensus_state_col: str = "cnv_state_post",
    state_cols: Mapping[str, str] | Sequence[str] | None = None,
    consensus_state_cols: Mapping[str, str] | Sequence[str] | None = None,
    mle_score_cols: Mapping[str, str] | Sequence[str] | None = None,
    resolution: str = "joint",
    probability_tolerance: float = 1e-8,
    mle_score_tolerance: float = 1e-8,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse alternative state-specific rows representing one consensus
    genomic segment.

    Parameters
    ----------
    joint_post
        Barcode-by-segment posterior table. Some genomic intervals may
        appear more than once because alternative states were retained,
        for example ``9a_del`` and ``9a_bdel``.

    segs_consensus
        Consensus segmentation table. Multi-state segments are described
        by columns such as:

            seg_cons = "9a"
            cnv_states = "bdel,del"
            n_states = 2

    resolution
        How to distribute the altered posterior mass among the states
        allowed by ``segs_consensus``.

        ``"joint"``
            Recommended. Average the barcode-level posterior vectors from
            joint_post, restrict them to the consensus-allowed states, and
            renormalize. This preserves barcode-specific evidence.

        ``"consensus"``
            Use the state proportions in segs_consensus, such as
            p_del=0.5 and p_bdel=0.5. This gives every barcode the same
            relative split among the allowed altered states, while retaining
            its barcode-specific p_neu.

    state_cols
        Posterior columns in joint_post.
        
    consensus_state_cols
        Posterior columns in segs_consensus used by the consensus fallback.
        If None, canonical p_* names are expected.
    
    mle_score_cols
        Per-state likelihood or log-likelihood columns in joint_post.
        Larger values must indicate a better likelihood.

    Returns
    -------
    resolved_joint_post
        One row per barcode and physical consensus segment.

    resolution_log
        Audit table describing every collapsed multi-state group.

    Notes
    -----
    Coordinates are used without conversion.
    """
    if resolution not in {"joint", "consensus"}:
        raise ValueError(
            "resolution must be 'joint' or 'consensus'."
        )

    joint, state_column_map = _harmonize_cna_probability_columns(
    joint_post,
    state_cols,
    table_name="joint_post",
    )
    
    canonical_state_cols = list(CNA_STATE_COLS)
    
    joint_required = {
        cell_col,
        chrom_col,
        start_col,
        end_col,
    }
    
    missing = joint_required.difference(joint.columns)
    
    if missing:
        raise KeyError(f"Missing joint_post columns: {sorted(missing)}")
    
    consensus = segs_consensus.copy()

    consensus_required = {chrom_col,
                          start_col,
                          end_col}

    missing = consensus_required.difference(consensus.columns)

    if missing:
        raise KeyError(f"Missing segs_consensus columns: {sorted(missing)}")
        
    consensus_probability_map = _resolve_cna_state_columns(consensus_state_cols)
    
    joint[chrom_col] = _normalize_chromosome(joint[chrom_col])
    consensus[chrom_col] = _normalize_chromosome(consensus[chrom_col])

    for table in (joint, consensus):
        for column in (start_col, end_col):
            values = pd.to_numeric(table[column], errors="raise")

            if not np.allclose(values, np.round(values)):
                raise ValueError(f"{column!r} contains non-integer coordinates.")

            table[column] = np.round(values).astype(np.int64)

    for column in state_cols:
        joint[column] = pd.to_numeric(joint[column], errors="raise")

    probabilities = joint[state_cols].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError("joint_post contains non-finite posterior probabilities.")

    row_sums = probabilities.sum(axis=1)

    if not np.allclose(
        row_sums,
        1.0,
        atol=1e-5,
        rtol=0,
    ):
        raise ValueError(
            "joint_post state probabilities do not sum to one. "
            f"Observed range: {row_sums.min():.6g}–"
            f"{row_sums.max():.6g}"
        )

    # Exact genomic interval used to match joint_post to consensus.
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
        examples = consensus.loc[duplicated_consensus, genomic_key,].head()

        raise ValueError("segs_consensus must contain one row per physical "
                         "genomic interval. Duplicates were found:\n"
                         f"{examples}")

    consensus_lookup = {
        tuple(row[column] for column in genomic_key): row
        for _, row in consensus.iterrows()
    }

    grouping_key = [
        cell_col,
        chrom_col,
        start_col,
        end_col,
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
        # Non-duplicated segment: preserve the original row.
        if len(group) == 1:
            resolved_rows.append(group.iloc[0].copy())
            continue

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

        if consensus_row is None:
            raise ValueError(
                "A duplicated joint_post interval could not be matched "
                "exactly to segs_consensus:\n"
                f"cell={barcode!r}, CHROM={chromosome!r}, "
                f"seg_start={segment_start}, seg_end={segment_end}"
            )

        allowed_states = []

        if consensus_states_col in consensus.columns:
            allowed_states = _parse_consensus_states(
                consensus_row[consensus_states_col]
            )

        if not allowed_states and (consensus_state_col in consensus.columns):
            consensus_state = _canonical_cna_state(consensus_row[consensus_state_col])

            if consensus_state is not None:
                allowed_states = [consensus_state]

        # Neutral is handled separately. Alternative duplicated rows
        # normally represent altered states.
        allowed_altered_states = [
            state
            for state in allowed_states
            if state in ALTERED_STATES
        ]

        if not allowed_altered_states:
            raise ValueError(
                "The matching consensus segment does not define any "
                "valid altered states:\n"
                f"cell={barcode!r}, interval={consensus_key}, "
                f"states={allowed_states}"
            )

        # Duplicate rows should describe the same probability of the
        # segment being neutral. Small numerical differences are allowed.
        neutral_values = group["p_neu"].to_numpy(
            dtype=float
        )

        p_neu = float(neutral_values.mean())
        altered_mass = 1.0 - p_neu

        if resolution == "joint":
            # Average the duplicated posterior rows.
            altered_evidence = {
                state: float(
                    group[f"p_{state}"].mean()
                )
                for state in allowed_altered_states
            }

            evidence_sum = sum(
                altered_evidence.values()
            )

            if evidence_sum > probability_tolerance:
                state_weights = {
                    state: value / evidence_sum
                    for state, value
                    in altered_evidence.items()
                }

            else:
                state_weights = {}

        else:
            state_weights = {}

        # Fallback, or requested consensus mode:
        # use state proportions from segs_consensus.
        if not state_weights:
            consensus_evidence = {}

            for state in allowed_altered_states:
                probability_col = consensus_probability_map[state]
                
                if probability_col in consensus.columns:
                    value = consensus_row[probability_col]

                    if pd.notna(value):
                        consensus_evidence[state] = float(value)

            evidence_sum = sum(consensus_evidence.values())

            if evidence_sum > probability_tolerance:
                state_weights = {state: value / evidence_sum
                                 for state, value
                                 in consensus_evidence.items()}

            else:
                # Last-resort equal weighting over consensus states.
                equal_weight = (
                    1.0 / len(allowed_altered_states)
                )

                state_weights = {
                    state: equal_weight
                    for state in allowed_altered_states
                }

        resolved = group.iloc[0].copy()

        for column in state_cols:
            resolved[column] = 0.0

        resolved["p_neu"] = p_neu

        for state, weight in state_weights.items():
            resolved[f"p_{state}"] = (
                altered_mass * weight
            )

        # Replace alternative state-specific label, such as
        # 9a_del/9a_bdel, with the physical consensus segment.
        if (
            consensus_segment_col
            in consensus.columns
            and pd.notna(
                consensus_row[consensus_segment_col]
            )
        ):
            resolved[segment_col] = consensus_row[
                consensus_segment_col
            ]

        # Useful explicit annotation for auditing.
        resolved["resolved_cnv_states"] = ",".join(
            allowed_altered_states
        )
        resolved["n_joint_rows_collapsed"] = len(group)

        # Recompute common derived fields when they exist.
        if "p_cnv" in resolved.index:
            resolved["p_cnv"] = 1.0 - p_neu

        if "p_n" in resolved.index:
            resolved["p_n"] = p_neu

        if "cnv_state_map" in resolved.index:
            resolved["cnv_state_map"] = max(
                state_weights,
                key=state_weights.get,
            )

        if "cnv_state_mle" in resolved.index:
            resolved["cnv_state_mle"] = max(
                state_weights,
                key=state_weights.get,
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
                )
                if segment_col in group.columns
                else None,
                "consensus_segment": (
                    consensus_row[
                        consensus_segment_col
                    ]
                    if consensus_segment_col
                    in consensus.columns
                    else None
                ),
                "allowed_states": ",".join(
                    allowed_altered_states
                ),
                "p_neu_resolved": p_neu,
                "altered_mass_resolved": altered_mass,
                "state_weights": state_weights,
                "resolution": resolution,
            }
        )

    resolved_joint_post = pd.DataFrame(
        resolved_rows
    ).reset_index(drop=True)

    resolution_log = pd.DataFrame(
        audit_rows
    )

    # Final guarantee: one physical interval per barcode.
    duplicated_after = resolved_joint_post.duplicated(
        grouping_key,
        keep=False,
    )

    if duplicated_after.any():
        raise RuntimeError(
            "Multi-state resolution did not produce unique "
            "barcode–interval rows."
        )

    resolved_probabilities = resolved_joint_post[
        state_cols
    ].to_numpy(dtype=float)

    if not np.allclose(
        resolved_probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0,
    ):
        raise RuntimeError(
            "Resolved state probabilities do not sum to one."
        )

    return resolved_joint_post, resolution_log



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
            raise KeyError(
                "clone_post is missing barcode column "
                f"{clone_post_barcode_col!r}."
            )

        barcode_values = clone_post[
            clone_post_barcode_col
        ].to_numpy(copy=True)

    if pd.isna(barcode_values).any():
        source = (
            "clone_post.index"
            if clone_post_barcode_col is None
            else f"clone_post[{clone_post_barcode_col!r}]"
        )

        raise ValueError(
            f"{source} contains missing barcode identifiers."
        )

    duplicated = pd.Index(barcode_values).duplicated()

    if duplicated.any():
        duplicated_barcodes = (
            pd.Index(barcode_values[duplicated])
            .unique()
            .tolist()
        )

        raise ValueError(
            "clone_post must contain one row per barcode. "
            f"Duplicated barcodes include: {duplicated_barcodes[:10]}"
        )

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
    ) -> pd.DataFrame:
    """
    Build a complete barcode × chromosome-arm CNA representation.

    Unreported genomic regions are treated as neutral.

    Parameters
    ----------
    joint_post
        Per-barcode, per-segment CNA posterior table.

    clone_post
        Clone-assignment table. It defines the complete barcode universe
        and provides optional clone metadata.

    arm_reference
        Output from ``prepare_arm_reference()``.

    cell_col
        Barcode column in joint_post and barcode column name in the
        returned table.

    clone_post_barcode_col
        Barcode column in clone_post. If None, clone_post.index is used.

    chrom_col, start_col, end_col
        Segment-coordinate columns in joint_post.

    state_cols
        Source columns containing the six required CNA-state posterior
        probabilities.

        A mapping is recommended:

            {
                "neu": "neutral_probability",
                "loh": "loh_probability",
                "amp": "gain_probability",
                "del": "loss_probability",
                "bamp": "biallelic_gain_probability",
                "bdel": "biallelic_loss_probability",
            }

        Keys may also use canonical names such as ``"p_neu"``.

        A sequence is accepted for backward compatibility and must follow
        this order:

            neu, loh, amp, del, bamp, bdel

        If None, canonical column names are used.

    probability_tolerance
        Numerical tolerance used when validating probabilities.

    validate_nonoverlap
        If True, reject overlapping reported segments within the same
        barcode and chromosome.

    Returns
    -------
    pd.DataFrame
        One row per barcode and chromosome arm.

        Input posterior columns are harmonized internally. The returned
        table always uses the canonical names:

            p_neu, p_loh, p_amp, p_del, p_bamp, p_bdel

    Notes
    -----
    joint_post segments and arm_reference coordinates must use compatible
    0-based coordinates. No coordinate conversion is performed.
    """
    state_column_map = _resolve_cna_state_columns(
        state_cols
    )

    source_state_cols = [
        state_column_map[state]
        for state in CNA_STATES
    ]

    canonical_state_cols = list(
        CNA_STATE_COLS
    )

    coordinate_columns = {
        cell_col,
        chrom_col,
        start_col,
        end_col,
    }

    conflicting_columns = coordinate_columns.intersection(
        source_state_cols
    )

    if conflicting_columns:
        raise ValueError(
            "CNA probability columns cannot also be used as barcode "
            "or coordinate columns. Conflicts: "
            f"{sorted(conflicting_columns)}"
        )

    required_joint_columns = {
        cell_col,
        chrom_col,
        start_col,
        end_col,
        *source_state_cols,
    }

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
            f"Missing arm-reference columns: {sorted(missing)}"
        )

    # --------------------------------------------------------------
    # Prepare clone_post barcodes
    # --------------------------------------------------------------

    clone_table = _prepare_clone_post_barcodes(
        clone_post,
        output_cell_col=cell_col,
        clone_post_barcode_col=clone_post_barcode_col,
    )

    barcodes = clone_table[
        [cell_col]
    ].copy()

    # --------------------------------------------------------------
    # Prepare and harmonize joint_post segments
    # --------------------------------------------------------------

    state_rename_map = {
        state_column_map[state]: f"p_{state}"
        for state in CNA_STATES
    }

    segments = (
        joint_post[
            [
                cell_col,
                chrom_col,
                start_col,
                end_col,
                *source_state_cols,
            ]
        ]
        .rename(
            columns={
                chrom_col: "CHROM",
                start_col: "seg_start",
                end_col: "seg_end",
                **state_rename_map,
            }
        )
        .copy()
    )

    if segments[cell_col].isna().any():
        raise ValueError(
            f"joint_post[{cell_col!r}] contains missing barcodes."
        )

    known_barcodes = pd.Index(
        barcodes[cell_col]
    )

    segment_barcodes = pd.Index(
        segments[cell_col].drop_duplicates()
    )

    unknown_barcodes = segment_barcodes.difference(
        known_barcodes
    )

    if len(unknown_barcodes) > 0:
        raise ValueError(
            "joint_post contains barcodes that are absent from "
            "clone_post. Examples: "
            f"{unknown_barcodes[:10].tolist()}"
        )

    segments["CHROM"] = _normalize_chromosome(
        segments["CHROM"]
    )

    for column in ("seg_start", "seg_end"):
        values = pd.to_numeric(
            segments[column],
            errors="raise",
        )

        if not np.allclose(
            values,
            np.round(values),
        ):
            raise ValueError(
                f"{column!r} contains non-integer coordinates."
            )

        segments[column] = np.round(
            values
        ).astype(np.int64)

    if (segments["seg_start"] < 0).any():
        raise ValueError(
            "CNA segment starts cannot be negative."
        )

    if (
        segments["seg_end"]
        <= segments["seg_start"]
    ).any():
        raise ValueError(
            "Every CNA segment must have seg_end > seg_start."
        )

    # --------------------------------------------------------------
    # Validate segment posterior probabilities
    # --------------------------------------------------------------

    for column in canonical_state_cols:
        segments[column] = pd.to_numeric(
            segments[column],
            errors="raise",
        )

    probabilities = segments[
        canonical_state_cols
    ].to_numpy(dtype=float)

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite CNA probabilities were found."
        )

    outside_probability_range = (
        (probabilities < -probability_tolerance).any()
        or
        (probabilities > 1 + probability_tolerance).any()
    )

    if outside_probability_range:
        raise ValueError(
            "CNA-state probabilities must lie in [0, 1]."
        )

    probability_sum = probabilities.sum(
        axis=1
    )

    if not np.allclose(
        probability_sum,
        1.0,
        atol=probability_tolerance,
        rtol=0,
    ):
        raise ValueError(
            "The CNA-state probabilities must sum to one. "
            f"Observed range: {probability_sum.min():.6g}–"
            f"{probability_sum.max():.6g}"
        )

    # Correct only negligible floating-point deviations.
    segments[canonical_state_cols] = (
        np.clip(
            probabilities,
            0.0,
            1.0,
        )
        / probability_sum[:, None]
    )

    # Remove chromosomes absent from the selected arm reference.
    segments = segments.loc[
        segments["CHROM"].isin(
            arm_reference["CHROM"]
        )
    ].copy()

    # --------------------------------------------------------------
    # Validate that source segments do not overlap
    # --------------------------------------------------------------

    if validate_nonoverlap and not segments.empty:
        ordered = segments.sort_values(
            [
                cell_col,
                "CHROM",
                "seg_start",
                "seg_end",
            ],
            key=natsort.natsort_keygen(),
        )

        previous_end = (
            ordered.groupby(
                [cell_col, "CHROM"],
                sort=False,
            )["seg_end"]
            .shift()
        )

        overlaps_previous = (
            ordered["seg_start"] < previous_end
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
                "Overlapping CNA segments were found within a "
                "barcode and chromosome:\n"
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

    # --------------------------------------------------------------
    # Create the complete neutral barcode × arm background
    # --------------------------------------------------------------

    background = (
        barcodes.assign(_join_key=1)
        .merge(
            arm_reference[
                arm_columns
            ].assign(_join_key=1),
            on="_join_key",
            how="inner",
        )
        .drop(columns="_join_key")
    )

    background["p_neu"] = 1.0

    for column in canonical_state_cols:
        if column != "p_neu":
            background[column] = 0.0

    background["reported_segment_fraction"] = 0.0

    # --------------------------------------------------------------
    # Replace neutral arm fractions with segment posterior mass
    # --------------------------------------------------------------

    if not segments.empty:
        overlaps = segments.merge(
            arm_reference[arm_columns],
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

        overlaps["overlap_fraction"] = (
            overlaps["overlap_bp"]
            / overlaps["arm_length"]
        )

        # Each arm starts as fully neutral. For an overlap fraction f,
        # the neutral mass changes from f to f * p_neu.
        overlaps["_neutral_delta"] = (
            overlaps["overlap_fraction"]
            * (overlaps["p_neu"] - 1.0)
        )

        mass_columns: list[str] = []

        for column in canonical_state_cols:
            if column == "p_neu":
                continue

            mass_column = f"_{column}_mass"

            overlaps[mass_column] = (
                overlaps["overlap_fraction"]
                * overlaps[column]
            )

            mass_columns.append(
                mass_column
            )

        aggregations = {
            "reported_segment_fraction": (
                "overlap_fraction",
                "sum",
            ),
            "_neutral_delta": (
                "_neutral_delta",
                "sum",
            ),
        }

        aggregations.update(
            {
                mass_column: (
                    mass_column,
                    "sum",
                )
                for mass_column in mass_columns
            }
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
            .agg(**aggregations)
        )

        background = background.merge(
            arm_mass,
            on=[
                cell_col,
                "arm_id",
            ],
            how="left",
            suffixes=("", "_projected"),
            validate="one_to_one",
        )

        background["p_neu"] += (
            background["_neutral_delta"]
            .fillna(0.0)
        )

        for column in canonical_state_cols:
            if column == "p_neu":
                continue

            background[column] += (
                background[f"_{column}_mass"]
                .fillna(0.0)
            )

        background["reported_segment_fraction"] = (
            background[
                "reported_segment_fraction_projected"
            ]
            .fillna(0.0)
        )

        background = background.drop(
            columns=[
                "_neutral_delta",
                "reported_segment_fraction_projected",
                *mass_columns,
            ]
        )

    # --------------------------------------------------------------
    # Validate projected arm-state probabilities
    # --------------------------------------------------------------

    arm_probabilities = background[
        canonical_state_cols
    ].to_numpy(dtype=float)

    if (
        background["reported_segment_fraction"]
        > 1 + 1e-8
    ).any():
        raise ValueError(
            "Reported segments cover an arm more than once for at "
            "least one barcode."
        )

    if (arm_probabilities < -1e-8).any():
        raise ValueError(
            "Projection produced negative arm probabilities."
        )

    if not np.allclose(
        arm_probabilities.sum(axis=1),
        1.0,
        atol=1e-6,
        rtol=0,
    ):
        raise ValueError(
            "Projected chromosome-arm state probabilities do not "
            "sum to one."
        )

    background[canonical_state_cols] = np.clip(
        arm_probabilities,
        0.0,
        1.0,
    )

    # --------------------------------------------------------------
    # Add derived arm-level quantities
    # --------------------------------------------------------------

    background["p_gain"] = (
        background["p_amp"]
        + background["p_bamp"]
    )

    background["p_loss"] = (
        background["p_del"]
        + background["p_bdel"]
    )

    background["p_cnv"] = (
        1.0 - background["p_neu"]
    )

    background["signed_cna"] = (
        background["p_gain"]
        - background["p_loss"]
    )

    background["expected_altered_bp"] = (
        background["p_cnv"]
        * background["arm_length"]
    )

    probabilities = background[
        canonical_state_cols
    ].to_numpy(dtype=float)

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        entropy = -np.where(
            probabilities > 0,
            probabilities * np.log(probabilities),
            0.0,
        ).sum(axis=1)

    background["state_entropy"] = (
        entropy
        / np.log(len(canonical_state_cols))
    )

    dominant_index = probabilities.argmax(
        axis=1
    )

    state_names = np.asarray(
        CNA_STATES
    )

    background["dominant_state"] = (
        state_names[dominant_index]
    )

    background["dominant_probability"] = probabilities[
        np.arange(len(background)),
        dominant_index,
    ]

    # --------------------------------------------------------------
    # Add optional clone metadata
    # --------------------------------------------------------------

    metadata_columns = [
        column
        for column in (
            "clone_opt",
            "GT_opt",
            "p_opt",
            "p_cnv",
            "compartment_opt",
        )
        if column in clone_table.columns
    ]

    metadata = clone_table[
        [
            cell_col,
            *metadata_columns,
        ]
    ].copy()

    if "p_cnv" in metadata.columns:
        metadata = metadata.rename(
            columns={
                "p_cnv": "clone_p_cnv",
            }
        )

    background = background.merge(
        metadata,
        on=cell_col,
        how="left",
        validate="many_to_one",
    )

    # Preserve chromosome-arm ordering from arm_reference.
    arm_order = {
        arm_id: position
        for position, arm_id
        in enumerate(arm_reference["arm_id"])
    }

    background["_arm_order"] = (
        background["arm_id"].map(
            arm_order
        )
    )

    return (
        background.sort_values(
            [
                cell_col,
                "_arm_order",
            ]
        )
        .drop(columns="_arm_order")
        .reset_index(drop=True)
    )



def arm_probability_matrices(
    barcode_arm_posteriors: pd.DataFrame,
    cell_col: str = "cell",
    arm_order: Sequence[str] | None = None,
    value_cols: Sequence[str] = (
        "p_neu",
        "p_loh",
        "p_amp",
        "p_del",
        "p_bamp",
        "p_bdel",
        "p_gain",
        "p_loss",
        "p_cnv",
        "signed_cna",
    ),
    ) -> dict[str, pd.DataFrame]:
    """
    Create one barcode × arm matrix for each CNA-state quantity.
    """
    if arm_order is None:
        arm_order = (
            barcode_arm_posteriors["arm_id"]
            .drop_duplicates()
            .tolist()
        )

    matrices = {}

    for value_col in value_cols:
        matrices[value_col] = (
            barcode_arm_posteriors.pivot(
                index=cell_col,
                columns="arm_id",
                values=value_col,
            )
            .reindex(columns=arm_order)
        )

    return matrices