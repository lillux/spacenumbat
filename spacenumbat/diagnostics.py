#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 17:43:20 2025

@author: lillux
"""

import os
import pandas as pd
import numpy as np

import natsort

from typing import Optional, Union
from pathlib import Path

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("Test diagnostics")


def validate_annotation(annotation: pd.DataFrame) -> pd.DataFrame:
    """
    Validate a gene-level annotation table.

    The ``gene`` column must uniquely identify one genomic interval because
    expression matrices and reference profiles use it as their feature key.
    """
    required = ["CHROM", "gene_start", "gene_end", "gene"]
    missing = [column for column in required if column not in annotation.columns]

    if missing:
        raise ValueError(
            "Gene annotation is missing required columns: "
            + ", ".join(missing)
        )

    annotation = annotation.loc[:, required].copy()

    annotation["gene"] = annotation["gene"].astype("string").str.strip()
    annotation["CHROM"] = annotation["CHROM"].astype("string").str.strip()
    annotation["gene_start"] = pd.to_numeric(annotation["gene_start"], errors="raise").astype(np.int64)
    annotation["gene_end"] = pd.to_numeric(annotation["gene_end"], errors="raise").astype(np.int64)

    missing_gene = annotation["gene"].isna() | annotation["gene"].eq("")
    if missing_gene.any():
        raise ValueError(
            f"Gene annotation contains {missing_gene.sum()} missing or empty "
            "gene identifiers."
        )

    invalid_coordinates = (
        annotation["gene_start"].lt(0)
        | annotation["gene_end"].lt(annotation["gene_start"])
    )
    if invalid_coordinates.any():
        raise ValueError(
            f"Gene annotation contains {invalid_coordinates.sum()} invalid "
            "genomic intervals."
        )

    duplicated = annotation["gene"].duplicated(keep=False)
    if duplicated.any():
        duplicate_counts = (annotation.loc[duplicated, "gene"].value_counts().head(10))
        preview = ", ".join(
            f"{gene} ({count})"
            for gene, count in duplicate_counts.items()
        )

        raise ValueError(
            "Column 'gene' must contain unique feature identifiers. "
            f"Found {annotation.loc[duplicated, 'gene'].nunique()} duplicated "
            f"identifiers. Examples: {preview}. "
            "Collapse transcript records to one gene-level interval or use "
            "unique gene identifiers consistently in the annotation, count "
            "matrix, and reference profile."
        )

    annotation["gene_length"] = (
        annotation["gene_end"] - annotation["gene_start"]
    )

    return annotation



def load_and_validate_annotation(
    file_path: str,
    sep: str = "\t",
    ) -> pd.DataFrame:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    return validate_annotation(pd.read_csv(file_path, sep=sep))


def load_annotation(annotation):
    if isinstance(annotation, pd.DataFrame):
        return validate_annotation(annotation)

    if isinstance(annotation, (str, Path)):
        return load_and_validate_annotation(str(annotation))

    raise TypeError("annotation must be a DataFrame or TSV path")


def check_segs_fix(segs_consensus_fix: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Validate and enrich the consensus segment dataframe.

    Parameters
    ----------
    segs_consensus_fix : pd.DataFrame or None
        Consensus segment dataframe with columns:
        ['CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state']

    Returns
    -------
    pd.DataFrame or None
        Cleaned and enriched dataframe, or None if input is None.

    Raises
    ------
    ValueError
        If the dataframe is malformed or missing required columns.
    """
    if segs_consensus_fix is None:
        return None

    required_cols = ['CHROM', 'seg', 'seg_start', 'seg_end', 'cnv_state']
    if not all(col in segs_consensus_fix.columns for col in required_cols):
        raise ValueError("The consensus segment dataframe appears to be malformed. Please fix.\n"
                         f"The dataframe requires the following columns:\n{required_cols}\n"
                         f"The current columns in your dataframe are:\n{segs_consensus_fix.columns}")

    # Chromosome relevel and sort
    # segs_consensus_fix = relevel_chrom(segs_consensus_fix)
    segs_consensus_fix.CHROM = segs_consensus_fix.CHROM.astype('string')
    segs_consensus_fix = segs_consensus_fix.sort_values(['CHROM', 'seg_start'], 
                                                        key=natsort.natsort_keygen()).reset_index(drop=True)

    # If seg column is integer, convert to string: CHROM_SEG
    if pd.api.types.is_integer_dtype(segs_consensus_fix['seg']):
        segs_consensus_fix = segs_consensus_fix.copy()
        segs_consensus_fix['seg'] = segs_consensus_fix['CHROM'].astype("string") + '_' + segs_consensus_fix['seg'].astype("string")

    # segs_consensus_fix = segs_consensus_fix.sort_values(['CHROM']).copy()
    segs_consensus_fix['cnv_state_post'] = segs_consensus_fix['cnv_state']
    segs_consensus_fix['seg_cons'] = segs_consensus_fix['seg']
    segs_consensus_fix['p_amp'] = (segs_consensus_fix['cnv_state'] == 'amp').astype(int)
    segs_consensus_fix['p_del'] = (segs_consensus_fix['cnv_state'] == 'del').astype(int)
    segs_consensus_fix['p_loh'] = (segs_consensus_fix['cnv_state'] == 'loh').astype(int)
    segs_consensus_fix['p_bamp'] = (segs_consensus_fix['cnv_state'] == 'bamp').astype(int)
    segs_consensus_fix['p_bdel'] = (segs_consensus_fix['cnv_state'] == 'bdel').astype(int)
    segs_consensus_fix['seg_length'] = segs_consensus_fix['seg_end'] - segs_consensus_fix['seg_start']
    segs_consensus_fix['LLR'] = np.where(
        segs_consensus_fix['cnv_state'] == 'neu',
        np.nan,
        np.inf
    )
    return segs_consensus_fix.reset_index(drop=True)


def check_segs_loh(segs_loh: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
    """
    Check and standardize the format of a clonal LOH segment dataframe.

    Parameters
    ----------
    segs_loh : pd.DataFrame or None
        DataFrame with columns ['CHROM', 'seg', 'seg_start', 'seg_end'].
        Can be None.

    Returns
    -------
    pd.DataFrame or None
        Cleaned dataframe (or None if input was None).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    if segs_loh is None:
        return None

    required_cols = ['CHROM', 'seg', 'seg_start', 'seg_end']
    if not all([col in segs_loh.columns for col in required_cols]):
        raise ValueError("The clonal LOH segment dataframe appears to be malformed. Please fix.\n"
                         f"The dataframe requires the following columns:\n{required_cols}\n"
                         f"The current columns in your dataframe are:\n{segs_loh.columns}")

    # If seg column is integer, convert to string: CHROM_SEG
    if pd.api.types.is_integer_dtype(segs_loh['seg']):
        segs_loh = segs_loh.copy()
        segs_loh['seg'] = segs_loh['CHROM'].astype("string") + '_' + segs_loh['seg'].astype("string")

    # Add loh = True column
    segs_loh = segs_loh.copy()
    segs_loh['loh'] = True

    # Relevel and sort by chromosome and seg_start
    # segs_loh = relevel_chrom(segs_loh)
    segs_loh = segs_loh.sort_values(['CHROM', 'seg_start'], key=natsort.natsort_keygen()).reset_index(drop=True)

    return segs_loh


def check_filter_segments(filter_segments):

    if filter_segments is None:
        return None

    if isinstance(filter_segments, pd.DataFrame):
        df = filter_segments.copy()

    elif isinstance(filter_segments,(str, Path),):
        if not os.path.isfile(filter_segments):
            raise FileNotFoundError(f"File not found: {filter_segments}")

        df = pd.read_table(filter_segments)

    else:
        raise TypeError("filter_segments must be a "
                        "DataFrame or TSV path.")
        
    if ("seg_start" not in df.columns and "start" in df.columns):
        df = df.rename(columns={"start": "seg_start"})
    
    if ("seg_end" not in df.columns and "end" in df.columns):
        df = df.rename(columns={"end": "seg_end"})

    required = {"CHROM",
                "seg_start",
                "seg_end"}

    missing = required.difference(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["CHROM"] = df["CHROM"].astype("string").str.strip()
    df["seg_start"] = pd.to_numeric(df["seg_start"], errors="raise").astype(np.int64)
    df["seg_end"] = pd.to_numeric(df["seg_end"], errors="raise").astype(np.int64)

    return df


def check_contam(bulk: pd.DataFrame) -> None:
    """
    Check inter-individual contamination by estimating the homozygous SNP rate.

    Parameters
    ----------
    bulk : pd.DataFrame
        Pseudobulk profile with columns:
        - 'DP' : read depth per SNP (numeric)
        - 'AR' : allele ratio per SNP in [0, 1] (numeric)

    Notes
    -----
    Computes the proportion of SNPs with DP ≥ 8 whose allele ratio is exactly 0 or 1.
    If this homozygous rate exceeds 40%, a warning is logged.
    """

    ar_filter = bulk[bulk.DP >= 8].AR.dropna()
    hom_rate = ((ar_filter == 0) | (ar_filter == 1)).mean()
    log.info(f"Homology rate of the sample is: {hom_rate*100:.2f}%")

    if hom_rate > 0.4:
        msg = (f"High SNP contamination detected ({hom_rate*100:.2f}%).\n"
                "Please make sure that cells from only one individual are included in the genotyping step.")
        log.warning(msg)

    return


def check_exp_noise(bulk: pd.DataFrame) -> None:
    """
    Check expression noise level based on MSE.

    Parameters
    ----------
    bulk : pd.DataFrame
        Pseudobulk profile containing a column:
        - 'mse' : model mean squared error (numeric).

    Notes
    -----
    Noise levels:
      - high   : mse > 1.5  -> suggests using a custom expression reference profile
      - medium : 0.5 < mse ≤ 1.5
      - low    : mse ≤ 0.5

    Logs a single-line summary with the noise level and MSE.
    """
    mse = bulk.mse.dropna().mean()
    if mse.size == 0:
        # Nothing to report
        log.info("Expression noise level (MSE): unavailable (no non-NA values).")
        return

    if np.any(mse > 1.5):
        noise_level  = "high"
        noise_msg = "Consider using a custom expression reference profile."
    elif np.any(mse > 0.5):
        noise_level = "medium"
        noise_msg = ""
    else:
        noise_level = "low"
        noise_msg = ""
    
    msg = (f"Expression noise level (MSE): {noise_level}.\n "
           f"MSE of the sample gene expression vs the reference profile is: {mse:.2f}.\n"
           f"{noise_msg}")
    log.info(msg)

    return

