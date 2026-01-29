#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 17:43:20 2025

@author: lillux
"""

import os
import pandas as pd
import numpy as np
import warnings

import natsort

from typing import Optional, Union
from pathlib import Path

from spacenumbat._log import get_logger
log = get_logger(__name__)
#log.info("Test diagnostics")


def load_and_validate_annotation(file_path: str, sep: str = "\t") -> pd.DataFrame:
    """
    Load and validate a gene annotation TSV file.

    The file is expected to have the following columns:
        - gene: gene name (string)
        - gene_start: gene start coordinate (integer)
        - gene_end: gene end coordinate (integer)
        - CHROM: chromosome identifier (string)

    If the first four columns are present and have the expected data types, the 
    function issues a warning and calculates 'gene_length' as gene_end - gene_start.
    
    Parameters
    ----------
    file_path : str
        The path to the TSV file containing gene annotations.
    sep : str, optional
        The delimiter used in the TSV file (default is tab, i.e., "\t").
    
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the gene annotations with the calculated 'gene_length'.
    
    aises
    ------
    FileNotFoundError
        If the file specified by file_path does not exist.
    ValueError
        If any of the required columns ('gene', 'gene_start', 'gene_end', 'CHROM') are missing
        or if their data types do not match the expected types.
    """
    # Check that the file exists.
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Load the TSV file.
    df = pd.read_csv(file_path, sep=sep)
    
    # Verify the presence of required columns.
    required_columns = ["gene", "gene_start", "gene_end", "CHROM"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required column(s): {missing_columns}")
    
    # Validate that 'gene' is of string type.
    if not pd.api.types.is_string_dtype(df["gene"]):
        raise ValueError("Column 'gene' must be of type string.")
    
    # Validate that 'gene_start' and 'gene_end' can be converted to integers.
    try:
        df["gene_start"] = df["gene_start"].astype(int)
    except Exception as e:
        raise ValueError("Column 'gene_start' must be convertible to integer.") from e
    
    try:
        df["gene_end"] = df["gene_end"].astype(int)
    except Exception as e:
        raise ValueError("Column 'gene_end' must be convertible to integer.") from e

    # Validate that 'CHROM' is of string type.
    try:
        df["CHROM"] = df["CHROM"].astype('string')
    except Exception as e:
        raise ValueError("Column 'CHROM' must be convertible to string.") from e

    # Warn the user that gene_length will be calculated.
    warnings.warn("Calculating 'gene_length' as gene_end - gene_start.", UserWarning)
    df["gene_length"] = df["gene_end"] - df["gene_start"]
    
    return df


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


def check_filter_segments(filter_segments_path: Union[Path, None]) -> pd.DataFrame:
    """
    Validate that the provided path exists, is a readable TSV file,
    and contains required columns with correct types.
    Required columns are: ['CHROM', 'seg_start', 'seg_end']

    Parameters
    ----------
    filter_segments_path : str
        File path to the TSV file containing segment data.

    Returns
    -------
    pd.DataFrame
        Loaded DataFrame if all checks pass.

    Raises
    ------
    FileNotFoundError
        If the file does not exist or path is invalid.
    ValueError
        If the file cannot be read as a TSV, or required columns are missing
        or have incorrect types.
    """
    if filter_segments_path is None:
        return filter_segments_path
    
    # Check path existence and validity
    if not os.path.exists(filter_segments_path):
        raise FileNotFoundError(f"Path does not exist: {filter_segments_path}")
    if not os.path.isfile(filter_segments_path):
        raise FileNotFoundError(f"Path is not a file: {filter_segments_path}")

    # Read the file as TSV
    try:
        df = pd.read_csv(filter_segments_path, sep='\t')
    except Exception as e:
        raise ValueError(f"Failed to read file as TSV: {e}")

    # Validate required columns
    required_columns = ['CHROM', 'seg_start', 'seg_end']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Validate column types: CHROM as string, seg_start and seg_end as integers
    # Coerce columns to expected types and check for errors
    if not pd.api.types.is_string_dtype(df['CHROM']):
        try:
            df['CHROM'] = df['CHROM'].astype("string")
        except Exception:
            raise ValueError("Column 'CHROM' cannot be converted to string")

    for col in ['seg_start', 'seg_end']:
        if not pd.api.types.is_integer_dtype(df[col]):
            # Try coercing to integers (may fail if non-numeric data present)
            try:
                df[col] = pd.to_numeric(df[col], errors='raise').astype(int)
            except Exception:
                raise ValueError(f"Column '{col}' cannot be converted to integer")

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

