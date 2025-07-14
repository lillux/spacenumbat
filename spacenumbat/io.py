#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 13 17:43:20 2025

@author: lillux
"""

import os
import pandas as pd
import warnings


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