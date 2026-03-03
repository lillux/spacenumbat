#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 17:21:24 2025

@author: lillux

Lazy loader for package data.

This module provides access to .tsv files as pandas DataFrames.
Data are only loaded when an attribute is actually accessed.
"""

import pandas as pd
import importlib.resources
from typing import Dict

# A mapping between attribute names and the corresponding .tsv file names
_DATA_FILES: Dict[str, str] = {
    "hg38": "gtf_hg38.tsv",
    "hg19": "gtf_hg19.tsv",
    "mm10": "gtf_mm10.tsv",
}

# Internal cache to hold loaded DataFrames
_cache: Dict[str, pd.DataFrame] = {}

def __getattr__(name: str) -> pd.DataFrame:
    """
    Lazily load a .tsv file as a pandas DataFrame when an attribute is accessed.

    When the user accesses an attribute, this function checks if that name
    corresponds to a file in _DATA_FILES. If so, it loads the file using pandas,
    caches the DataFrame, and returns it. Otherwise, it raises an AttributeError.

    Parameters
    ----------
    name : str
        The name of the attribute being accessed.

    Returns
    -------
    pd.DataFrame
        The DataFrame loaded from the corresponding .tsv file.

    Raises
    ------
    AttributeError
        If the attribute name does not correspond to any available data file.
    """
    global _cache
    # Check if the attribute matches a known .tsv file
    if name in _DATA_FILES:
        # Load and cache the DataFrame if not already done
        if name not in _cache:
            file_name = _DATA_FILES[name]
            # Use importlib.resources to open the text file from the package
            with importlib.resources.open_text(__package__, file_name) as f:
                _cache[name] = pd.read_csv(f, sep="\t")
        return _cache[name]
    raise AttributeError(f"module {__name__} has no attribute {name}")
    
    