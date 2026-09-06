#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 23:57:36 2026

@author: carlino.calogero
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import pandas as pd


_NUMERIC_CHROM = re.compile(r"^(?:chr)?([1-9][0-9]*)$", 
                            re.IGNORECASE)

_SEX_CHROM = re.compile(r"^(?:chr)?([XY])$",
                        re.IGNORECASE)


def canonical_chromosome(value, include_x:bool=False, include_y:bool=False) -> str | None:
    """
    Convert an accepted chromosome label to
    canonical representation.

    Numeric chromosomes:
        chr1 -> 1
        1    -> 1

    Sex chromosomes are opt-in:
        chrX/X -> X only when include_x=True
        chrY/Y -> Y only when include_y=True

    Any other contig is rejected.
    """
    if pd.isna(value):
        return None

    value = str(value).strip()

    match = _NUMERIC_CHROM.fullmatch(value)

    if match is not None:
        return str(int(match.group(1)))

    match = _SEX_CHROM.fullmatch(value)

    if match is None:
        return None

    chrom = match.group(1).upper()

    if chrom == "X" and include_x:
        return "X"

    if chrom == "Y" and include_y:
        return "Y"

    return None


@dataclass(frozen=True)
class GenomeSpec:

    name: str

    # Only accepted analysis chromosomes are retained here.
    # source_chrom preserves the FASTA/FAI spelling
    # needed by SnapATAC2.
    chrom_sizes: pd.DataFrame

    include_x: bool = False
    include_y: bool = False

    # Internal chromosome names: 1, 2, ..., X, Y
    excluded_regions: pd.DataFrame | None = None
    
    
    @classmethod
    def from_chrom_sizes(
        cls,
        name: str,
        chrom_sizes,
        include_x: bool = False,
        include_y: bool = False,
        excluded_regions: pd.DataFrame | None = None
        ):
    
        if isinstance(chrom_sizes, dict):
    
            raw = pd.DataFrame({"source_chrom": list(chrom_sizes.keys()),
                                "length": list(chrom_sizes.values())})
    
        elif isinstance(chrom_sizes, pd.DataFrame):
    
            raw = chrom_sizes.copy()
    
            if "source_chrom" not in raw.columns:
    
                if "CHROM" not in raw.columns:
                    raise ValueError("chrom_sizes must contain either "
                                     "'source_chrom' or 'CHROM'.")
    
                raw = raw.rename(columns={"CHROM": "source_chrom"})
    
            raw = raw[["source_chrom", "length"]].copy()
    
        else:
    
            raise TypeError("chrom_sizes must be a dict or DataFrame.")
    
        raw["source_chrom"] = raw["source_chrom"].astype(str).str.strip()
        raw["length"] = pd.to_numeric(raw["length"],errors="raise").astype(np.int64)
    
        raw["CHROM"] = raw["source_chrom"].map(
            lambda x: canonical_chromosome(
                x,
                include_x=include_x,
                include_y=include_y,
                ))
    
        raw = raw.dropna(subset=["CHROM"]).copy()
    
        if raw.empty:
            raise ValueError("No valid analysis chromosomes found.")
        if (raw["length"] <= 0).any():
            raise ValueError("Chromosome lengths must be positive.")
    
        duplicated = raw["CHROM"].duplicated(keep=False)
    
        if duplicated.any():
            raise ValueError("Multiple reference contigs map to "
                             "the same canonical chromosome.")
        if include_x and "X" not in set(raw["CHROM"]):
            raise ValueError("include_x=True but X is absent.")
        if include_y and "Y" not in set(raw["CHROM"]):
            raise ValueError("include_y=True but Y is absent.")
    
        numeric = raw[raw["CHROM"].str.fullmatch(r"[0-9]+")].copy()
        numeric["_order"] = numeric["CHROM"].astype(int)
        numeric = numeric.sort_values("_order").drop(columns="_order")
        ordered = [numeric]
    
        if include_x:
            ordered.append(raw[raw["CHROM"] == "X"])
        if include_y:
            ordered.append(raw[raw["CHROM"] == "Y"])
    
        chrom_sizes = pd.concat(ordered, ignore_index=True)
    
        return cls(
            name=name,
            chrom_sizes=chrom_sizes,
            include_x=include_x,
            include_y=include_y,
            excluded_regions=excluded_regions)
    

    @classmethod
    def from_fai(
        cls,
        name,
        fai_path,
        include_x=False,
        include_y=False,
        excluded_regions=None,
        ):
    
        raw = pd.read_table(fai_path,
                            header=None,
                            usecols=[0, 1],
                            names=["source_chrom", "length"])
    
        return cls.from_chrom_sizes(
            name=name,
            chrom_sizes=raw,
            include_x=include_x,
            include_y=include_y,
            excluded_regions=excluded_regions,
        )

    @property
    def analysis_chromosomes(self) -> tuple[str, ...]:

        return tuple(self.chrom_sizes["CHROM"].astype(str))

    @property
    def chromosome_lengths(self) -> dict[str, int]:

        return dict(zip(self.chrom_sizes["CHROM"],
                        self.chrom_sizes["length"]))

    @property
    def source_chromosome_lengths(self) -> dict[str, int]:
        """
        Chromosome names as they occur in the original FASTA.

        Used only when communicating with tools reading
        FASTA-coordinate files.
        """
        return dict(zip(self.chrom_sizes["source_chrom"],
                        self.chrom_sizes["length"]))

    @property
    def canonical_to_source(self) -> dict[str, str]:

        return dict(zip(self.chrom_sizes["CHROM"],
                        self.chrom_sizes["source_chrom"]))
    
    def normalize_table(
            self,
            table: pd.DataFrame,
            chrom_col: str = "CHROM",
            table_name: str = "table",
            ) -> pd.DataFrame:

        if chrom_col not in table.columns:
            raise KeyError(f"{table_name} is missing {chrom_col!r}.")
    
        out = table.copy()
        original = out[chrom_col].copy()
        out[chrom_col] = original.map(
            lambda x: canonical_chromosome(
                x,
                include_x=self.include_x,
                include_y=self.include_y,
                ))
    
        # Also enforce membership in the selected assembly.
        valid = (out[chrom_col].notna()
                 & out[chrom_col].isin(self.analysis_chromosomes))
    
        out = out.loc[valid].copy()
    
        if out.empty:
            raise ValueError(f"No valid analysis chromosomes remain in "
                             f"{table_name} after genome filtering.")
    
        return out
    
    
    def normalize_binning(self, binning: pd.DataFrame) -> pd.DataFrame:
    
        required = {
            "CHROM",
            "start",
            "end",
        }
    
        missing = required.difference(binning.columns)
    
        if missing:
            raise ValueError("Binning table is missing columns: "
                             f"{sorted(missing)}")
    
        out = self.normalize_table(binning, table_name="genomic binning")
        out["start"] = pd.to_numeric(out["start"], errors="raise").astype(np.int64)
        out["end"] = pd.to_numeric(out["end"], errors="raise").astype(np.int64)
        chromosome_length = out["CHROM"].map(self.chromosome_lengths)
        invalid = (
            (out["start"] < 0)
            | (out["end"] <= out["start"])
            | (out["end"] > chromosome_length)
        )
    
        if invalid.any():
            raise ValueError("Genomic bins contain invalid or "
                             "out-of-bound coordinates.")
    
        source_chrom = out["CHROM"].map(self.canonical_to_source)
        
        out["bin_id"] = (
            out["CHROM"].astype(str)
            + ":"
            + out["start"].astype(str)
            + "-"
            + out["end"].astype(str)
        )
    
        out["source_bin_id"] = (
            source_chrom.astype(str)
            + ":"
            + out["start"].astype(str)
            + "-"
            + out["end"].astype(str)
        )
    
        out["width"] = out["end"] - out["start"]
    
        if out["bin_id"].duplicated().any():
            raise ValueError("Duplicated genomic bins after "
                             "chromosome normalization.")
    
        return out.reset_index(drop=True)






