#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 21 03:36:20 2025

@author: carlino.calogero

This script runs SNP pileup with cellsnp-lite, phases variants with Eagle2,
then prepares allele count tables for Spacenumbat.
"""

import argparse
import os
import subprocess
from typing import List

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.io import mmread

import scipy.sparse as sp
import pyranges as pr

import spacenumbat
from spacenumbat import diagnostics


# Utility functions

VALID_UMI_TAGS = {"Auto", "UB", "None", "XM"}


def _split_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def _validate_10x_inputs(
    samples: list[str],
    bams: list[str],
    barcodes: list[str],
    umi_tags: list[str],
    ) -> None:
    n = len(samples)

    if n == 0:
        raise ValueError("At least one sample must be provided.")

    if len(set(samples)) != n:
        raise ValueError("Sample names must be unique.")

    lengths = {
        "samples": len(samples),
        "bams": len(bams),
        "barcodes": len(barcodes),
        "UMItag": len(umi_tags),
    }

    if len(set(lengths.values())) != 1:
        raise ValueError(
            "For 10x/mixed-10x mode, --samples, --bams, --barcodes and "
            "--UMItag must contain the same number of entries. "
            f"Received: {lengths}"
        )

    invalid = sorted(set(umi_tags) - VALID_UMI_TAGS)
    if invalid:
        raise ValueError(
            f"Invalid UMI tag(s): {invalid}. "
            f"Allowed values are {sorted(VALID_UMI_TAGS)}."
        )

    for path in [*bams, *barcodes]:
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
            
    return


def make_cell_manifest(
    sample: str,
    modality: str,
    barcodes: list[str],
    namespace: bool,
    ) -> pd.DataFrame:
    """Map raw barcodes to SpaceNumbat observation identifiers."""
    cells = ([f"{sample}::{bc}" for bc in barcodes] if namespace else list(barcodes))

    return pd.DataFrame({
        "cell": cells,
        "cell_id": cells,   # unpaired mode: one observation = one biological cell
        "barcode": barcodes,
        "library": sample,
        "modality": modality,
        })
            

def load_annotation(
    gtf_path: str | None = None,
    genome: str = "hg38",
    ) -> pd.DataFrame:
    """
    Load the genomic feature annotation used to annotate phased SNPs.

    A custom TSV supplied through `gtf_path` takes precedence over the
    packaged genome annotation.
    """
    if gtf_path is not None:
        return diagnostics.load_and_validate_annotation(gtf_path)

    if genome == "hg38":
        gtf = spacenumbat.data.hg38
    elif genome == "hg38_old":
        gtf = spacenumbat.data.hg38_old
    else:
        raise ValueError(f"Unsupported packaged genome {genome!r}. "
                         "Supply a custom annotation with --gtf.")

    return diagnostics.validate_annotation(gtf)


def parse_info(info: str) -> dict:
    """Parse INFO field from cellsnp-lite VCF."""
    out = {}
    for item in info.split(";"):
        if "=" in item:
            key, val = item.split("=")
            out[key] = val
    return out


def load_vcf(path: str) -> pd.DataFrame:
    """Read a VCF produced by cellsnp-lite into a DataFrame."""
    lines = []
    with open(path, "r") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip().split("\t")
            info = parse_info(parts[7])
            lines.append({
                "CHROM": parts[0].replace("chr", ""),
                "POS": int(parts[1]),
                "REF": parts[3],
                "ALT": parts[4],
                "AD": int(info.get("AD", 0)),
                "DP": int(info.get("DP", 0)),
                "OTH": int(info.get("OTH", 0)),
            })
    df = pd.DataFrame(lines)
    df["snp_id"] = df.CHROM.astype(str) + "_" + df.POS.astype(str) + "_" + df.REF + "_" + df.ALT
    df["AR"] = df.AD / df.DP.replace({0: pd.NA})
    df = df.dropna(subset=["AR"])
    return df


def write_vcf_chr(path: str, snps: pd.DataFrame, label: str, chr_prefix: bool = True) -> None:
    """Write per-chromosome VCF with proper INFO/FORMAT header lines."""
    # declare the contigs that may be emitted
    contigs = [f"chr{i}" for i in range(1, 23)] if chr_prefix else [str(i) for i in range(1, 23)]

    header = [
        "##fileformat=VCFv4.2",
        "##source=numbat",
        # INFO field definitions
        '##INFO=<ID=AD,Number=1,Type=Integer,Description="Alt read count across all cells/samples">',
        '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total read depth across all cells/samples">',
        '##INFO=<ID=OTH,Number=1,Type=Integer,Description="Other reads (non-REF/ALT) across all cells/samples">',
        # FORMAT field definitions
        '##FORMAT=<ID=GT,Number=1,Type=String,Description="Unphased genotype">',
    ]
    # Add contig lines (optional)
    header += [f"##contig=<ID={c}>" for c in contigs]

    # Column header line with sample label
    header.append("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + label)

    with open(path, "w") as out:
        for h in header:
            out.write(h + "\n")
        for _, row in snps.iterrows():
            chrom = row.CHROM
            if chr_prefix:
                chrom = f"chr{chrom}"
            info = f"AD={row.AD};DP={row.DP};OTH={row.OTH}"
            line = [
                chrom,
                str(int(row.POS)),
                ".",
                row.REF,
                row.ALT,
                ".",
                "PASS",
                info,
                "GT",
                row.GT,
            ]
            out.write("\t".join(line) + "\n")
            
    return


def genotype(label: str, vcfs: List[str], outdir: str, het_only: bool = False, chr_prefix: bool = True) -> None:
    dfs = [load_vcf(v) for v in vcfs]
    snps = pd.concat(dfs)
    snps = snps.groupby(["CHROM", "POS", "REF", "ALT", "snp_id"], as_index=False).agg({"AD": "sum", "DP": "sum", "OTH": "sum"})
    snps["AR"] = snps.AD / snps.DP.replace({0: pd.NA})
    snps = snps.sort_values(["CHROM", "POS"])

    for chr_num in range(1, 23):
        chr_snps = snps[snps.CHROM.astype("string") == str(chr_num)].copy()
        if chr_snps.empty:
            continue
        chr_snps["het"] = (chr_snps.AR >= 0.1) & (chr_snps.AR <= 0.9)
        chr_snps["hom_alt"] = (chr_snps.AR == 1) & (chr_snps.DP >= 10)
        chr_snps["hom_ref"] = (chr_snps.AR == 0) & (chr_snps.DP >= 10)
        chr_snps = chr_snps[chr_snps.het | chr_snps.hom_alt]
        chr_snps.loc[chr_snps.het, "GT"] = "0/1"
        chr_snps.loc[chr_snps.hom_alt, "GT"] = "1/1"
        chr_snps.loc[chr_snps.hom_ref, "GT"] = "0/0"
        if het_only:
            chr_snps = chr_snps[chr_snps.het]
        if chr_snps.empty:
            continue

        out_file = os.path.join(outdir, f"{label}_chr{chr_num}.vcf")
        write_vcf_chr(out_file, chr_snps, label, chr_prefix=chr_prefix)

        # compress to .vcf.gz and tabix-index ---
        gz_path = out_file + ".gz"
        try:
            import pysam
            # compress then index; remove uncompressed file
            pysam.tabix_compress(out_file, gz_path, force=True)
            pysam.tabix_index(gz_path, preset="vcf", force=True)
            try:
                os.remove(out_file)
            except OSError:
                pass
        except Exception:
            # fallback to system bgzip/tabix
            subprocess.run(["bgzip", "-f", out_file], check=True)
            subprocess.run(["tabix", "-f", "-p", "vcf", gz_path], check=True)
    return


def read_vcf_table(path: str) -> pd.DataFrame:
    """Fast VCF body reader into a DataFrame (no parsing of INFO/FORMAT)."""
    df = pd.read_csv(path, sep="\t", comment="#", header=None, low_memory=False)
    return df

def load_phased_concat(outdir: str, label: str) -> pd.DataFrame:
    """Concatenate {label}_chr*.phased.vcf.gz into one DataFrame with CHROM stripped of 'chr'."""
    dfs = []
    for chr_num in range(1, 23):
        vcf_gz = os.path.join(outdir, "phasing", f"{label}_chr{chr_num}.phased.vcf.gz")
        if not os.path.exists(vcf_gz):
            raise FileNotFoundError(f"Phased VCF not found: {vcf_gz}")
        df = pd.read_csv(vcf_gz, sep="\t", comment="#", header=None, low_memory=False)
        dfs.append(df)
    phased = pd.concat(dfs, axis=0, ignore_index=True)
    # Standard VCF format
    phased = phased.rename(columns={0: "CHROM", 1: "POS", 3: "REF", 4: "ALT"})
    phased["CHROM"] = phased["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    return phased

def load_pileup_body(pu_dir: str) -> pd.DataFrame:
    """Read cellSNP.base.vcf and strip 'chr' from CHROM."""
    vcf_pu = pd.read_csv(os.path.join(pu_dir, "cellSNP.base.vcf"),
                         sep="\t", comment="#", header=None, low_memory=False)
    vcf_pu = vcf_pu.rename(columns={0: "CHROM", 1: "POS", 3: "REF", 4: "ALT"})
    vcf_pu["CHROM"] = vcf_pu["CHROM"].astype(str).str.replace("^chr", "", regex=True)
    return vcf_pu

def read_cellsnp_mtx(pu_dir: str):
    """Load AD/DP as CSR matrices and cell barcodes list."""
    ad_path = os.path.join(pu_dir, "cellSNP.tag.AD.mtx")
    dp_path = os.path.join(pu_dir, "cellSNP.tag.DP.mtx")
    bc_path = os.path.join(pu_dir, "cellSNP.samples.tsv")
    AD = mmread(ad_path).tocsr()
    DP = mmread(dp_path).tocsr()
    barcodes = pd.read_csv(bc_path, header=None, sep="\t")[0].astype(str).tolist()
    return AD, DP, barcodes



def preprocess_allele(
    sample: str,
    vcf_pu: pd.DataFrame,
    vcf_phased: pd.DataFrame,
    AD: sp.spmatrix,
    DP: sp.spmatrix,
    barcodes: List[str],
    gtf: pd.DataFrame,
    gmap: str,
    ) -> pd.DataFrame:
    """
    Preprocess allele counts and annotations for one sample.

    This function combines per-cell allele depths from pileup (DP, AD)
    with SNP-level information from the pileup VCF
    and phased genotypes from Eagle2, then annotates SNPs with gene and genetic
    map positions and keeps only heterozygous SNPs.

    Parameters
    ----------
    sample : str
        Sample label. Must match a genotype column name in `vcf_phased`.
    vcf_pu : pandas.DataFrame
        Pileup VCF table from cellsnp-lite, with at least columns:
        ['CHROM', 'POS', 'REF', 'ALT', 'INFO'] or already parsed
        ['CHROM', 'POS', 'REF', 'ALT', 'AD', 'DP', 'OTH'].
        If INFO is present and AD/DP/OTH are missing, they will be parsed.
    vcf_phased : pandas.DataFrame
        Phased VCF from Eagle2 (concatenated across chromosomes), with at least
        columns ['CHROM', 'POS', 'REF', 'ALT'] and a column named `sample`
        containing phased genotypes ('0|1' or '1|0').
    AD : scipy.sparse.spmatrix
        Sparse alternative allele depth matrix (SNPs × cells), typically in
        COO/CSR/CSC format.
    DP : scipy.sparse.spmatrix
        Sparse total depth matrix (SNPs × cells), same shape and ordering as AD.
    barcodes : list of str
        Cell barcodes; length must match the number of columns in AD/DP.
    gtf : pandas.DataFrame
        Gene annotation with at least columns:
        ['CHROM', 'gene_start', 'gene_end', 'gene'].
    gmap : str
        Path to genetic map file with columns:
        CHROM, POS, rate, cM.

    Returns
    -------
    pandas.DataFrame
        Tidy allele table with one row per (cell, SNP) for heterozygous SNPs,
        with columns:
        ['cell', 'snp_id', 'CHROM', 'POS', 'cM', 'REF', 'ALT', 'AD', 'DP', 'GT', 'gene'].

    Notes
    -----
    - Assumes that SNP order in AD/DP rows matches the order of rows in `vcf_pu`.
    - Only SNPs with DP_all > 1 and OTH_all == 0 (from pileup VCF) are kept.
    - Only heterozygous phased SNPs (GT in {'1|0', '0|1'}) are returned.
    """
    # Parse INFO and create snp_id
    vcf_pu = vcf_pu.copy()

    if "INFO" in vcf_pu.columns and not {"AD", "DP", "OTH"}.issubset(vcf_pu.columns):
        info_numeric = vcf_pu["INFO"].astype(str).str.replace(r"[A-Za-z=]", "", regex=True)
        ad_dp_oth = info_numeric.str.split(";", expand=True)
        ad_dp_oth.columns = ["AD", "DP", "OTH"]
        vcf_pu[["AD", "DP", "OTH"]] = ad_dp_oth.astype("Int64")

    vcf_pu["snp_id"] = (
        vcf_pu["CHROM"].astype(str) + "_"
        + vcf_pu["POS"].astype(str) + "_"
        + vcf_pu["REF"].astype(str) + "_"
        + vcf_pu["ALT"].astype(str)
    )

    # Convert DP and AD sparse matrices into long format
    dp_coo = DP.tocoo()
    dp_df = pd.DataFrame({
        "i": dp_coo.row,
        "j": dp_coo.col,
        "DP": dp_coo.data,
    })
    dp_df["cell"] = [barcodes[j] for j in dp_df["j"]]
    snp_ids = vcf_pu["snp_id"].to_numpy()
    dp_df["snp_id"] = snp_ids[dp_df["i"].values]
    dp_df = dp_df.drop(columns=["i", "j"])[["cell", "snp_id", "DP"]]

    ad_coo = AD.tocoo()
    ad_df = pd.DataFrame({
        "i": ad_coo.row,
        "j": ad_coo.col,
        "AD": ad_coo.data,
    })
    ad_df["cell"] = [barcodes[j] for j in ad_df["j"]]
    ad_df["snp_id"] = snp_ids[ad_df["i"].values]
    ad_df = ad_df.drop(columns=["i", "j"])[["cell", "snp_id", "AD"]]

    # Merge DP and AD, fill missing AD with 0
    df = dp_df.merge(ad_df, on=["cell", "snp_id"], how="left")
    df["AD"] = df["AD"].fillna(0).astype(int)

    # Join pileup-level info and compute allele ratios
    vcf_pu_renamed = vcf_pu.rename(columns={"AD": "AD_all", "DP": "DP_all", "OTH": "OTH_all"})
    df = df.merge(
        vcf_pu_renamed[["snp_id", "CHROM", "POS", "REF", "ALT", "AD_all", "DP_all", "OTH_all"]],
        on="snp_id",
        how="left",
    )

    # Avoid division by zero
    df["AR"] = df["AD"] / df["DP"].replace({0: np.nan})
    df["AR_all"] = df["AD_all"] / df["DP_all"].replace({0: np.nan})

    # Filter by global pileup quality
    df = df[(df["DP_all"] > 1) & (df["OTH_all"] == 0)].drop_duplicates()

    # Process phased VCF and attach sample genotypes
    vcf_phased = vcf_phased.copy()
    vcf_phased["snp_id"] = (
        vcf_phased["CHROM"].astype(str) + "_"
        + vcf_phased["POS"].astype(str) + "_"
        + vcf_phased["REF"].astype(str) + "_"
        + vcf_phased["ALT"].astype(str)
    )
    vcf_phased["GT"] = vcf_phased[sample]

    # Annotate SNPs with gene information via overlaps
    vcf_phased = vcf_phased.reset_index(drop=True)
    vcf_phased["snp_index_tmp"] = np.arange(len(vcf_phased))

    pr_snps = pr.PyRanges(pd.DataFrame({
        "Chromosome": vcf_phased["CHROM"].astype(str),
        "Start": vcf_phased["POS"].astype(int),
        "End": vcf_phased["POS"].astype(int) + 1,
        "snp_index_tmp": vcf_phased["snp_index_tmp"],
    }))

    gtf_tmp = gtf.reset_index(drop=True).copy()
    gtf_tmp["gene_index_tmp"] = np.arange(len(gtf_tmp))

    pr_genes = pr.PyRanges(pd.DataFrame({
        "Chromosome": gtf_tmp["CHROM"].astype(str),
        "Start": gtf_tmp["gene_start"].astype(int),
        "End": gtf_tmp["gene_end"].astype(int) + 1,
        "gene_index_tmp": gtf_tmp["gene_index_tmp"],
    }))

    ov = pr_snps.join(pr_genes).as_df()
    if not ov.empty:
        ov = ov[["snp_index_tmp", "gene_index_tmp"]]
        ov = ov.merge(
            vcf_phased[["snp_index_tmp", "snp_id"]],
            on="snp_index_tmp",
            how="left",
        )
        ov = ov.merge(
            gtf_tmp[["gene_index_tmp", "gene", "gene_start", "gene_end"]],
            on="gene_index_tmp",
            how="left",
        )
        ov = ov.sort_values(["snp_index_tmp", "gene"]).drop_duplicates(
            subset="snp_index_tmp",
            keep="first",
        )
        vcf_phased = vcf_phased.merge(
            ov[["snp_id", "gene", "gene_start", "gene_end"]],
            on="snp_id",
            how="left",
        )
    else:
        vcf_phased["gene"] = np.nan
        vcf_phased["gene_start"] = np.nan
        vcf_phased["gene_end"] = np.nan

    # Annotate SNPs with genetic map cM using interpolation
    gmap_tmp = pd.read_csv(gmap, sep=r"\s+", header=None, engine="python")
    gmap_tmp.columns = ["CHROM", "POS", "rate", "cM"]

    gmap_tmp["CHROM"] = (
        gmap_tmp["CHROM"]
        .astype(str)
        .str.replace("^chr", "", regex=True)
        .str.replace(r"\.0$", "", regex=True)
    )
    gmap_tmp["POS"] = pd.to_numeric(gmap_tmp["POS"], errors="coerce")
    gmap_tmp["cM"] = pd.to_numeric(gmap_tmp["cM"], errors="coerce")
    gmap_tmp = gmap_tmp.dropna(subset=["CHROM", "POS", "cM"]).copy()
    gmap_tmp = gmap_tmp.sort_values(["CHROM", "POS"]).drop_duplicates(["CHROM", "POS"])

    vcf_phased["cM"] = np.nan

    for chrom, idx in vcf_phased.groupby("CHROM").groups.items():
        gm = gmap_tmp[gmap_tmp["CHROM"] == str(chrom)]
        if gm.empty:
            continue

        snp_idx = list(idx)
        snp_pos = vcf_phased.loc[snp_idx, "POS"].astype(float).to_numpy()
        map_pos = gm["POS"].astype(float).to_numpy()
        map_cm = gm["cM"].astype(float).to_numpy()

        if len(map_pos) == 1:
            vcf_phased.loc[snp_idx, "cM"] = map_cm[0]
        else:
            vcf_phased.loc[snp_idx, "cM"] = np.interp(
                snp_pos,
                map_pos,
                map_cm,
                left=map_cm[0],
                right=map_cm[-1],
            )

    # Merge phased annotations into cell-wise counts and filter hets
    df = df.merge(
        vcf_phased[["snp_id", "gene", "GT", "cM"]],
        on="snp_id",
        how="left",
    )

    df_out = df[["cell", "snp_id", "CHROM", "POS", "cM", "REF", "ALT", "AD", "DP", "GT", "gene"]]
    df_out = df_out[df_out["GT"].isin(["1|0", "0|1"])].reset_index(drop=True)

    return df_out


def main():
    parser = argparse.ArgumentParser(description="Run SNP pileup and phasing with 1000G")
    parser.add_argument("--label", default="subject")
    parser.add_argument("--samples", default="sample")
    parser.add_argument("--bams", required=True)
    parser.add_argument("--barcodes")
    parser.add_argument("--gmap", required=True)
    parser.add_argument("--eagle", default="eagle")
    parser.add_argument("--snpvcf", required=True)
    parser.add_argument("--paneldir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--genome", choices=["hg38", "hg38_old"], default="hg38", help=(
        "Packaged genome annotation to use when --gtf is not supplied. "
        "Default: hg38."))
    parser.add_argument("--gtf", default=None, help=(
        "Custom genomic annotation TSV with columns CHROM, gene_start, "
        "gene_end and gene. Overrides --genome. The annotation must use "
        "the same genome build as --snpvcf, --gmap and --paneldir."))
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument("--UMItag", default="Auto")
    parser.add_argument("--cellTAG", default="CB")
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--smartseq", action="store_true")
    mode_group.add_argument("--bulk", action="store_true")
    
    parser.add_argument("--modalities", default=None, help=(
        "Comma-separated modality labels, one per sample, "
        "for example: rna,atac."),)

    args = parser.parse_args()
    
    # Annotation
    gtf = load_annotation(gtf_path=args.gtf, genome=args.genome)

    # Parse inputs
    samples = _split_csv(args.samples)
    bams = _split_csv(args.bams)
    barcodes = _split_csv(args.barcodes)
    
    umi_tags = _split_csv(args.UMItag)
    if len(umi_tags) == 1:
        umi_tags *= len(samples)
    
    if not args.bulk and not args.smartseq:
        _validate_10x_inputs(samples=samples,
                             bams=bams,
                             barcodes=barcodes,
                             umi_tags=umi_tags,
                             )
    
    modalities = _split_csv(args.modalities)
    multi_sample = len(samples) > 1
    
    if multi_sample:
    
        if not modalities:
            raise ValueError("--modalities is required when multiple samples are supplied. "
                             "Provide one modality per sample, for example: "
                             "--modalities rna,atac.")
        if len(modalities) != len(samples):
            raise ValueError("--modalities must contain one value per sample. "
                             f"Received {len(modalities)} modalities for "
                             f"{len(samples)} samples.")
    
        modalities = [modality.strip().lower() for modality in modalities]
        invalid = set(modalities) - {"rna", "atac"}
    
        if invalid:
            raise ValueError(f"Invalid modality labels: {sorted(invalid)}. "
                             "Allowed values are 'rna' and 'atac'.")
    else:
        # In single-library behavior no modality namespace is required.
        modalities = [None]
        
    # Validate reference inputs.
    for path, name in [(args.snpvcf, "--snpvcf"), (args.gmap, "--gmap")]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} file not found: {path}")

    if not os.path.isdir(args.paneldir):
        raise FileNotFoundError(f"--paneldir directory not found: {args.paneldir}")

    # Output directories
    os.makedirs(args.outdir, exist_ok=True)

    pileup_dir = os.path.join(args.outdir, "pileup")
    phasing_dir = os.path.join(args.outdir, "phasing")

    os.makedirs(pileup_dir, exist_ok=True)
    os.makedirs(phasing_dir, exist_ok=True)

    for sample in samples:
        os.makedirs(os.path.join(pileup_dir, sample), exist_ok=True)
    
    ## Pileup
    cmds = []
    if args.bulk:
        for sample, bam in zip(samples, bams):
            bam_file = os.path.join(args.outdir, "pileup", sample, "bam_path.tsv")
            sample_file = os.path.join(args.outdir, "pileup", sample, "sample.tsv")
            with open(bam_file, "w") as fh:
                fh.write(bam + "\n")
            with open(sample_file, "w") as fh:
                fh.write(sample + "\n")
            cmd = [
                "cellsnp-lite",
                "-S", bam_file,
                "-i", sample_file,
                "-O", os.path.join(pileup_dir, sample),
                "-R", args.snpvcf,
                "-p", str(args.ncores),
                "--minMAF", "0",
                "--minCOUNT", "2",
                "--UMItag", "None",
                "--cellTAG", "None",
            ]
            cmds.append(" ".join(cmd))
    elif args.smartseq:
        cmd = [
            "cellsnp-lite",
            "-S", args.bams,
            "-i", args.barcodes,
            "-O", os.path.join(pileup_dir, samples[0]),
            "-R", args.snpvcf,
            "-p", str(args.ncores),
            "--minMAF", "0",
            "--minCOUNT", "2",
            "--UMItag", "None",
            "--cellTAG", "None",
        ]
        cmds.append(" ".join(cmd))
    else:
        for sample, bam, bc, tag in zip(samples, bams, barcodes, umi_tags):
            cmd = [
                "cellsnp-lite",
                "-s", bam,
                "-b", bc,
                "-O", os.path.join(pileup_dir, sample),
                "-R", args.snpvcf,
                "-p", str(args.ncores),
                "--minMAF", "0",
                "--minCOUNT", "2",
                "--UMItag", tag,
                "--cellTAG", args.cellTAG,
            ]
            cmds.append(" ".join(cmd))
            
    print("Running pileup\n")

    pileup_script = os.path.join(args.outdir, "run_pileup.sh")

    with open(pileup_script, "w") as fh:
        # -e stops early at failure
        fh.write("set -e\n")

        for cmd in cmds:
            fh.write(cmd + "\n")

    pileup_log = os.path.join(args.outdir, "pileup.log")

    try:
        with open(pileup_log, "w") as log:
            subprocess.run(
                ["sh", pileup_script],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "cellSNP-lite pileup failed. "
            f"See log file: {pileup_log}") from exc

    # Validate pileup output
    vcfs = []

    for sample in samples:
        vcf = os.path.join(pileup_dir, sample, "cellSNP.base.vcf")

        if not os.path.isfile(vcf):
            raise FileNotFoundError(
                f"Pileup VCF not found for sample {sample!r}: "
                f"{vcf}")

        if os.path.getsize(vcf) == 0:
            raise RuntimeError(
                f"Pileup VCF is empty for sample {sample!r}: "
                f"{vcf}")

        vcfs.append(vcf)

    # Joint genotyping
    print("Creating joint genotype VCF")
    genotype(args.label, vcfs, phasing_dir, chr_prefix=True)

    ## Phasing
    print("Running phasing\n")
    phasing_cmds = []
    for chr_num in range(1, 23):
        target_vcf = os.path.join(phasing_dir, f"{args.label}_chr{chr_num}.vcf.gz")
        reference_bcf = os.path.join(args.paneldir, f"chr{chr_num}.genotypes.bcf")
        out_prefix = os.path.join(phasing_dir, f"{args.label}_chr{chr_num}.phased")

        phasing_cmds.append(
            " ".join([
                args.eagle,
                "--numThreads", str(args.ncores),
                "--vcfTarget", target_vcf,
                "--vcfRef", reference_bcf,
                f"--geneticMapFile={args.gmap}",
                "--outPrefix", out_prefix]))
        
    phasing_script = os.path.join(args.outdir, "run_phasing.sh")
    
    
    with open(phasing_script, "w") as fh:
        fh.write("set -e\n")

        for cmd in phasing_cmds:
            fh.write(cmd + "\n")

    phasing_log = os.path.join(args.outdir, "phasing.log")

    try:
        with open(phasing_log, "w") as log:
            subprocess.run(
                ["sh", phasing_script],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True)

    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "Eagle phasing failed. "
            f"See log file: {phasing_log}") from exc
        
   # Generate allele-count dataframes
    print("Generating allele count dataframes...")

    # Concatenate all phased chromosomes once (same phased VCF used for all samples)
    vcf_phased_all = load_phased_concat(args.outdir, args.label)
    # Put the single target-sample phased GT column into a named column for convenience
    # FORMAT is column 8, sample is column 9 if a single target sample
    # If Eagle produced exactly one sample (the target), its GT is in column 10 (0-based index=9)
    # We keep only CHROM, POS, REF, ALT, and sample GT
    if vcf_phased_all.shape[1] < 10:
        # FORMAT + one sample expected; if not, raise for clarity
        raise RuntimeError("Unexpected phased VCF structure: FORMAT/sample columns missing.")
    vcf_phased_all = vcf_phased_all.rename(columns={8: "FORMAT", 9: args.label})
    vcf_phased_all = vcf_phased_all.loc[:, ["CHROM", "POS", "REF", "ALT", args.label]]

    allele_tables = []
    cell_manifests = []
    namespace_cells = multi_sample
    
    for sample, modality in zip(samples, modalities):
    
        sample_pileup_dir = os.path.join(pileup_dir, sample)
        vcf_pu = load_pileup_body(sample_pileup_dir)
        AD, DP, raw_barcodes = read_cellsnp_mtx(sample_pileup_dir)
    
        # Multiple libraries require a namespace because identical
        # 10x barcodes can occur in different experiments.
        if multi_sample:
    
            cell_manifest = make_cell_manifest(sample=sample,
                                               modality=modality,
                                               barcodes=raw_barcodes,
                                               namespace=namespace_cells)
            allele_barcodes = cell_manifest["cell"].tolist()
    
        else:
            # single-library behavior preserve the original barcode exactly.
            cell_manifest = None
            allele_barcodes = raw_barcodes
    
        df_allele = preprocess_allele(
            sample=args.label,
            vcf_pu=vcf_pu.rename(columns={7: "INFO"}),
            vcf_phased=vcf_phased_all.copy(),
            AD=AD,
            DP=DP,
            barcodes=allele_barcodes,
            gmap=args.gmap,
            gtf=gtf,
            )
    
        # namespace metadata are necessary only when
        # multiple libraries must be distinguished.
        if multi_sample:
    
            metadata = cell_manifest.set_index("cell")
    
            df_allele["barcode"] = df_allele["cell"].map(metadata["barcode"])
            df_allele["cell_id"] = df_allele["cell"].map(metadata["cell_id"])
            df_allele["library"] = df_allele["cell"].map(metadata["library"])
            df_allele["modality"] = df_allele["cell"].map(metadata["modality"])
            cell_manifests.append(cell_manifest)
    
        allele_file = os.path.join(args.outdir,f"{sample}_allele_counts.tsv.gz")
        df_allele.to_csv(allele_file, sep="\t", index=False, compression="gzip", na_rep="nan")
        allele_tables.append(df_allele)
        
        cell_manifests.append(cell_manifest)
        
    # Cell manifest
    if multi_sample:

        cell_manifest = pd.concat(cell_manifests, ignore_index=True)
        cell_manifest.to_csv(os.path.join(args.outdir, f"{args.label}_cell_manifest.tsv.gz"),
                             sep="\t",
                             index=False,
                             compression="gzip")
    
        df_allele_combined = pd.concat(allele_tables, ignore_index=True)
    
        df_allele_combined.to_csv(os.path.join(args.outdir,f"{args.label}_allele_counts.tsv.gz"),
                                  sep="\t",
                                  index=False,
                                  compression="gzip",
                                  na_rep="nan")
    
    print("All done!")


if __name__ == "__main__":
    main()
    
    
    
    
    
    
