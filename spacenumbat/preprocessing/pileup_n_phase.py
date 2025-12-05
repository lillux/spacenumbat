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


# Utility functions
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
    gmap: pd.DataFrame,
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
    gmap : pandas.DataFrame
        Genetic map with at least columns:
        ['CHROM', 'start', 'end', 'cM'].

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

    vcf_pu["snp_id"] = (vcf_pu["CHROM"].astype(str) + "_"
                        + vcf_pu["POS"].astype(str) + "_"
                        + vcf_pu["REF"].astype(str) + "_"
                        + vcf_pu["ALT"].astype(str))

    # Convert DP and AD sparse matrices into long format
    # DP
    dp_coo = DP.tocoo()
    dp_df = pd.DataFrame({"i": dp_coo.row,
                          "j": dp_coo.col,
                          "DP": dp_coo.data,
                          })
    
    dp_df["cell"] = [barcodes[j] for j in dp_df["j"]]
    snp_ids = vcf_pu["snp_id"].to_numpy()
    dp_df["snp_id"] = snp_ids[dp_df["i"].values]
    dp_df = dp_df.drop(columns=["i", "j"])[["cell", "snp_id", "DP"]]

    # AD
    ad_coo = AD.tocoo()
    ad_df = pd.DataFrame({"i": ad_coo.row,
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
    vcf_phased["snp_id"] = (vcf_phased["CHROM"].astype(str) + "_"
                            + vcf_phased["POS"].astype(str) + "_"
                            + vcf_phased["REF"].astype(str) + "_"
                            + vcf_phased["ALT"].astype(str))
    vcf_phased["GT"] = vcf_phased[sample]

    # Annotate SNPs with gene information via overlaps
    # PyRanges for SNPs
    vcf_phased = vcf_phased.reset_index(drop=True)
    vcf_phased["snp_index_tmp"] = np.arange(len(vcf_phased))

    pr_snps = pr.PyRanges(pd.DataFrame({"Chromosome": vcf_phased["CHROM"].astype(str),
                                        "Start": vcf_phased["POS"].astype(int),
                                        "End": vcf_phased["POS"].astype(int),
                                        "snp_index_tmp": vcf_phased["snp_index_tmp"],
                                        }))

    # PyRanges for genes
    gtf_tmp = gtf.reset_index(drop=True).copy()
    gtf_tmp["gene_index_tmp"] = np.arange(len(gtf_tmp))

    pr_genes = pr.PyRanges(pd.DataFrame({"Chromosome": gtf_tmp["CHROM"].astype(str),
                                         "Start": gtf_tmp["gene_start"].astype(int),
                                         "End": gtf_tmp["gene_end"].astype(int),
                                         "gene_index_tmp": gtf_tmp["gene_index_tmp"],
                                         }))

    ov = pr_snps.join(pr_genes).as_df()
    if not ov.empty:
        ov = ov[["snp_index_tmp", "gene_index_tmp"]]
        ov = ov.merge(vcf_phased[["snp_index_tmp", "snp_id"]],
                      on="snp_index_tmp",
                      how="left",
                      )
        ov = ov.merge(gtf_tmp[["gene_index_tmp", "gene", "gene_start", "gene_end"]],
                      on="gene_index_tmp",
                      how="left",
                      )
        # sort by snp_index_tmp and gene name, keep first gene per SNP
        ov = (ov.sort_values(["snp_index_tmp", "gene"]).drop_duplicates(subset="snp_index_tmp", keep="first"))
        vcf_phased = vcf_phased.merge(ov[["snp_id", "gene", "gene_start", "gene_end"]],
                                      on="snp_id",
                                      how="left",
                                      )
    else:
        vcf_phased["gene"] = np.nan
        vcf_phased["gene_start"] = np.nan
        vcf_phased["gene_end"] = np.nan

    # Annotate SNPs with genetic map cM
    gmap_tmp = pd.read_csv(gmap, sep=" ")
    gmap_tmp.columns = ["CHROM", "POS", "rate", "cM"]
    gmap_tmp = gmap_tmp.reset_index(drop=True).copy()
    gmap_tmp["map_index_tmp"] = np.arange(len(gmap_tmp))
    gmap_tmp["start"] = gmap_tmp["POS"]
    gmap_tmp["end"] = (gmap_tmp.groupby("CHROM", sort=False)["POS"].transform(lambda s: s.shift(-1).fillna(s.iloc[-1])))

    pr_snps2 = pr.PyRanges(
        pd.DataFrame({"Chromosome": vcf_phased["CHROM"].astype(str),
                      "Start": vcf_phased["POS"].astype(int),
                      "End": vcf_phased["POS"].astype(int),
                      "marker_index_tmp": vcf_phased["snp_index_tmp"],
                      }))

    pr_map = pr.PyRanges(pd.DataFrame({"Chromosome": gmap_tmp["CHROM"].astype(str),
                                       "Start": gmap_tmp["start"].astype(int),
                                       "End": gmap_tmp["end"].astype(int),
                                       "map_index_tmp": gmap_tmp["map_index_tmp"],
                                       "cM": gmap_tmp["cM"].astype(float),
                                       }))

    ov_map = pr_snps2.join(pr_map).as_df()
    if not ov_map.empty:
        # PyRanges join gives Start/End for SNP (un-suffixed) and map (Start_b/End_b)
        ov_map = ov_map[["marker_index_tmp", "Start_b", "cM"]]
        ov_map = (ov_map.sort_values(["marker_index_tmp", "Start_b"], ascending=[True, False]).drop_duplicates(subset="marker_index_tmp", keep="first"))
        marker_map = ov_map.rename(columns={"marker_index_tmp": "snp_index_tmp"})[["snp_index_tmp", "cM"]]
        vcf_phased = vcf_phased.merge(marker_map, on="snp_index_tmp", how="left")
    else:
        vcf_phased["cM"] = np.nan

    # Merge phased annotations into cell-wise counts and filter hets
    df = df.merge(vcf_phased[["snp_id", "gene", "GT", "cM"]],
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
    parser.add_argument("--ncores", type=int, default=1)
    parser.add_argument("--UMItag", default="Auto")
    parser.add_argument("--cellTAG", default="CB")
    parser.add_argument("--smartseq", action="store_true")
    parser.add_argument("--bulk", action="store_true")

    args = parser.parse_args()

    samples = args.samples.split(",") if args.samples else []
    bams = args.bams.split(",")
    barcodes = args.barcodes.split(",") if args.barcodes else []
    UMItag = args.UMItag.split(",") if "," in args.UMItag else [args.UMItag] * len(samples)

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "pileup"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "phasing"), exist_ok=True)
    for s in samples:
        os.makedirs(os.path.join(args.outdir, "pileup", s), exist_ok=True)
    
    
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
                "-O", os.path.join(args.outdir, "pileup", sample),
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
            "-O", os.path.join(args.outdir, "pileup", samples[0]),
            "-R", args.snpvcf,
            "-p", str(args.ncores),
            "--minMAF", "0",
            "--minCOUNT", "2",
            "--UMItag", "None",
            "--cellTAG", "None",
        ]
        cmds.append(" ".join(cmd))
    else:
        for sample, bam, bc, tag in zip(samples, bams, barcodes, UMItag):
            cmd = [
                "cellsnp-lite",
                "-s", bam,
                "-b", bc,
                "-O", os.path.join(args.outdir, "pileup", sample),
                "-R", args.snpvcf,
                "-p", str(args.ncores),
                "--minMAF", "0",
                "--minCOUNT", "2",
                "--UMItag", tag,
                "--cellTAG", args.cellTAG,
            ]
            cmds.append(" ".join(cmd))
            
    print("Running pileup\n")

    script = os.path.join(args.outdir, "run_pileup.sh")
    with open(script, "w") as fh:
        for c in cmds:
            fh.write(c + "\n")
    subprocess.run(["chmod", "+x", script])
    subprocess.run(["sh", script], stdout=open(os.path.join(args.outdir, "pileup.log"), "w"))

    vcfs = [os.path.join(args.outdir, "pileup", s, "cellSNP.base.vcf") for s in samples]
    genotype(args.label, vcfs, os.path.join(args.outdir, "phasing"), chr_prefix=True)


    ## Phasing
    print("Running phasing\n")
    phasing_cmds = []
    for chr_num in range(1, 23):
        phasing_cmds.append(
            " ".join([
                args.eagle,
                f"--numThreads {args.ncores}",
                f"--vcfTarget {os.path.join(args.outdir, 'phasing', args.label)}_chr{chr_num}.vcf.gz",
                f"--vcfRef {os.path.join(args.paneldir, f'chr{chr_num}.genotypes.bcf')}",
                f"--geneticMapFile={args.gmap}",
                f"--outPrefix {os.path.join(args.outdir, 'phasing', args.label)}_chr{chr_num}.phased",
            ])
        )
    script = os.path.join(args.outdir, "run_phasing.sh")
    with open(script, "w") as fh:
        for c in phasing_cmds:
            fh.write(c + "\n")
    subprocess.run(["chmod", "+x", script])
    subprocess.run(script, shell=True, stdout=open(os.path.join(args.outdir, "phasing.log"), "w"))
    
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

    for sample in samples:
        pu_dir = os.path.join(args.outdir, "pileup", sample)

        # pileup VCF (strip 'chr' to match phased table)
        vcf_pu = load_pileup_body(pu_dir)

        # matrices and barcodes
        AD, DP, barcodes = read_cellsnp_mtx(pu_dir)

        # preprocess allele counts (filters to GT in {'1|0','0|1'} inside the function)
        df_allele = preprocess_allele(
            sample=args.label,
            vcf_pu=vcf_pu.rename(columns={7: "INFO"}),
            vcf_phased=vcf_phased_all.copy(),
            AD=AD,
            DP=DP,
            barcodes=barcodes,
            gmap=args.gmap, 
            gtf=spacenumbat.data.hg38 # TODO: Hardcoded hg38. Make it generic
        )

        out_tsv_gz = os.path.join(args.outdir, f"{sample}_allele_counts.tsv.gz")
        df_allele.to_csv(out_tsv_gz, sep="\t", index=False, compression="gzip")

    print("All done!")


if __name__ == "__main__":
    main()
    
    
