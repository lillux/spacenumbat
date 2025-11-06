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
    """Write per-chromosome VCF."""
    header = [
        "##fileformat=VCFv4.2",
        "##source=numbat",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + label,
    ]
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

def genotype(label: str, vcfs: List[str], outdir: str, het_only: bool = False, chr_prefix: bool = True) -> None:
    """Python port of numbat:::genotype."""
    dfs = [load_vcf(v) for v in vcfs]
    snps = pd.concat(dfs)
    snps = snps.groupby(["CHROM", "POS", "REF", "ALT", "snp_id"], as_index=False).agg({"AD": "sum", "DP": "sum", "OTH": "sum"})
    snps["AR"] = snps.AD / snps.DP.replace({0: pd.NA})
    snps = snps.sort_values(["CHROM", "POS"])

    for chr_num in range(1, 23):
        chr_snps = snps[snps.CHROM.astype(int) == chr_num].copy()
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

def preprocess_allele(sample: str, vcf_pu: pd.DataFrame, vcf_phased: pd.DataFrame, AD, DP, barcodes: List[str]) -> pd.DataFrame:
    """Simplified version of numbat:::preprocess_allele."""
    vcf_pu = vcf_pu.copy()
    vcf_pu["snp_id"] = vcf_pu.CHROM.astype(str) + "_" + vcf_pu.POS.astype(str) + "_" + vcf_pu.REF + "_" + vcf_pu.ALT

    dp_df = DP.tocoo()
    ad_df = AD.tocoo()
    rows = []
    for i, j, dp in zip(dp_df.row, dp_df.col, dp_df.data):
        snp_id = vcf_pu.iloc[i].snp_id
        ad = 0
        for i2, j2, ad_val in zip(ad_df.row, ad_df.col, ad_df.data):
            if i2 == i and j2 == j:
                ad = ad_val
                break
        rows.append({"cell": barcodes[j], "snp_id": snp_id, "DP": dp, "AD": ad})
    df = pd.DataFrame(rows)
    df = df.merge(
        vcf_pu.rename(columns={"AD": "AD_all", "DP": "DP_all", "OTH": "OTH_all"})[
            ["snp_id", "CHROM", "POS", "REF", "ALT", "AD_all", "DP_all", "OTH_all"]
        ],
        on="snp_id",
        how="left",
    )
    df["AR"] = df.AD / df.DP.replace({0: pd.NA})
    df["AR_all"] = df.AD_all / df.DP_all.replace({0: pd.NA})
    df = df[(df.DP_all > 1) & (df.OTH_all == 0)].drop_duplicates()

    vcf_phased = vcf_phased.copy()
    vcf_phased["snp_id"] = vcf_phased.CHROM.astype(str) + "_" + vcf_phased.POS.astype(str) + "_" + vcf_phased.REF + "_" + vcf_phased.ALT
    vcf_phased["GT"] = vcf_phased[sample]

    df = df.merge(vcf_phased[["snp_id", "GT"]], on="snp_id", how="left")
    df = df[df.GT.isin(["1|0", "0|1"])]
    return df[["cell", "snp_id", "CHROM", "POS", "REF", "ALT", "AD", "DP", "GT"]]

# Main entry
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

    script = os.path.join(args.outdir, "run_pileup.sh")
    with open(script, "w") as fh:
        for c in cmds:
            fh.write(c + "\n")
    subprocess.run(["chmod", "+x", script])
    subprocess.run(["sh", script], stdout=open(os.path.join(args.outdir, "pileup.log"), "w"))

    vcfs = [os.path.join(args.outdir, "pileup", s, "cellSNP.base.vcf") for s in samples]
    genotype(args.label, vcfs, os.path.join(args.outdir, "phasing"), chr_prefix=True)

    phasing_cmds = []
    for chr_num in range(1, 23):
        phasing_cmds.append(
            " ".join([
                args.eagle,
                f"--numThreads {args.ncores}",
                f"--vcfTarget {os.path.join(args.outdir, 'phasing', args.label)}_chr{chr_num}.vcf",
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

    print("All done!")


if __name__ == "__main__":
    main()