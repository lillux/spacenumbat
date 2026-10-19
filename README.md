<h1>spacenumbat <img src="pics/space_geom_trasp.png" width="200" alt="spacenumbat logo" align="right"></h1>

[![DOI](https://zenodo.org/badge/890859694.svg)](https://doi.org/10.5281/zenodo.19503719)

`spacenumbat` is a Python library for haplotype-aware copy-number alteration (CNA) inference and tumor subclone reconstruction from single-cell and spatial data. It combines expression or accessibility counts, allelic imbalance, and phylogenetic inference, with optional graph-based spatial regularization.

This independent Python implementation extends [Numbat](https://github.com/kharchenkolab/numbat), originally developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab). It is compatible with the [scverse](https://scverse.org/) ecosystem and developed by the [λ Lab](https://research.hsr.it/en/centers/omics-sciences/lambda-lab.html).

## Analysis modes

The input modality (`mode`) and the evidence used for inference (`evidence_mode`) are separate settings.

| `mode` | Count input | Reference | Supported evidence |
| --- | --- | --- | --- |
| `"rna"` | Cells/spots × genes in an `AnnData` | Gene-level RNA profiles | `"joint"` or `"expression"` |
| `"rna_bin"` | 10x RNA matrix, features, and barcodes; aggregated into genomic bins | Gene-level RNA profiles, binned automatically | `"joint"` or `"expression"` |
| `"atac_bin"` | ATAC fragments and barcodes; counted in genomic bins | Matching bin-level ATAC profiles | `"joint"` |
| `"combined"` | Separate RNA and ATAC cell populations in a common bin space | RNA and ATAC profiles | `"joint"` |

`mode="atac"` is an alias for `"atac_bin"`. The defaults are `mode="rna"` and `evidence_mode="joint"`.

**Joint** inference combines count and phased allele evidence and requires `df_allele`. **Expression-only** inference requires no allele table or SNP preprocessing and supports RNA inputs only. RNA with phased SNPs remains an RNA analysis; it does not require ATAC data.

## Installation

Python ≥3.10 is required. A dedicated [Miniforge](https://github.com/conda-forge/miniforge) environment is recommended.

```bash
conda create -n space python=3.11 pip
conda activate space

git clone --branch rna_ref --single-branch https://github.com/lillux/spacenumbat.git
cd spacenumbat
pip install -e .
```

For ATAC or combined analysis, install the optional SnapATAC2 dependency:

```bash
pip install -e ".[atac]"
```

Alternatively, install the branch directly:

```bash
pip install "spacenumbat @ git+https://github.com/lillux/spacenumbat.git@rna_ref"
# With ATAC support:
pip install "spacenumbat[atac] @ git+https://github.com/lillux/spacenumbat.git@rna_ref"
```

For the spatial example below:

```bash
pip install spatialdata spatialdata-io squidpy
```

Joint inference additionally needs [samtools](https://www.htslib.org/), [cellsnp-lite](https://github.com/single-cell-genetics/cellsnp-lite), and [Eagle2](https://alkesgroup.broadinstitute.org/Eagle/) for allele preprocessing:

```bash
conda install -c conda-forge -c bioconda samtools cellsnp-lite
```

Download Eagle2 separately and supply its executable and genetic map.

## Allele preprocessing

Skip this section for `evidence_mode="expression"`.

To perform SNPs pileup and allele phasing two reference panels are required:

- [1000G SNP VCF](https://sourceforge.net/projects/cellsnp/files/SNPlist/)

```bash
# hg38
wget https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz
```

- [1000G Reference Panel](http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip)

```bash
# hg38
wget http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip
```

These are:

 - A [candidate SNP VCF](https://sourceforge.net/projects/cellsnp/files/SNPlist/), such as `genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz`, for cellsnp-lite.
 - The [1000G hg38 phasing panel](http://pklab.med.harvard.edu/teng/data/1000G_hg38.zip), extracted into a directory containing `chr1.genotypes.bcf` through `chr22.genotypes.bcf`.

> WARNING!
> `genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz` is a liftover of hg37 genome to hg38, and we found some inconsistencies in it!
> We produced a new `candidate VCF profile` and a new `phasing panel` from 1000 Genomes data phase 3, restoring consistency in the reference.
> We will make these references available soon!

Use `pileup_n_phase.py` to count SNP alleles and phase them. Standard human hg38 analysis requires:

 - `--label`: label for the current run. One per run.
 - `--samples`: sample name(s). Used to create per-sample pileup directories and to name final output files.
 - `--bams`: Path(s) to input BAM file(s). This is always required. The interpretation depends on the selected mode: one BAM per sample in default and bulk modes, or a BAM list in `--smartseq` mode.
 - `--barcodes` Path(s) to barcode file(s). Required in default single-cell mode and in spatial transcriptomics data. Passed differently in `--smartseq` mode. Ignored in `--bulk` mode.
 - `--gmap`: Path to the genetic map file. Used both by `Eagle2` during phasing and later by Python to interpolate centiMorgan (cM) positions for SNPs. This is provided by the `Eagle2` downloaded with the instruction above, in `Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz`.
 - `--eagle`: Path to the `Eagle2` executable. The default assumes `eagle` is available in the shell PATH. If eagle is not available in shell PATH the correct path to eagle executable should be given.
 - `--snpvcf`: Path to the candidate 1000G SNP VCF used by cellsnp-lite as the pileup target loci. 
 - `--paneldir`: Directory containing `Eagle2` reference panel files, expected as `chr1.genotypes.bcf` through `chr22.genotypes.bcf`. This is the path to the directory in which the 1000G Reference Panel downloaded above had been decompressed.
 - `--outdir`: Output directory where the script writes pileup results, phasing files, logs, and final allele-count tables. 
 - `--ncores`: Number of threads to use for both `cellsnp-lite` and `Eagle2`.
 
The resources above have different roles: `--snpvcf` selects pileup loci; `--paneldir` supplies phased haplotypes to Eagle2.

Example code to run the script in single-cell mode (this works for single-cell, spatial transcriptomics and scATAC):

### Single RNA or spatial library

```bash
python -m spacenumbat.preprocessing.pileup_n_phase \
    --label sample1 \
    --samples sample1 \
    --bams sample1/outs/possorted_genome_bam.bam \
    --barcodes sample1/outs/filtered_feature_bc_matrix/barcodes.tsv.gz \
    --snpvcf genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz \
    --paneldir 1000G_hg38 \
    --gmap Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz \
    --eagle Eagle_v2.4.1/eagle \
    --outdir alleles/sample1 \
    --ncores 16
```

The allele table is `alleles/sample1/sample1_allele_counts.tsv.gz`; single-library cell identifiers retain the original barcodes.

### Unpaired RNA and ATAC libraries

Process libraries from the **unpaired biological sample** together to establish shared phasing and unique cell identifiers. Comma-separated arguments must use the same library order.

```bash
python -m spacenumbat.preprocessing.pileup_n_phase \
    --label tumor1 \
    --samples tumor1_rna,tumor1_atac \
    --modalities rna,atac \
    --UMItag UB,None \
    --bams rna/possorted_genome_bam.bam,atac/possorted_bam.bam \
    --barcodes rna/barcodes.tsv.gz,atac/barcodes.tsv.gz \
    --snpvcf genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz \
    --paneldir 1000G_hg38 \
    --gmap Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz \
    --eagle Eagle_v2.4.1/eagle \
    --outdir alleles/tumor1 \
    --ncores 16
```

Use `tumor1_allele_counts.tsv.gz` and `tumor1_cell_manifest.tsv.gz` from that output directory for combined inference. `--UMItag UB,None` selects RNA UMI counting and ATAC read counting; adjust tags to your BAMs. The manifest maps `(modality, barcode)` to a unique `cell_id`, such as `tumor1_rna::AAAC…-1`; allele-table `cell` identifiers use the same namespace. Identical raw barcodes in different modalities represent separate observations.

Additional preprocessing options include `--UMItag`, `--cellTAG`, `--smartseq`, and `--bulk`; inspect `--help` for details. `--prephased` skips Eagle2 when the SNP VCF already contains the sample's known phased heterozygous genotypes, usefull for analysis on inbred mouse genomes.

## Run CNA inference

The entry point is `spacenumbat.run_spacenumbat(...)`. Supply raw, non-negative integer counts, unique cell/feature identifiers, and references from healthy diploid cells. Gene names must match across counts, RNA reference, and annotation.

The following shared setup prepares the bundled RNA reference as a **genes × profiles** numeric matrix. Its packaged table stores feature identifiers in the first column.

```python
import anndata as ad
import pandas as pd
import spacenumbat

rna_ref = spacenumbat.data.ref_hca.copy()
rna_ref = rna_ref.set_index(rna_ref.columns[0])
```

### RNA with phased allele evidence

`rna_counts.h5ad` must contain raw counts in `X`, with cells in rows and genes in columns.

```python
counts = ad.read_h5ad("rna_counts.h5ad")
alleles = pd.read_table("alleles/sample1/sample1_allele_counts.tsv.gz")

result = spacenumbat.run_spacenumbat(
    count_mat=counts,
    lambdas_ref=rna_ref,
    df_allele=alleles,
    mode="rna",
    evidence_mode="joint",
    genome="hg38",
    out_dir="results/rna_joint",
    ncores=16,
)
```

### RNA expression-only inference

```python
counts = ad.read_h5ad("rna_counts.h5ad")

result = spacenumbat.run_spacenumbat(
    count_mat=counts,
    lambdas_ref=rna_ref,
    mode="rna",
    evidence_mode="expression",
    logphi_min=0.25,
    expression_likelihood_weight=1.0,
    multi_allelic=False,
    out_dir="results/rna_expression",
    ncores=16,
)
```

This mode infers neutral, deletion, and amplification states from relative expression. It cannot identify LOH or haplotype-specific alterations. Leave `call_clonal_loh=False` and do not supply `segs_loh` or `use_loh=True`. A neutral expression baseline is required; supply `diploid_chroms` only when supported by independent evidence.

`logphi_min` sets the minimum absolute log2 expression effect used for deletion/amplification alternatives. `expression_likelihood_weight` lies in `(0, 1]`; values below one temper expression evidence.

### Unpaired multiomic RNA + ATAC inference

```python
result = spacenumbat.run_spacenumbat(
    mode="combined",
    evidence_mode="joint",
    lambdas_ref=rna_ref,
    rna_mtx_path="rna/matrix.mtx.gz",
    rna_barcodes_path="rna/barcodes.tsv.gz",
    rna_features_path="rna/features.tsv.gz",
    atac_fragments_path="atac/fragments.tsv.gz",
    atac_barcodes_path="atac/barcodes.tsv.gz",
    atac_reference=spacenumbat.data.ref_atac_numbat.copy(),
    df_allele=pd.read_table("alleles/tumor1/tumor1_allele_counts.tsv.gz"),
    cell_manifest="alleles/tumor1/tumor1_cell_manifest.tsv.gz",
    binning="numbat",
    genome="hg38",
    out_dir="results/unpaired_multiome",
    ncores=16,
)
```

RNA counts and the RNA reference are aggregated into genomic bins; ATAC fragments are counted in the same bins. The workflow concatenates the distinct cell populations and reference panels for CNA and clone inference. **This is an unpaired workflow:** matching RNA and ATAC barcodes are not fused into a paired-cell likelihood. `spatial=True` is currently unsupported for `mode="combined"`.

For individual binned modalities, use:

```python
# RNA bins; add df_allele=... for evidence_mode="joint".
rna_binned = spacenumbat.run_spacenumbat(
    mode="rna_bin",
    evidence_mode="expression",
    lambdas_ref=rna_ref,
    rna_mtx_path="rna/matrix.mtx.gz",
    rna_barcodes_path="rna/barcodes.tsv.gz",
    rna_features_path="rna/features.tsv.gz",
    binning="numbat",
    multi_allelic=False,
    out_dir="results/rna_bin",
    ncores=16,
)

atac_binned = spacenumbat.run_spacenumbat(
    mode="atac_bin",
    atac_fragments_path="atac/fragments.tsv.gz",
    atac_barcodes_path="atac/barcodes.tsv.gz",
    atac_reference=spacenumbat.data.ref_atac_numbat.copy(),
    df_allele=pd.read_table("alleles/atac/atac_allele_counts.tsv.gz"),
    binning="numbat",
    out_dir="results/atac_bin",
    ncores=16,
)
```

The ATAC example assumes a separately preprocessed single ATAC library with unchanged barcodes. When reusing namespaced multi-library allele tables, also supply the corresponding `cell_manifest`.

### Spatial RNA inference

Build the graph before calling SpaceNumbat. This example uses the raw count table and spatial coordinates imported from Space Ranger.

```python
import spatialdata_io
import squidpy as sq

sdata = spatialdata_io.visium(
    "sample1/outs",
    dataset_id="sample1",
    var_names_make_unique=False,
)
counts = sdata.tables["table"].copy()
sq.gr.spatial_neighbors(counts, coord_type="grid")

result = spacenumbat.run_spacenumbat(
    count_mat=counts,
    lambdas_ref=rna_ref,
    df_allele=pd.read_table("alleles/sample1/sample1_allele_counts.tsv.gz"),
    spatial=True,
    connectivity_key="spatial_connectivities",
    spatial_method="hmrf",
    spatial_method_kwargs={
        "beta": 0.25, "max_iter": 15, "tol": 1e-5, "damping": 0.5,
    },
    max_entropy=0.8,
    out_dir="results/spatial_rna",
    ncores=16,
)
```

`max_entropy=0.8` relaxes the default uncertainty filter (`0.5`) for mixed-cell spots; assess the effect on your data. Spatial regularization is also available for expression-only RNA. Binned spatial analyses (`rna_bin`, `atac_bin`) additionally require `spatial_adata`, containing the graph and raw barcode identifiers.

## Algorithms and spatial regularization

The workflow assigns per-cell reference profiles, clusters smoothed count profiles, and fits pseudobulk HMMs. It constructs and retests consensus CNA segments, computes per-cell likelihoods and posteriors, filters uncertain calls, and iteratively reconstructs clone phylogenies. Joint inference includes allelic imbalance and phase-switch modeling; expression-only inference uses count evidence alone.

| `spatial_method` | Operation | `spatial_method_kwargs` |
| --- | --- | --- |
| `"hmrf"` (default) | Segment-wise Potts HMRF after local posterior calculation | `beta`, `max_iter`, `tol`, `damping` |
| `"degree"` | One-hop averaging of modality-specific log summaries before posterior calculation | None |
| `"diffuse"` | Random-walk diffusion with restart before posterior calculation | `alpha`, `steps` |
| `"cpr"` | PageRank-style diffusion with Coifman density correction before posterior calculation | `alpha`, `coifman_alpha`, `lazy`, `steps` |

The HMRF uses a symmetrized, binary graph with self-loops removed and symmetric degree normalization. It updates segment-state probabilities by damped mean-field inference. Canonical posterior columns are replaced by regularized values; `cnv_state_map`, `hmrf_iterations`, and `hmrf_converged` record the result.

For the other methods, expression distances on spatial edges are converted into affinities in `count_mat.obsp[distance_key]` (default `"weighted_adjacency"`). `spatial_decay` selects `"gaussian"`, `"exp"`, `"invdist"`, or `"cauchy"`.

Important inference settings:

- `min_LLR`, `min_overlap`: CNA confidence and consensus overlap requirements.
- `min_genes`, `min_cells`: minimum feature and group support; in binned analyses the features are bins.
- `max_entropy`: uncertainty threshold before phylogeny reconstruction.
- `clustering_window`: rolling window in genomic features, not base pairs.
- `gamma`, `t`, `nu`: allele dispersion, HMM transitions, and phase-switch behavior.
- `max_iter`, `max_nni`: workflow and tree-search iterations. HMRF `spatial_method_kwargs["max_iter"]` controls a separate inner loop.
- `multi_allelic`, `p_multi`: optional multi-allelic CNA testing.

## Build RNA reference profiles

Reference builders accept raw healthy-cell atlas counts and donor, tissue, cell-type, and assay metadata. They aggregate donor/context pseudobulks, restrict compression to biological families within assays, and return normalized gene-by-reference profiles, a manifest, and aligned annotation. Set `return_diagnostics=True` to inspect selection and QC decisions.

| Module in `spacenumbat.preprocessing` | Approach |
| --- | --- |
| `build_rna_ref` | Greedy subtype compression using Jensen–Shannon information relative to donor variability, followed by context splitting and robust donor consensus |
| `build_rna_ref_ot` | Adds constituent-level JS/KL distortion bounds, explicit unshared-context handling, similar-context pooling, family fallbacks, and optional Sinkhorn divergence checks |
| `build_rna_ref_ib` | Conditional information-bottleneck candidates, donor-blocked validation, robust Aitchison/Huber centers, multiscale genomic losses, and likelihood-geometry collinearity checks |

### Information-bottleneck reference

```python
from spacenumbat.preprocessing.build_rna_ref_ib import build_rna_reference_ib
from spacenumbat.preprocessing.build_rna_ref import save_rna_reference

atlas = ad.read_h5ad("healthy_atlas.h5ad")
ref_rna, manifest, gene_gtf, diagnostics = build_rna_reference_ib(
    atlas,
    gtf=spacenumbat.data.hg38.copy(),
    donor_col="donor",
    tissue_col="tissue",
    cell_type_col="free_annotation",
    assay_col="method",
    assays=("10x",),
    ib_min_retained_fraction=0.99,
    ib_max_excess_loss=0.005,
    ib_max_condition_number=1e6,
    ib_min_eigenvalue=1e-4,
    return_diagnostics=True,
)

# The destination must not already exist.
save_rna_reference(ref_rna, manifest, gene_gtf, diagnostics, "references/rna_ib")

# Supply ref_rna and gene_gtf directly to run_spacenumbat:
# lambdas_ref=ref_rna, gtf=gene_gtf
# Reload the saved matrix with:
# ref_rna = pd.read_table("references/rna_ib/ref_rna.tsv", index_col=0)
```

The conditional bottleneck compresses cell-type labels while retaining gene information conditional on tissue/anatomical context. It uses deterministic Blahut–Arimoto updates. Selection additionally evaluates donor-held-out genomic prediction and reference identifiability. If no candidate satisfies every criterion, selection uses a fallback; inspect the selected diagnostics and validate downstream CNA performance. These settings do not request a fixed number of profiles.

### Distortion-bounded and transport checks

```python
from spacenumbat.preprocessing.build_rna_ref_ot import build_rna_reference

ref_rna, manifest, gene_gtf, diagnostics = build_rna_reference(
    atlas,
    gtf=spacenumbat.data.hg38.copy(),
    unshared_context_policy="defer",
    max_merge_js=0.02,
    max_merge_kl=0.05,
    fallback_mode="missing",
    return_diagnostics=True,
)
```

`"defer"` leaves unsupported comparisons unresolved; `"compare"` enables separately guarded comparisons without shared contexts. Add `sinkhorn_embedding="X_pca"` to check cell-population distributions in a common atlas embedding. Sinkhorn checks are optional and depend on the embedding and transport scale. `fallback_mode="separate"` additionally retains eligible family profiles alongside subtype profiles; review the manifest's reference roles.

## Build ATAC reference profiles

The bundled `spacenumbat.data.ref_atac_numbat` matches the packaged hg38 Numbat bins. A custom ATAC reference must be a numeric **bins × diploid profiles** matrix with feature identifiers in its index, generated using the same assembly, intervals, and fragment-counting convention as the sample.

`preprocessing.build_atac_ref.build_atac_reference()` builds this matrix from a healthy scATAC atlas already counted in genomic bins. Its input is an `AnnData` with cells × bins, raw fragment counts in `X`, and unique coordinate-based `var_names`, such as `chr1:0-100000`. The current builder retains human autosomes 1–22.

The algorithm aggregates donor × tissue × cell-type pseudobulks and applies cell-count and fragment-count QC. Within curated biological families, it merges subtypes using tissue-conditioned Jensen–Shannon divergence relative to donor variability. It then tests whether tissue-specific profiles are supported. Final profiles combine independently normalized donors using a median in log-probability space and are normalized to sum to one.

```python
from pathlib import Path
from spacenumbat.preprocessing.build_atac_ref import build_atac_reference

atac_atlas = ad.read_h5ad("healthy_atac_binned.h5ad")
ref_atac, atac_manifest, bin_gtf, atac_diagnostics = build_atac_reference(
    atac_atlas,
    donor_col="donor",
    tissue_col="tissue_ref",
    cell_type_col="cell_type",
    life_stage_col="life_stage",
    life_stage="Adult",
    min_cells=50,
    min_fragments=50_000,
    min_donors=2,
    min_tissue_donors=2,
    subtype_ratio_threshold=1.5,
    tissue_ratio_threshold=2.0,
    return_diagnostics=True,
)

out = Path("references/atac")
out.mkdir(parents=True, exist_ok=False)
ref_atac.to_csv(out / "ref_atac.tsv", sep="\t")
atac_manifest.to_csv(out / "reference_manifest.tsv", sep="\t", index=False)
bin_gtf.to_csv(out / "bin_gtf.tsv", sep="\t", index=False)
for name in ("cell_type_map", "merge_history", "tissue_tests"):
    atac_diagnostics[name].to_csv(out / f"{name}.tsv", sep="\t", index=False)

# Use atac_reference=ref_atac in the ATAC or combined examples above.
# Alternatively: atac_reference="references/atac/ref_atac.tsv"
```

The four metadata columns are required, including `life_stage_col` even when `life_stage=None` disables age filtering. `min_cells` and `min_fragments` apply to pseudobulks; `min_donors` controls final profile support (the API default is one; the example requires two). Without diagnostics, the function returns three objects.

Inspect `atac_manifest` and the subtype/tissue diagnostics before selecting profiles. The ATAC builder uses its own curated label mapping and JS-based rules; the RNA OT and information-bottleneck options do not apply to this API. For inference, select bins matching the atlas reference exactly; coordinate identifiers are mapped to the selected bin IDs automatically, and the pipeline constructs its inference annotation from that bin table.

## Genomes and binning

- `genome="hg38"` uses the packaged 10x 2024 / GENCODE 44 gene annotation. `"hg38_old"` selects the annotation retained for R Numbat compatibility.
- Custom `gtf` accepts a DataFrame or annotation TSV with `gene`, `CHROM`, `gene_start`, and `gene_end` (1-based inclusive). This argument expects the processed table, not a raw GTF file.
- Custom genomes require matching annotation and chromosome lengths: use `chrom_size_fai_path`; ATAC also accepts `snap_chrom_sizes`.
- Autosomes are selected by default. `include_x` and `include_y` require matching features and references; the packaged Numbat bins are autosomal.
- `binning="numbat"` uses packaged hg38 bins. Use `binning="fixed", bin_size=...` for fixed-width bins, or `binning="fixed", custom_binning=...` with `bin_id`, `CHROM`, `start`, `end` (0-based, half-open). ATAC requires a reference built on those exact bins.

Use the same assembly for counts, allele positions, annotation, chromosome sizes, and phasing resources.

## Outputs and postprocessing

On successful completion, the function returns a dictionary containing count data, posterior tables, consensus segments, clone assignments, trees, clone pseudobulks, and `success=True`. Expression-only results have `allele_post=None`. Some unsupported-data conditions terminate early with a message; check the return value before accessing dictionary keys.

Files written to `out_dir` include:

| File | Contents |
| --- | --- |
| `sc_refs.tsv` | Per-cell/spot reference assignments |
| `gexp_roll_wide.h5ad`, `hc_initial_hierarchical_clustering.tsv` | Initial smoothed count profiles and clustering |
| `bulk_subtrees_*.tsv`, `bulk_subtrees_retest_*.tsv` | Group pseudobulks before and after consensus retesting |
| `bulk_clones_*.tsv`, `bulk_clones_final.tsv` | Iteration and final clone pseudobulk CNA profiles |
| `segs_consensus_*.tsv` | Consensus genomic segments and CNA priors |
| `exp_post_*.tsv`, `allele_post_*.tsv`, `joint_post_*.tsv` | Per-cell/spot posterior tables; allele files are joint-mode only |
| `clone_post_*.tsv`, `geno_*.tsv` | Clone-assignment probabilities and phylogeny input matrices |

Logs record inference parameters; `plot_results=True` writes diagnostic plots. The `cna_postprocessing` module resolves expanded CNA states and aggregates segment probabilities into chromosome-arm posteriors using `build_spacenumbat_arm_posteriors()`; `arm_probability_matrices()` exports state-specific matrices.

# Aknowledgments

`spacenumbat` is developed by the [λ Lab](https://research.hsr.it/en/centers/omics-sciences/lambda-lab.html).

This project is an independent Python implementation of the ideas described in the [`Numbat`](https://github.com/kharchenkolab/numbat) publications and software ecosystem originally developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab).

