<img src="https://github.com/lillux/spacenumbat/blob/alpha/pics/space_geom_trasp.png" width="200" title="spacenumbat logo" alt="spacenumbat logo"/>

# spacenumbat

`spacenumbat` is a haplotype-aware copy-number alterations (CNA) inference library for single-cell and spatial transcriptomics data. 

`spacenumbat` is a Python porting of the R implementation of [`Numbat`](https://github.com/kharchenkolab/numbat), originally developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab).

Our implementation expands the original algorithm by including an optional spatial signal enhancement algorithm that can be used for the analysis of spatial transcriptomics data.
`spacenumbat` is compatible with the [scverse](https://scverse.org/) ecosystem, and is developed by the [λ Lab](https://research.hsr.it/en/centers/omics-sciences/lambda-lab.html).

As the original R implementation, to recover tumor subclones and their CNA genotypes `spacenumbat` combines:

- **Expression-derived CNA signal** (gene-level count shifts),
- **Allele-specific signal** (allelic imbalance),
- **Phylogenetic structure** (clone relationships inferred from per-cell CNA posteriors),




## Spatial algorithm

To denoise segment-level CNA signals across spatial transcriptomics spots, we implemented a method to perform graph-based diffusion on a spatially constrained affinity graph, defined by the argument `"spatial_method" = "cpr"` in the main pipeline: `spacenumbat.run_spacenumbat()`.\
Spots were connected using the tissue graph adjacency map, and edge weights were modulated by a kernel of pairwise distance calculated between the CNAs probability vector of connected spots.


Let

$$
A
$$

denote the resulting weighted adjacency matrix, and let

$$
d_i = \sum_j A_{ij}
$$

be the node degrees. To reduce bias induced by nonuniform sampling density, we applied the anisotropic normalization of Coifman,

$$
W = D^{-\beta} A D^{-\beta}, \qquad \beta = 0.5,
$$

followed by row normalization to obtain a Markov transition matrix $P$. For a matrix of spot-wise CNA features $X$, we then computed a personalized PageRank diffusion $Z$ by iterating

$$
Z^{(t+1)} = \alpha P Z^{(t)} + (1-\alpha)X,
$$

initialized at 

$$
Z^{(0)} = X
$$

This procedure is a random walk with restart and yields a density-corrected, locality-preserving smoother that borrows information across neighboring spots while retaining fidelity to the original measurements. This regularization is aimed to enhances spatially coherent clonal CNA patterns and reduce technical noise without enforcing global homogenization.

# Installation

`spacenumbat` is currently available for download at its GitHub [repo](https://github.com/lillux/spacenumbat).

Installation in a [miniforge](https://github.com/conda-forge/miniforge) environment is suggested.

### Conda env creation

An *env* called *space* can be created with:
```bash
conda create -n space python=3.13 pip
```

The env can be accessed with:
```bash
conda activate space
```

### `spacenumbat` installation

Once in your env, the library can be istalled using `pip` in two ways:

### Editable mode

Clone the library from GitHub with:

```bash
git clone https://github.com/lillux/spacenumbat.git
```

```bash
cd spacenumbat
```

```bash
pip install -e .
```

### From github

Install `spacenumbat` directly from GitHub:

```bash
pip install git+https://github.com/lillux/spacenumbat.git#egg=spacenumbat
``` 

### Required libraries

To run the preprocessing step, consisting in SNPs pileup and allele phasing, the following tools are required:

- [`samtools`](https://www.htslib.org/)
- [`cellsnp-lite`](https://github.com/single-cell-genetics/cellsnp-lite)
- [`eagle2`](https://alkesgroup.broadinstitute.org/Eagle/)

`samtools` and `cellsnp-lite` can be installed with `conda` in your active env:

```bash
conda install samtools cellsnp-lite -c conda-forge
```

`Eagle2` can be found at the following link: [`Eagle2`](https://alkesgroup.broadinstitute.org/Eagle/downloads/), where the `Eagle_v2.4.1.tar.gz` file can be download.\
It contains the executable file `eagle` and the tables required by `spacenumbat` preprocessing.


At April 2026 some dependencies are outdated on conda, but can be installed through `pip`, specifically:

- [`spatialdata`](https://spatialdata.scverse.org/en/stable/)
- [`squidpy`](https://squidpy.readthedocs.io/en/stable/)

``` bash
pip install spatialdata spatialdata_io spatialdata_plot squidpy
```

### Required panels

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

# Data preprocessing
 The script `spacenumbat/preprocessing/pileup_n_phase.py` is used to perform allele data preprocessing.

`pileup_n_phase.py` has the following arguments:

 - `--label`: label for the current run. One per run.
 - `--samples`: sample name(s). Used to create per-sample pileup directories and to name final output files.
 - `--bams`: Path(s) to input BAM file(s). This is always required. The interpretation depends on the selected mode: one BAM per sample in default and bulk modes, or a BAM list in `--smartseq` mode.
 - `--barcodes` Path(s) to barcode file(s). Required in default single-cell mode and in spatial transcriptomics data. Passed differently in `--smartseq` mode. Ignored in `--bulk` mode.
 - `--gmap`: Path to the genetic map file. Used both by `Eagle2` during phasing and later by Python to interpolate centiMorgan (cM) positions for SNPs. This is provided by the `Eagle2` downloaded with the instruction above, in `Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz`.
 - `--eagle`: Path to the Eagle2 executable. The default assumes eagle is available in the shell PATH. If eagle is not available in shell PATH the correct path to eagle executable should be given.
 - `--snpvcf`: Path to the candidate 1000G SNP VCF used by cellsnp-lite as the pileup target loci. 
 - `--paneldir`: Directory containing Eagle2 reference panel files, expected as `chr1.genotypes.bcf` through `chr22.genotypes.bcf`. This is the path to the directory in which the 1000G Reference Panel downloaded above had been decompressed.
 - `--outdir`: Output directory where the script writes pileup results, phasing files, logs, and final allele-count tables. 
 - `--ncores`: Number of threads to use for both `cellsnp-lite` and `Eagle2`.

Example code to run the script in single-cell mode (this works for spatial transcriptomics and scATAC):

```bash
python /spacenumbat/preprocessing/pileup_n_phase.py \
    --label sample1 \
    --samples sample1 \
    --bams sample1/outs/possorted_genome_bam.bam \
    --barcodes sample1/outs/filtered_feature_bc_matrix/barcodes.tsv \
    --gmap Eagle_v2.4.1/tables/genetic_map_hg38_withX.txt.gz \
    --eagle Eagle_v2.4.1/eagle \
    --snpvcf genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz \
    --paneldir 1000G_hg38 \
    --outdir path/to/out \
    --ncores 16
```

At the end of a succesfull run of preprocessing, in the directory specified in the `--outdir` argument there will be some directories and files, including a file called `{--sample}_allele_counts.tsv.gz` that is required for the `spacenumbat` pipeline. 

# Run

The main entry point is:

- `spacenumbat.run_spacenumbat(...)` (implemented in `spacenumbat/main.py`).

We may use this code as an example of running the `spacenumbat` pipeline after preprocessing:

```python
import pandas as pd
import spacenumbat
import spatialdata_io

10x_spaceranger_outs_path = "sample1/outs"
sample_id = "sample1"
df_allele_path = "sample1_allele_counts.tsv"

counts_mat_space = spatialdata_io.visium(10x_cellranger_outs_path,
                                         dataset_id = sample_id,
                                         var_names_make_unique = False)

counts_mat = counts_mat_space.tables['table'].copy()
lambdas_ref = spacenumbat.data.ref_hca.copy()
df_allele = pd.read_table(df_allele_path, sep='\t')

current_out_path = "path/to/sample1_out"
ncores = 16

sn_out = spacenumbat.run_spacenumbat(count_mat=counts_mat.copy(),
                                     lambdas_ref=lambdas_ref.copy(),
                                     df_allele=df_allele.copy(),
                                     genome="hg38",
                                     ncores=ncores,
                                     call_clonal_loh=True,
                                     filter_hla_hg38=True,
                                     out_dir=current_out_path, 
                                     max_entropy=0.8, 
                                     ncores_nni=ncores, 
                                     spatial=True,
                                     )
```


## `run_spacenumbat()` in detail

### Function signature

```python
spacenumbat.run_spacenumbat(count_mat=counts_mat.copy(),
                            lambdas_ref=lambdas_ref.copy(),
                            df_allele=df_allele.copy(),
                            genome="hg38",
                            ncores=ncores,
                            call_clonal_loh=True,
                            filter_hla_hg38=True,
                            out_dir=current_out_path, 
                            max_entropy=0.8, 
                            ncores_nni=ncores, 
                            spatial=True,
                            )
```

### Core required inputs

- **`count_mat`** (`anndata.AnnData`): expression count matrix (cells × genes in `AnnData` convention).
- **`lambdas_ref`** (`DataFrame`): reference normalized expression profile(s). A reference profile is integrated in the library, and can be found at `spacenumbat.data.ref_hca`.\
It is recommendend to use a reference profiles of euploid samples obtained with the same sequencing technology of the samples to be analyzed.
- **`df_allele`** (`DataFrame`): per-cell allele counts from the allele preprocessing workflow.

### Optional genomic annotation

- `gtf=None` and `genome in {"hg38", "hg19", "mm10"}` uses packaged annotation tables.
- If custom `gtf` is provided, it is validated and used directly.

### Parameters most relevant to CNA prediction quality

- **`min_LLR`**: confidence threshold for CNA retention (higher = stricter).
- **`min_overlap`**: agreement requirement when deriving consensus segments.
- **`max_entropy`**: filters uncertain single-cell CNA calls before phylogeny. Default to 0.5.\
It is recommended to increase it (eg. to 0.8) when analyzing spatial trascriptomics samples with low resolution (big spot with signal from multiple cells, eg. 10X Visium).
- **`min_genes`**: minimum genes per segment for stable calls.
- **`gamma`, `t`, `nu`**: model parameters controlling allele dispersion, transition rate, and phase switching behavior.
- **`multi_allelic`, `p_multi`**: enables and thresholds multi-allelic CNA detection.
- **`min_cells`**: drops very small groups to avoid unstable HMM and phylogeny reconstruction steps.

### Spatial CNA mode (optional)

Set `spatial=True` to integrate the spatial graph connectivity structure in the posterior smoothing. Key options:

#### `spatial_decay`

Implementations of distance-to-weight kernels that transform a *dissimilarity* matrix
(for example, a distance matrix) into an ***affinity*** matrix.

| kind         | Weight function            | Behavior                         | Use when                                     |
| ------------ | -------------------------- | -------------------------------- | -------------------------------------------- |
| `"gaussian"` | $w(d)=\exp(-d^2/\sigma^2)$ | Very local, fast decay           | Sharp local structure, boundary preservation |
| `"exp"`      | $w(d)=\exp(-d/\ell)$       | Broader than Gaussian            | Slightly smoother local borrowing            |
| `"invdist"`  | $w(d)=1/(d+\varepsilon)^p$ | Strong nearest-neighbor emphasis | Nearest neighbors should dominate            |
| `"cauchy"`   | $w(d)=1/(1+(d/\sigma)^2)$  | Robust, moderate tail            | Noisy or heterogeneous distances             |

---
---

#### `spatial_method`

Chooses the method used to perform spatial smoothing of the CNA probability graph.


| method      | Update / rule                                                   | Behavior                                    | Use when                                  |
| ----------- | --------------------------------------------------------------- | ------------------------------------------- | ----------------------------------------- |
| `"degree"`  | $Z_i=\frac{\sum_j (A_{ij}+I_{ij})X_j}{\sum_j (A_{ij}+I_{ij})}$  | One-hop local average                       | Mild local smoothing is enough            |
| `"diffuse"` | $Z^{(t+1)}=\alpha PZ^{(t)}+(1-\alpha)X$                         | Multi-step diffusion with restart           | Spatially coherent signal needs denoising |
| `"cpr"`     | $Z^{(t+1)}=\alpha PZ^{(t)}+(1-\alpha)X$ with density correction | Geometry-aware diffusion, less density bias | Graph density is uneven                   |



## Typical output written to `out_dir`

During execution, `run_spacenumbat` writes intermediate and final files such as:

- `sc_refs.tsv`: Per-cell (or per-spot) reference assignment: for each barcode, which reference profile column from `lambdas_ref` was selected as best matching by correlation.
- `bulk_subtrees_*.tsv`, `bulk_subtrees_retest_*.tsv`: Iteration-level pseudobulk profiles for current subtrees (cell groups).
    - `bulk_subtrees_{i}.tsv`: output after HMM-based group analysis.
    - `bulk_subtrees_retest_{i}.tsv`: same bulks after re-annotation/retest against consensus segments; low-support calls are reset to neutral based on min_LLR. 
- `bulk_clones_*.tsv`, `bulk_clones_final.tsv`: Iteration-level and final pseudobulk profiles for inferred clones.
    - `bulk_clones_{i}.tsv`: clone bulks after HMM + retest in iteration i.
    - `bulk_clones_final.tsv`: final rerun on the end-of-workflow clone definitions (final clone pseudobulk CNA profiles).
- `segs_consensus_*.tsv`: Iteration-level consensus CNA segment table built across groups/samples: merged CNV intervals, overlap-resolved consensus calls, optional retest intervals, and neutral segments filled in; includes segment-level CNA states and prior.
- `exp_post_*.tsv`, `allele_post_*.tsv`, `joint_post_*.tsv`: Per-cell, per-segment, per-state posterior tables:
    - exp_post: expression-only evidence + segment priors -> posterior CNV probabilities.
    - allele_post: allele-count evidence + segment priors -> posterior CNV probabilities.
    - joint_post: merged expression + allele evidence (optionally spatially smoothed) with recomputed joint posterior/state calls.
- `clone_post_*.tsv`: per spot prediction on clone assignment labels and probabilities and tumor/normal labels and probabilities.
- `geno_*.tsv`: per-spot CNAs probability matrix.
- Optional plots (`*.jpg`, `*.png`) when `plot_results=True`.


## What the library does for CNA prediction

At a high level, `spacenumbat` predicts CNAs by iteratively:

1. **Validating and harmonizing inputs** across expression, allele counts, and genome annotation.
2. **Building initial cell groupings** from smoothed expression profiles.
3. **Calling group-level CNAs** with HMM-based segmentation.
4. **Deriving consensus segments** and retesting them.
5. **Computing per-cell posterior probabilities** from expression and allele evidence.
6. **Combining evidence** into joint CNA posteriors (optionally spatially smoothed).
7. **Inferring clone phylogeny**, reassigning cells, and refining clone/subtree definitions.
8. Repeating for `max_iter` iterations, then writing final clone-level profiles and outputs.


# Aknowledgments

`spacenumbat` is developed by the [λ Lab](https://research.hsr.it/en/centers/omics-sciences/lambda-lab.html).

This project is an independent Python implementation of the ideas described in the [`Numbat`](https://github.com/kharchenkolab/numbat) publications and software ecosystem originally developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab).

