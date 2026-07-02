<h1>spacenumbat <img src="pics/space_geom_trasp.png" width="200" alt="spacenumbat logo" align="right"></h1>

[![DOI](https://zenodo.org/badge/890859694.svg)](https://doi.org/10.5281/zenodo.19503719)

`spacenumbat` is a haplotype-aware copy-number alterations (CNA) inference library for single-cell and spatial transcriptomics data.

`spacenumbat` is a Python porting of the R implementation of [`Numbat`](https://github.com/kharchenkolab/numbat) originally developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab).

Our implementation expands the original algorithm by including an optional graph-based spatial regularization for spatial transcriptomics data.
`spacenumbat` is compatible with the [scverse](https://scverse.org/) ecosystem, and is developed by the [λ Lab](https://research.hsr.it/en/centers/omics-sciences/lambda-lab.html).

As in the original R implementation, `spacenumbat` combines the following signals to infer tumor subclones and their CNA genotypes:

- **Expression-derived CNA signal** from gene-level count shifts.
- **Allele-specific signal** from allelic imbalance.
- **Phylogenetic structure** inferred from per-cell CNA posteriors.


## Spatial CNA inference

We added few algorithms to integrate a spatial context in CNAs prediction on spatial transcriptomics experiment. 

Set `spatial=True` in `spacenumbat.run_spacenumbat()` to use a spatial graph during CNA inference. The input `AnnData` must contain a spatial adjacency matrix in `count_mat.obsp[connectivity_key]`; the default key is `"spatial_connectivities"`.\
You can calculate the adjacency matrix for your data using [squidpy](https://squidpy.readthedocs.io/en/stable/api.html#module-squidpy.gr).

The default spatial method is `spatial_method="hmrf"`.

Additional graph-smoothing methods (`"degree"`, `"diffuse"`, and `"cpr"`) are available, and they act at a different point in the posterior calculation, operating on clone state label assignment while `spatial_method="hmrf"` operates on segment latent state.


### Potts HMRF regularization (`spatial_method="hmrf"`)

The Hidden Markov Random Field (HMRF) is applied independently to each consensus genomic segment. The data flow is:

1. The pipeline first builds preliminary cell groups from smoothed expression profiles using [hierarchical clustering](https://github.com/lillux/spacenumbat/blob/main/spacenumbat/clustering.py#L230).
2. Group-level pseudobulk HMMs identify CNA segments, which are merged into consensus segments and provide segment-level CNA priors.
3. Per-cell expression and allele likelihoods are computed on those consensus segments, merged, and passed to [`compute_posterior()`](https://github.com/lillux/spacenumbat/blob/main/spacenumbat/operations.py#L1234) to obtain the local non-spatial posterior.
4. For each spot $i$, the six HMRF unary log scores are

$$
s_i = (Z_{n,i}, Z_{loh,i}, Z_{del,i}, Z_{amp,i}, Z_{bamp,i}, Z_{bdel,i}).
$$

5. Mean-field probabilities are initialized deterministically from the local posterior:

$$
q_i^{(0)} = \operatorname{softmax}(s_i).
$$

Preliminary expression clustering influences the HMRF indirectly through the pseudobulk HMM calls, consensus segmentation, and segment-level priors.

For the spots represented in a segment, the implementation extracts the corresponding subgraph from `count_mat.obsp[connectivity_key]`, removes self-loops, symmetrizes it, and converts it to a binary adjacency matrix. It then applies symmetric degree normalization:

$$
\bar{A} = D^{-1/2} A D^{-1/2}.
$$

At iteration $t$, the mean-field proposal and damped update are:

$$
\widetilde{q}_i^{(t+1)} =
\operatorname{softmax}\left(
    s_i + \beta \sum_j \bar{A}_{ij} q_j^{(t)}
\right),
$$

$$
q_i^{(t+1)} =
(1-\delta)q_i^{(t)} + \delta\widetilde{q}_i^{(t+1)},
$$

where $\beta$ is the spatial coupling strength and $\delta$ is the damping fraction. Iteration stops when the maximum absolute probability change is below `tol` or when `max_iter` is reached.

The current defaults are:

```python
spatial_method="hmrf"
spatial_method_kwargs={
    "beta": 0.25,
    "max_iter": 15,
    "tol": 1e-5,
    "damping": 0.5,
}
```

`spatial_method_kwargs["max_iter"]` controls mean-field updates within each segment and is distinct from the top-level `run_spacenumbat(max_iter=...)`, which controls the number of phylogeny-refinement iterations.

After regularization, the canonical posterior columns (`p_neu`, `p_loh`, `p_del`, `p_amp`, `p_bamp`, `p_bdel`, `p_cnv`, and `p_n`) contain the HMRF probabilities. `cnv_state_map` is the six-state HMRF MAP call. The original `cnv_state` candidate-state column is retained. The canonical `Z_*`, `Z`, `Z_cnv`, `Z_n`, and `logBF` columns are updated to be internally consistent with the HMRF posterior, while the raw expression and allele likelihood columns remain available. Downstream entropy and the genotype-probability matrix use the regularized `p_cnv`; candidate-segment filtering still uses the retained `cnv_state` and segment-level `LLR` columns.

HMRF regularization occurs before entropy filtering, construction of the genotype-probability matrix, phylogeny reconstruction, and optional multi-allelic state expansion.




### Clone level graph smoothing method

To denoise clone-level CNA signals across spatial transcriptomics spots, we implemented three methods to perform graph-based diffusion on a spatially constrained affinity graph, defined by the argument `"spatial_method"` equal to `"degree"`, `"diffuse"`, or `"cpr"`.

The pipeline smooths expression and allele log-likelihood summaries before merging the two modalities and recomputing the joint posterior.

For these methods only, [`get_spatial_info()`](https://github.com/lillux/spacenumbat/blob/hmrf/spacenumbat/spatial_utils.py#L21) computes pairwise distances between spot expression-count vectors, restricts them to edges in the spatial adjacency graph, converts the distances to affinities using `spatial_decay`, and stores the resulting weighted graph in `count_mat.obsp[distance_key]`.

For `spatial_method = "cpr"` we implemented the following smoothing.

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

This procedure is a random walk with restart and yields a density-corrected, locality-preserving smoother that borrows information across neighboring spots while retaining fidelity to the original measurements. This regularization is aimed to enhances spatially coherent clonal CNA patterns and reduce technical noise.

# Installation

`spacenumbat` is currently available for download at its GitHub [repository](https://github.com/lillux/spacenumbat).

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

cd spacenumbat

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
 - `--eagle`: Path to the `Eagle2` executable. The default assumes `eagle` is available in the shell PATH. If eagle is not available in shell PATH the correct path to eagle executable should be given.
 - `--snpvcf`: Path to the candidate 1000G SNP VCF used by cellsnp-lite as the pileup target loci. 
 - `--paneldir`: Directory containing `Eagle2` reference panel files, expected as `chr1.genotypes.bcf` through `chr22.genotypes.bcf`. This is the path to the directory in which the 1000G Reference Panel downloaded above had been decompressed.
 - `--outdir`: Output directory where the script writes pileup results, phasing files, logs, and final allele-count tables. 
 - `--ncores`: Number of threads to use for both `cellsnp-lite` and `Eagle2`.

Example code to run the script in single-cell mode (this works for single-cell, spatial transcriptomics and scATAC):

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

At the end of a succesfull run of preprocessing, in the directory specified in the `--outdir` argument there will be some directories and files, including a file called `{--samples}_allele_counts.tsv.gz` which is required for the `spacenumbat` pipeline. 

# Run

The main entry point is:

- `spacenumbat.run_spacenumbat(...)` (implemented in `spacenumbat/main.py`).

We may use this code as an example of running the `spacenumbat` pipeline after preprocessing:

```python
import pandas as pd
import spacenumbat
import spatialdata_io

spaceranger_10x_outs_path = "sample1/outs"
sample_id = "sample1"
df_allele_path = "sample1_allele_counts.tsv". # path to the output file of the preprocessing step.

counts_mat_space = spatialdata_io.visium(spaceranger_10x_outs_path,
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
                                     spatial_method="hmrf",
                                     spatial_method_kwargs={
                                         "beta": 0.25,
                                         "max_iter": 15,
                                         "tol": 1e-5,
                                         "damping": 0.5,
                                         },
                                     )
```


## `run_spacenumbat()` in detail

### Core required inputs

- **`count_mat`** (`anndata.AnnData`): expression count matrix (cells × genes in `AnnData` convention). With `spatial=True`, `count_mat.obsp[connectivity_key]` must contain the spatial graph.
- **`lambdas_ref`** (`DataFrame`): reference normalized expression profile(s). A reference is bundled as `spacenumbat.data.ref_hca`. A euploid reference generated with the same sequencing technology is recommended.
- **`df_allele`** (`DataFrame`): per-cell allele counts from the allele preprocessing workflow.

### Optional genomic annotation

- With `gtf=None`, `genome="hg38"` or `genome="hg38_old"` selects the corresponding packaged annotation table. `"hg38_old"` is kept for compatibility with the hg38 genome version packaged with the R version of `Numbat`.
- If custom `gtf` is provided, it is validated and used directly.

### Parameters most relevant to CNA prediction quality

- **`min_LLR`**: confidence threshold for CNA retention (higher values are stricter).
- **`min_overlap`**: agreement requirement when deriving consensus segments.
- **`max_entropy`**: filters uncertain single-cell CNA calls before phylogeny. Default to 0.5.\
It is recommended to increase it (eg. to 0.8) when analyzing spatial trascriptomics samples with low resolution (big spot with signal from multiple cells, eg. 10X Visium).
- **`min_genes`**: minimum genes per segment for stable calls.
- **`gamma`, `t`, `nu`**: model parameters controlling allele dispersion, transition rate, and phase switching behavior.
- **`multi_allelic`, `p_multi`**: enables and thresholds multi-allelic CNA detection.
- **`min_cells`**: drops very small groups to avoid unstable HMM and phylogeny reconstruction steps.

### Spatial CNA mode (optional)

Set `spatial=True` to integrate the spatial graph connectivity structure, `count_mat.obsp[connectivity_key]`, in the posterior smoothing. Key options:

#### `spatial_method`

Chooses the method used to perform spatial smoothing of the CNA probability graph.

| Method | Where it acts | Behavior |
| --- | --- | --- |
| `"hmrf"` | After local joint-posterior calculation | Segment-wise six-state Potts HMRF; default method |
| `"degree"` | Before joint-posterior calculation | One-hop neighbor averaging of modality-specific log summaries |
| `"diffuse"` | Before joint-posterior calculation | Multi-step random-walk diffusion with restart |
| `"cpr"` | Before joint-posterior calculation | Personalized PageRank-style diffusion with Coifman density correction |


#### `spatial_method_kwargs`

Accepted keys depend on `spatial_method`:

| Method | Accepted keys |
| --- | --- |
| `"hmrf"` | `beta`, `max_iter`, `tol`, `damping` |
| `"degree"` | none |
| `"diffuse"` | `alpha`, `steps` |
| `"cpr"` | `alpha`, `coifman_alpha`, `lazy`, `steps` |


#### `spatial_decay`

Implementations of distance-to-weight kernels that transform a *dissimilarity* matrix
(for example, a distance matrix) into an ***affinity*** matrix.

| kind         | Weight function            | Behavior                         |
| ------------ | -------------------------- | -------------------------------- |
| `"gaussian"` | $w(d)=\exp(-d^2/\sigma^2)$ | Very local, fast decay           |
| `"exp"`      | $w(d)=\exp(-d/\ell)$       | Broader than Gaussian            |
| `"invdist"`  | $w(d)=1/(d+\varepsilon)^p$ | Strong nearest-neighbor emphasis |
| `"cauchy"`   | $w(d)=1/(1+(d/\sigma)^2)$  | Robust, moderate tail            |



## Typical output written to `out_dir`

During execution, `run_spacenumbat` writes intermediate and final files such as:

- `sc_refs.tsv`: Per-cell (and per-spot) reference profile assignment.
- `gexp_roll_wide.h5ad` and `hc_initial_hierarchical_clustering.tsv`: smoothed expression used for the initial hierarchical clustering.
- `bulk_subtrees_*.tsv`, `bulk_subtrees_retest_*.tsv`: Iteration-level pseudobulk profiles for current subtrees (cell groups), before and after retesting consensus segments.
    - `bulk_subtrees_{i}.tsv`: output after HMM-based group analysis.
    - `bulk_subtrees_retest_{i}.tsv`: same bulks after re-annotation/retest against consensus segments; low-support calls are reset to neutral based on min_LLR. 
- `bulk_clones_*.tsv`, `bulk_clones_final.tsv`: Iteration-level and final pseudobulk profiles for inferred clones.
    - `bulk_clones_{i}.tsv`: clone bulks after HMM + retest in iteration i.
    - `bulk_clones_final.tsv`: final rerun on the end-of-workflow clone definitions (final clone pseudobulk CNA profiles).
- `segs_consensus_*.tsv`: Iteration-level consensus CNA segment table built across groups/samples: merged CNV intervals, overlap-resolved consensus calls, optional retest intervals, and neutral segments filled in; includes segment-level CNA states and prior.
- `exp_post_*.tsv`: expression-only per cell posterior
- `allele_post_*.tsv`: allele-only per cell posterior
- `joint_post_*.tsv`: merged joint posterior. In HMRF mode, canonical posterior columns contain the spatially regularized probabilities and include `cnv_state_map`, `hmrf_iterations`, and `hmrf_converged`. Alternative spatial methods overwrite local HMMs calls and no additional columns are created.
- `clone_post_*.tsv`: per-cell or per-spot clone-assignment and tumor/normal probabilities.
- `geno_*.tsv`: per-cell or per-spot CNA probability matrix used for phylogenetic reconstruction.
- Optional plots (`*.jpg`, `*.png`) when `plot_results=True`.


## CNA inference data flow

At a high level, `spacenumbat` iteratively:

1. Validates and harmonizes expression, allele-count, and genomic-annotation inputs.
2. Assigns per-cell reference profiles and builds preliminary groups from smoothed expression.
3. Constructs group pseudobulks and calls group-level CNAs with HMMs.
4. Derives and retests consensus segments, producing segment-level CNA priors.
5. Computes per-cell expression and allele likelihoods on the consensus segments.
6. Merges both modalities and computes the local joint CNA posterior.
7. Applies optional spatial processing: the segment-wise Potts HMRF after local posterior calculation, or legacy graph smoothing before joint posterior calculation.
8. Computes entropy, optionally expands multi-allelic states, filters CNA candidates, and builds the genotype-probability matrix.
9. Infers the clone phylogeny, reassigns cells, and updates clone and subtree definitions.
10. Repeats for `max_iter` iterations and writes final clone-level profiles.

# Postprocessing Functionalities

## Chromosome arm level CNAs burden calculation

We included the module `spacenumbat.cna_postprocessing` to calculate chromosome arms CNAs burden from the `joint_post_*.tsv` output.


# Aknowledgments

`spacenumbat` is developed by the [λ Lab](https://research.hsr.it/en/centers/omics-sciences/lambda-lab.html).

This project is an independent Python implementation of the ideas described in the [`Numbat`](https://github.com/kharchenkolab/numbat) publications and software ecosystem originally developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab).

