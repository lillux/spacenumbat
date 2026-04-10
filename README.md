# spacenumbat

`spacenumbat` is a haplotype-aware copy-number alterations (CNA) inference library for single-cell and spatial transcriptomics data. 

`spacenumbat` is a Python porting of the R implementation of [`Numbat`](https://github.com/kharchenkolab/numbat), originally developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab). Our implementation expands the original algorithm by including an optional spatial signal enhancement algorithm, that can be used for the analysis of spatial transcriptomics data. 
`spacenumbat` is compatible with the [scverse](https://scverse.org/) ecosystem.

As the original R implementation, `spacenumbat` combines:

- **Expression-derived CNA signal** (gene-level count shifts),
- **Allele-specific signal** (allelic imbalance), and
- **Phylogenetic structure** (clone relationships inferred from per-cell CNA posteriors),

to recover tumor subclones and their CNA genotypes.


## Spatial algorithm

To denoise segment-level CNA signals across spatial transcriptomics spots, we implemented methods to perform graph-based diffusion on a spatially constrained affinity graph, including a Personalized PageRank-style diffusion with Coifman density correction, defined by the argument `"spatial_method" = "cpr"` in the main pipeline: `"spacenumbat.run_spacenumbat()"`.\
Spots were connected using the tissue graph adjacency map, and edge weights were modulated by a kernel of pairwise distance calculated between the CNA soft assignment profiles of connected spots.


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

This procedure is a random walk with restart and yields a density-corrected, locality-preserving smoother that borrows information across neighboring spots while retaining fidelity to the original measurements. In the context of tumor tissues, this regularization is aimed to enhances spatially coherent clonal CNA patterns and suppresses high-frequency technical noise without enforcing global homogenization.

# Installation

`spacenumbat` is currently available for download at its GitHub [repo](https://github.com/lillux/spacenumbat).

Installation in a [miniforge](https://github.com/conda-forge/miniforge) environment is suggested.

## Conda env creation

An *env* called *space* can be created with:
```bash
conda create -n space python=3.13 pip
```

The env can be accessed with:
```bash
conda activate space
```

## `spacenumbat` installation

Once in your env, the library can be istalled using `pip` in two ways:

### Editable mode

Clone the library from GitHub with:
```bash
git clone https://github.com/lillux/spacenumbat.git
```

```
cd spacenumbat
```

```
pip install -e .
```

### From github

Install `spacenumbat` directly from GitHub:

```bash
pip install git+https://github.com/lillux/spacenumbat.git#egg=spacenumbat
``` 

### Required libraries

Some dependencies are required to run `spacenumbat` that can be installed through `pip`, specifically:

``` bash
pip install spatialdata spatialdata_io spatialdata_plot squidpy
```

# Run

The main entry point is:

- `spacenumbat.run_spacenumbat(...)` (implemented in `spacenumbat/main.py`).

---

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


---

## `run_spacenumbat()` in detail

## Function signature

```python
spacenumbat.run_spacenumbat(
    count_mat,
    lambdas_ref,
    df_allele,
    gtf=None,
    genome='hg38',
    out_dir="path/to/dir",
    ...
)
```

## Core required inputs

- **`count_mat`** (`anndata.AnnData`): expression count matrix (cells × genes in `AnnData` convention).
- **`lambdas_ref`** (`DataFrame`/array/mapping): reference normalized expression profile(s). A reference profile is integrated in the library, and can be found at `spacenumbat.data.ref_hca`. It is recommendend to use a reference profiles of euploid samples obtained with the same sequencing technology of the samples to be analyzed.
- **`df_allele`** (`DataFrame`): per-cell allele counts from the allele preprocessing workflow.

## Optional genomic annotation

- `gtf=None` and `genome in {"hg38", "hg19", "mm10"}` uses packaged annotation tables.
- If custom `gtf` is provided, it is validated and used directly.

---

## Parameters most relevant to CNA prediction quality

- **`min_LLR`**: confidence threshold for CNA retention (higher = stricter).
- **`min_overlap`**: agreement requirement when deriving consensus segments.
- **`max_entropy`**: filters uncertain single-cell CNA calls before phylogeny. Default to 0.5. It is recommended to increase it (eg. to 0.8) when analyzing spatial trascriptomics samples with low resolution (big spot with signal from multiple cells, eg. 10X Visium).
- **`min_genes`**: minimum genes per segment for stable calls.
- **`gamma`, `t`, `nu`**: model parameters controlling allele dispersion, transition rate, and phase switching behavior.
- **`multi_allelic`, `p_multi`**: enables and thresholds multi-allelic CNA detection.
- **`min_cells`**: drops very small groups to avoid unstable HMM and phylogeny reconstruction steps.

---

## Spatial CNA mode (optional)

Set `spatial=True` to incorporate neighborhood structure in posterior smoothing. Key options:

---

### `spatial_decay`

Implementations of distance-to-weight kernels that transform a *dissimilarity* matrix
(for example, a distance matrix) into an ***affinity*** matrix.

**`"gaussian"`**

$w(d)=\exp\left(-d^2/\sigma^2\right)$\
Fast decay and strongly local. Preserves boundaries well.\
**Use when:** sharp local structure matters and leakage across boundaries should be minimized.

**`"exp"`**

$w(d)=\exp\left(-d/\ell\right)$\
Allows moderate borrowing across somewhat more distant neighbors. Less aggressive than Gaussian.\
**Use when:** a smoother, slightly broader local kernel is desired.

**`"invdist"`**

$w(d)=1/(d+\varepsilon)^p$\
Strongly emphasizes very small distances. Scale-free, but can become unstable or overly dominated by near-zero distances.\
**Use when:** nearest-neighbor dominance is explicitly desired and distance values are well behaved.


**`"cauchy"`**

$w(d)=1/(1+(d/\sigma)^2)$\
Bounded and robust. More tolerant of moderate distances than Gaussian, and more stable than inverse-distance near zero.\
**Use when:** distances are noisy or heterogeneous and a robust compromise is needed.

---

### `spatial_method`

Chooses the method used to perform spatial smoothing of the CNA probability graph.


**`"degree"`**

$$Z_i=\frac{\sum_j (A_{ij}+I_{ij})X_j}{\sum_j (A_{ij}+I_{ij})}$$

One-step neighborhood average using the connectivity matrix. Includes self-loops and only immediate neighbors. Does not explicitly use expression distance, only spatial constraint.\
**Cons:** limited to one-hop smoothing.\
**Use when:** a mild local average is sufficient.


**`"diffuse"`**

$$Z^{(t+1)}=\alpha P Z^{(t)}+(1-\alpha)X$$ where $$P=D^{-1}A$$

It is an iterative random-walk diffusion with restart. Performs multi-step smoothing over the graph. `alpha` controls smoothing strength and `steps` controls diffusion depth. The restart term preserves fidelity to the original signal.\
**Cons:** can oversmooth or leak across boundaries if `alpha` or `steps` are too large.\
**Use when:** signals are spatially coherent and moderate denoising is needed.


**`"cpr"`**

$$Z^{(t+1)}=\alpha P Z^{(t)}+(1-\alpha)X$$

with a density-corrected transition operator.\
Performs personalized PageRank-style diffusion with Coifman density correction. Reduces bias toward densely connected regions. `coifman_alpha` controls density correction, and `lazy` adds self-retention to reduce leakage and improve stability.\
**Cons:** more parameters to tune and slightly less direct to interpret.\
**Use when:** graph density is uneven or a more geometry-aware diffusion is desired.

---

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

---

## Minimal usage sketch

```python
import spacenumbat

results = spacenumbat.run_spacenumbat(
    count_mat=count_adata,
    lambdas_ref=reference_profile,
    df_allele=allele_df,
    genome="hg38",
    out_dir="./numbat_out",
    init_k=3,
    max_iter=2,
    min_LLR=5,
    spatial=True,
    max_entropy=0.8,
)
```

