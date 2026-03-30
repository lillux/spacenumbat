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

To denoise segment-level CNA signals across spatial transcriptomics spots, we performed graph-based diffusion on a spatially constrained affinity graph. Spots were connected using the tissue neighborhood graph, and edge weights were modulated by a kernel of pairwise distance calculated between the CNA soft assignment profiles of connected spots.


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

- `spacenumbat.run_numbat(...)` (implemented in `spacenumbat/main.py`).

---

## What the library does for CNV prediction

At a high level, `spacenumbat` predicts CNVs by iteratively:

1. **Validating and harmonizing inputs** across expression, allele counts, and genome annotation.
2. **Building initial cell groupings** from smoothed expression profiles.
3. **Calling group-level CNVs** with HMM-based segmentation.
4. **Deriving consensus segments** and retesting them.
5. **Computing per-cell posterior probabilities** from expression and allele evidence.
6. **Combining evidence** into joint CNV posteriors (optionally spatially smoothed).
7. **Inferring clone phylogeny**, reassigning cells, and refining clone/subtree definitions.
8. Repeating for `max_iter` iterations, then writing final clone-level profiles and outputs.


---

## `run_numbat()` in detail

## Function signature

```python
spacenumbat.run_numbat(
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
- **`lambdas_ref`** (`DataFrame`/array/mapping): reference normalized expression profile(s). A reference profile is integrated in the library, and can be found at `spacenumbat.data.ref_hca`. It is recommendend to use a reference profile of an euploid sample obtained with the same sequencing technology of the sample to be analyzed.
- **`df_allele`** (`DataFrame`): per-cell allele counts from the allele preprocessing workflow.

## Optional genomic annotation

- `gtf=None` and `genome in {"hg38", "hg19", "mm10"}` uses packaged annotation tables.
- If custom `gtf` is provided, it is validated and used directly.

---

## Parameters most relevant to CNV prediction quality

- **`min_LLR`**: confidence threshold for CNV retention (higher = stricter).
- **`min_overlap`**: agreement requirement when deriving consensus segments.
- **`max_entropy`**: filters uncertain single-cell CNV calls before phylogeny. Default to 0.5. It is recommended to increase it (eg. to 0.8) when analyzing spatial trascriptomics samples with low resolution (big spot with signal from multiple cells).
- **`min_genes`**: minimum genes per segment for stable calls.
- **`gamma`, `t`, `nu`**: model parameters controlling allele dispersion, transition rate, and phase switching behavior.
- **`multi_allelic`, `p_multi`**: enables and thresholds multi-allelic CNV detection.
- **`min_cells`**: drops very small groups to avoid unstable HMM/phylogeny steps.

---

## Spatial CNV mode (optional)

Set `spatial=True` to incorporate neighborhood structure in posterior smoothing. Key options:

- `spatial_method`: `"degree"`, `"weighted"`, `"diffuse"`, or `"cpr"`.
- `spatial_decay`: distance-to-weight kernel (`"gaussian"`, `"exp"`, `"invdist"`, `"cauchy"`).
- `connectivity_key` / `distance_key`: where adjacency info is read from `AnnData`.

---

## Typical output written to `out_dir`

During execution, `run_numbat` writes intermediate and final files such as:

- `sc_refs.tsv`: mapping of cell type reference used for each cell or spot
- `bulk_subtrees_*.tsv`, `bulk_subtrees_retest_*.tsv`
- `bulk_clones_*.tsv`, `bulk_clones_final.tsv`
- `segs_consensus_*.tsv`
- `exp_post_*.tsv`, `allele_post_*.tsv`, `joint_post_*.tsv`
- `clone_post_*.tsv`
- `geno_*.tsv`
- Optional plots (`*.jpg`, `*.png`) when `plot_results=True`

---

## Minimal usage sketch

```python
import spacenumbat

results = spacenumbat.run_numbat(
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

