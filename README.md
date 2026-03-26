# spacenumbat

`spacenumbat` is a haplotype-aware copy-number alterations (CNA) inference library for single-cell and spatial transcriptomics data. 

`spacenumbat` is a Python porting of the R implementation of [`Numbat`](https://github.com/kharchenkolab/numbat), developed by [Teng Gao](https://github.com/teng-gao) and colleagues at the [Kharchenko Lab](https://github.com/kharchenkolab). Our implementation expands the original algorithm by including an optional spatial signal enhancement algorithm, that can be used for the analysis of spatial transcriptomics data.

As the original R implementation, `spacenumbat` combines:

- **Expression-derived CNV signal** (gene-level count shifts),
- **Allele-specific signal** (allelic imbalance), and
- **Phylogenetic structure** (clone relationships inferred from per-cell CNV posteriors),

to recover tumor subclones and their CNV genotypes.


## Spatial algorithm

To denoise segment-level CNA signals across spatial transcriptomics spots, we performed graph-based diffusion on a spatially constrained affinity graph. Spots were connected using the tissue neighborhood graph, and edge weights were modulated by a kernel of pairwise distance between spots transcriptional profiles.

Let $A$ denote the resulting weighted adjacency matrix and $d_i = \sum_j A_{ij}$ the node degrees. To reduce bias induced by nonuniform sampling density, we applied the anisotropic normalization of Coifman, $W = D^{-\beta} A D^{-\beta}$, with $\beta = 0.5$, followed by row normalization to obtain a Markov transition matrix $P$. For a matrix of spot-wise CNA features $X$, we then computed a personalized PageRank diffusion $Z$ by iterating $Z^{(t+1)} = \alpha P Z^{(t)} + (1-\alpha)X$, initialized at $Z^{(0)} = X$. This procedure is a random walk with restart and yields a density-corrected, locality-preserving smoother that borrows information across neighboring spots while retaining fidelity to the original measurements. In the context of tumor tissues, this regularization enhances spatially coherent clonal CNA patterns and suppresses high-frequency technical noise without enforcing global homogenization (boundary preservation).
