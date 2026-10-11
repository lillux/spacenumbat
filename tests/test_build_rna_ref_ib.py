import numpy as np
import pandas as pd

from spacenumbat.preprocessing.build_rna_ref_ib import (
    _aitchison_huber_center,
    _genome_loss,
    _likelihood_collinearity,
    _make_genome_geometry,
    build_rna_reference_ib,
    select_information_bottleneck,
)


def test_huber_center_is_compositional_and_downweights_outlier():
    profiles = np.array(
        [
            [0.70, 0.20, 0.10],
            [0.69, 0.21, 0.10],
            [0.10, 0.10, 0.80],
        ]
    )
    center, weights = _aitchison_huber_center(
        profiles, eps=1e-12, huber_delta=0.5
    )
    assert np.isclose(center.sum(), 1.0)
    assert np.all(center > 0)
    assert weights[2] < weights[:2].min()


def test_genome_geometry_equalizes_chromosomes_and_detects_regional_error():
    geometry = _make_genome_geometry(["1", "1", "1", "2"], [2])
    assert np.isclose(geometry.gene_weights[:3].sum(), 0.5)
    assert np.isclose(geometry.gene_weights[3], 0.5)

    observed = np.array([0.25, 0.25, 0.25, 0.25])
    predicted = np.array([0.35, 0.35, 0.15, 0.15])
    total, clr_loss, track_loss = _genome_loss(
        observed,
        predicted,
        geometry,
        eps=1e-12,
        multiscale_weight=1.0,
    )
    assert total == clr_loss + track_loss
    assert track_loss > 0


def test_likelihood_geometry_flags_nearly_identical_references():
    weights = np.full(4, 0.25)
    separated = np.array(
        [[0.70, 0.10], [0.10, 0.70], [0.10, 0.10], [0.10, 0.10]]
    )
    nearly_equal = np.column_stack(
        [separated[:, 0], separated[:, 0] + np.array([1e-7, -1e-7, 0, 0])]
    )
    nearly_equal /= nearly_equal.sum(axis=0)
    separated_metrics, _ = _likelihood_collinearity(separated, weights)
    equal_metrics, _ = _likelihood_collinearity(nearly_equal, weights)
    assert equal_metrics["min_eigenvalue"] < separated_metrics["min_eigenvalue"]


def test_selection_uses_robust_genome_aware_validation():
    rng = np.random.default_rng(7)
    profiles = {
        "a": np.array([0.65, 0.15, 0.10, 0.05, 0.03, 0.02]),
        "b": np.array([0.05, 0.08, 0.12, 0.30, 0.25, 0.20]),
    }
    counts = []
    rows = []
    for donor in range(6):
        for label in ("a", "b"):
            for context in ("x", "y"):
                counts.append(rng.multinomial(20_000, profiles[label]))
                rows.append(
                    {"label": label, "context": context, "donor": f"d{donor}"}
                )
    selected = select_information_bottleneck(
        np.asarray(counts),
        pd.DataFrame(rows),
        gene_chrom=["1", "1", "1", "2", "2", "2"],
        betas=(1, 8),
        window_sizes=(2, 3),
        n_folds=3,
        n_restarts=1,
        max_iter=200,
        min_retained_fraction=0.9,
    )
    required = {
        "validation_loss",
        "validation_multiscale_loss",
        "excess_validation_loss",
        "condition_number",
        "min_eigenvalue",
        "selected",
    }
    assert required.issubset(selected.curve.columns)
    assert selected.curve.selected.sum() == 1
    assert selected.model.n_clusters == 2


def test_selection_without_valid_folds_prefers_information_retention():
    counts = np.array([[600, 300, 100], [100, 300, 600]])
    obs = pd.DataFrame(
        {
            "label": ["a", "b"],
            "context": ["x", "x"],
            "donor": ["only_donor", "only_donor"],
        }
    )
    selected = select_information_bottleneck(
        counts,
        obs,
        gene_chrom=["1", "1", "2"],
        betas=(1, 16),
        window_sizes=(2,),
        n_restarts=1,
        max_iter=200,
        min_retained_fraction=0.9,
    )
    assert selected.curve.valid_folds.eq(0).all()
    assert selected.model.n_clusters == 2
    assert selected.model.retained_fraction >= 0.9


def test_builder_emits_huber_and_collinearity_diagnostics():
    class MinimalAnnData:
        pass

    rng = np.random.default_rng(11)
    genes = [f"g{i}" for i in range(6)]
    profiles = {
        "a": np.array([0.65, 0.15, 0.10, 0.05, 0.03, 0.02]),
        "b": np.array([0.05, 0.08, 0.12, 0.30, 0.25, 0.20]),
    }
    rows = []
    counts = []
    for donor in range(6):
        for label in ("a", "b"):
            for tissue in ("x", "y"):
                rows.append(
                    {
                        "donor": f"d{donor}",
                        "tissue": tissue,
                        "free_annotation": label,
                        "method": "10x",
                    }
                )
                counts.append(rng.multinomial(5_000, profiles[label]))
    adata = MinimalAnnData()
    adata.X = np.asarray(counts)
    adata.layers = {}
    adata.obs = pd.DataFrame(rows, index=[f"c{i}" for i in range(len(rows))])
    adata.var = pd.DataFrame(index=genes)
    adata.obs_names = adata.obs.index
    adata.var_names = adata.var.index
    adata.n_obs, adata.n_vars = adata.X.shape
    gtf = pd.DataFrame(
        {
            "gene": genes,
            "CHROM": ["1"] * 3 + ["2"] * 3,
            "gene_start": [1, 2, 3, 1, 2, 3],
            "gene_end": [1, 2, 3, 1, 2, 3],
        }
    )
    reference, manifest, _, diagnostics = build_rna_reference_ib(
        adata,
        gtf=gtf,
        family_mapping={"a": "ab", "b": "ab"},
        min_cells=1,
        min_counts=1,
        min_donors=2,
        min_context_donors=2,
        min_cell_counts=1,
        min_cell_genes=1,
        ib_betas=(1, 8),
        ib_validation_folds=3,
        ib_restarts=1,
        ib_max_iter=200,
        genomic_window_sizes=(2, 3),
        return_diagnostics=True,
    )
    assert np.allclose(reference.sum(axis=0), 1.0)
    assert "effective_donors" in manifest
    assert "reference_collinearity" in diagnostics
    assert "reference_collinearity_pairs" in diagnostics
