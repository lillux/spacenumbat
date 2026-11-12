"""Regression checks for chromosome-local clonal LOH inference."""

import numpy as np
import pandas as pd
import pytest

from spacenumbat import hmm, utils


def make_bulk(specs):
    """Each specification is (chromosome, number of genes, SNPs/gene, counts)."""
    rows = []
    for chrom, n_genes, n_snps, counts in specs:
        for i in range(n_genes):
            start = 100 + 200_000 * i
            for j in range(max(n_snps, 1)):
                rows.append({
                    "CHROM": chrom, "gene": f"{chrom}_g{i}",
                    "gene_start": start, "gene_end": start + 100_000,
                    "AD": 1.0 if n_snps else np.nan,
                    "DP": 10.0 if n_snps else np.nan,
                    "Y_obs": counts if j == 0 else np.nan,
                    "lambda_ref": 1e-4, "d_obs": 1_000_000,
                    "logFC": np.log2(counts / 100),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def fixed_fits(monkeypatch):
    # Hold genome-wide nuisance estimates constant to isolate the HMM chain.
    monkeypatch.setattr(utils.dist_prob, "fit_lnpois", lambda *args: (0.0, 0.2))
    monkeypatch.setattr(utils, "fit_snp_rate", lambda *args: np.array([100.0, 2.0]))


def test_clonal_loh_decodes_each_chromosome_separately(monkeypatch, fixed_fits):
    bulk = make_bulk([("1", 2, 0, 50), ("2", 3, 0, 50)])
    calls = []

    def decode(model):
        calls.append(len(model["x"]))
        return np.repeat("loh", len(model["x"]))

    monkeypatch.setattr(hmm, "viterbi_loh", decode)
    result = utils.detect_clonal_loh(bulk.sample(frac=1, random_state=4))

    assert calls == [2, 3]
    assert result["CHROM"].tolist() == ["1", "2"]
    assert result["seg"].tolist() == ["1a", "2a"]


def test_neighboring_chromosome_cannot_change_loh_path(fixed_fits):
    bulk = make_bulk([("1", 10, 0, 50), ("2", 2, 1, 65)])
    together = utils.detect_clonal_loh(bulk)
    alone = utils.detect_clonal_loh(bulk.loc[bulk["CHROM"].eq("2")])

    together_chr2 = [] if together is None else together.loc[
        together["CHROM"].eq("2"), ["seg_start", "seg_end"]
    ].to_dict("records")
    alone_chr2 = [] if alone is None else alone[["seg_start", "seg_end"]].to_dict("records")
    assert together_chr2 == alone_chr2


def test_single_gene_loh_decoder():
    model = {
        "x": np.array([0]), "Pi": np.array([[[0.99], [0.01]], [[0.01], [0.99]]]),
        "delta": [0.99, 0.01], "pm": np.array([100.0, 5.0]),
        "pn": np.array([0.1]), "snp_sig": 2.0,
        "y": np.array([50]), "phi": np.array([1.0, 0.5]),
        "lambda_star": np.array([1e-4]), "d": np.array([1_000_000]),
        "mu": 0.0, "sig": 0.2, "states": ["neu", "loh"],
    }

    result = hmm.viterbi_loh(model)
    assert result.shape == (1,)
    assert result[0] in {"neu", "loh"}


def test_clonal_loh_handles_a_single_gene_chromosome(fixed_fits):
    bulk = make_bulk([("1", 1, 0, 50), ("2", 2, 1, 65)])
    result = utils.detect_clonal_loh(bulk)
    assert result is None


@pytest.mark.parametrize("filtered", [False, True])
def test_no_eligible_genes_returns_none(monkeypatch, filtered):
    bulk = make_bulk([("1", 1, 0, 50)])
    if filtered:
        bulk["logFC"] = 9.0
    else:
        bulk = bulk.iloc[:0]

    def unexpected_fit(*args):
        pytest.fail("No statistical model should be fitted without eligible genes")

    monkeypatch.setattr(utils.dist_prob, "fit_lnpois", unexpected_fit)
    assert utils.detect_clonal_loh(bulk) is None


def test_loh_overlap_preserves_chromosome_and_inclusive_end():
    markers = pd.DataFrame({
        "CHROM": ["1", "1", "2"], "POS": [100, 200, 100],
        "snp_id": ["s1", "s2", "s3"],
    })
    loh = pd.DataFrame({
        "CHROM": ["1"], "seg": ["1a"], "seg_start": [100],
        "seg_end": [200], "loh": [True],
    })
    result = utils.annot_consensus(markers, loh, join_mode="left")
    assert result["loh"].fillna(False).tolist() == [True, True, False]


def test_identical_states_on_adjacent_chromosomes_form_distinct_segments():
    bulk = pd.DataFrame({
        "CHROM": ["1", "1", "2", "2"], "POS": [100, 200, 100, 200],
        "snp_index": [0, 1, 0, 1], "gene": ["a", "b", "c", "d"],
        "pAD": [1, 1, 1, 1], "cnv_state": ["loh"] * 4,
    })
    result = utils.annot_segs(bulk)
    assert result["seg"].tolist() == ["1a", "1a", "2a", "2a"]
    assert result.groupby("seg")["CHROM"].nunique().eq(1).all()
