import pandas as pd
import pytest

from spacenumbat import hmm
from spacenumbat import utils


def test_generate_postfix_matches_r_style_sequence():
    assert utils.generate_postfix([1, 2, 26, 27, 28, 52, 53]) == [
        "a",
        "b",
        "z",
        "aa",
        "ab",
        "az",
        "ba",
    ]


def test_annot_segs_sorts_by_chromosome_and_snp_index_before_segmenting():
    # The rows deliberately interleave chromosomes and positions. The R helper
    # arranges by CHROM/snp_index before boundary detection, so chr1's two neu
    # rows should form one segment with two unique genes, not two one-gene
    # segments split by the chr2 row.
    bulk = pd.DataFrame(
        {
            "CHROM": ["1", "2", "1", "1"],
            "POS": [100, 50, 200, 300],
            "snp_index": [1, 1, 2, 3],
            "gene": ["A", "C", "B", "D"],
            "pAD": [1, 1, 1, 1],
            "cnv_state": ["neu", "amp", "neu", "del"],
        }
    )

    out = utils.annot_segs(bulk, var="cnv_state")

    assert out["CHROM"].tolist() == ["1", "1", "1", "2"]
    assert out["POS"].tolist() == [100, 200, 300, 50]
    assert out["seg"].tolist() == ["1a", "1a", "1b", "2a"]
    assert out.loc[out["seg"] == "1a", "n_genes"].unique().tolist() == [2]


def test_smooth_segs_fills_only_within_each_chromosome():
    bulk = pd.DataFrame(
        {
            "CHROM": ["1", "1", "2", "2"],
            "seg": ["1a", "1b", "2a", "2b"],
            "n_genes": [11, 1, 1, 1],
            "cnv_state": ["neu", "amp", "del", "amp"],
        }
    )

    with pytest.raises(ValueError, match="CHROM 2"):
        hmm.smooth_segs(bulk, min_genes=10)
