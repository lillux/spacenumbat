import pandas as pd
import pytest

from spacenumbat import hmm
from spacenumbat import utils


def test_generate_postfix_uses_python_zero_based_indices():
    assert utils.generate_postfix([0, 1, 25, 26, 27, 51, 52]) == [
        "a",
        "b",
        "z",
        "aa",
        "ab",
        "az",
        "ba",
    ]


def test_generate_postfix_rejects_negative_indices():
    with pytest.raises(ValueError, match="non-negative"):
        utils.generate_postfix([-1])


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

    with pytest.raises(ValueError, match="CHROM 2.*max_segment_genes=1"):
        hmm.smooth_segs(bulk, min_genes=10)


def test_fill_neu_segs_uses_inclusive_segment_ends_for_pyranges():
    segs_consensus = pd.DataFrame(
        {
            "CHROM": ["1"],
            "seg_start": [3],
            "seg_end": [5],
            "cnv_state": ["amp"],
        }
    )
    segs_neu = pd.DataFrame(
        {
            "CHROM": ["1"],
            "seg_start": [1],
            "seg_end": [7],
            "seg_length": [7],
        }
    )

    out = utils.fill_neu_segs(segs_consensus, segs_neu)

    assert out[["seg_start", "seg_end", "cnv_state", "seg_cons"]].to_dict("records") == [
        {"seg_start": 1, "seg_end": 2, "cnv_state": "neu", "seg_cons": "1a"},
        {"seg_start": 3, "seg_end": 5, "cnv_state": "amp", "seg_cons": "1b"},
        {"seg_start": 6, "seg_end": 7, "cnv_state": "neu", "seg_cons": "1c"},
    ]
    assert out["seg_length"].tolist() == [2, 3, 2]


def test_smooth_segs_error_reports_gene_and_segment_diagnostics():
    bulk = pd.DataFrame(
        {
            "CHROM": ["1", "1", "1"],
            "seg": ["1a", "1b", "1c"],
            "n_genes": [3, 7, 2],
            "gene": ["A", "B", "C"],
            "cnv_state": ["neu", "amp", "del"],
        }
    )

    with pytest.raises(ValueError) as excinfo:
        hmm.smooth_segs(bulk, min_genes=10)

    message = str(excinfo.value)
    assert "unique_genes=3" in message
    assert "segments=3" in message
    assert "max_segment_genes=7" in message
    assert "top_segments=1b:7,1a:3,1c:2" in message
