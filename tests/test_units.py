"""Unit tests: translation, GFF escaping, bundled model, _core sync."""
from __future__ import annotations

import hashlib
import io
import subprocess
import sys
from pathlib import Path

import pytest

from tprok.pipeline import Gene, data_path, model_card
from tprok.writers import _esc_attr, _esc_seqid, gene_ids, translate, write_gff3

PKG = Path(__file__).resolve().parents[1]


def test_translate_table11():
    assert translate("ATGAAATTTTAA") == "MKF"
    assert translate("GTGAAATAG") == "MK"            # alternative start read as Met
    assert translate("TTGTGGTGA") == "MW"
    assert translate("ATGNNNTAA") == "MX"            # ambiguous codon
    assert translate("ATGTGATAA") == "M*"            # internal stop kept visible


def test_escaping():
    assert _esc_seqid("chr 1;x") == "chr%201%3Bx"
    assert _esc_seqid("NC_000913.3|a") == "NC_000913.3|a"
    assert _esc_attr("a;b=c,d&e%") == "a%3Bb%3Dc%2Cd%26e%25"


def test_gene_ids_count_per_sequence():
    g = lambda s: Gene(s, 0, 9, "+", "ATG", "TAA", 0.9, 0.9, 0.9)
    assert gene_ids([g("a"), g("a"), g("b"), g("a")]) == ["a_1", "a_2", "b_1", "a_3"]


def test_gff_header_and_one_based_coordinates():
    fh = io.StringIO()
    write_gff3(fh, [Gene("c1", 9, 18, "-", "ATG", "TAA", 0.5, 0.6, 0.7)], {"c1": "A" * 30},
               [("mode", "single")])
    lines = fh.getvalue().splitlines()
    assert lines[:3] == ["##gff-version 3", "##sequence-region c1 1 30", "# mode: single"]
    f = lines[3].split("\t")
    assert (f[3], f[4], f[5], f[6]) == ("10", "18", "0.500", "-")


def test_bundled_model_matches_card():
    card = model_card()
    blob = data_path("model.pt").read_bytes()
    assert hashlib.sha256(blob).hexdigest() == card["sha256"]
    assert card["n_parameters"] == 1145313


def test_core_in_sync():
    """_core must equal the research modules it was copied from (only checkable in
    the research checkout, where package/ sits inside the repo)."""
    sync = PKG / "dev" / "sync_core.py"
    if not (PKG.parent / "hmm_duration_decode.py").exists():
        pytest.skip("not inside the research repository")
    r = subprocess.run([sys.executable, str(sync), "--check"], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
