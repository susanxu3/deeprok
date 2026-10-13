"""End-to-end tests of the command line on the bundled 200-kb E. coli slice."""
from __future__ import annotations

import gzip
import subprocess
import sys
from pathlib import Path

import pytest

from tprok.cli import main
from tprok.writers import CODON_TABLE_11, revcomp

DATA = Path(__file__).parent / "data"
FNA = DATA / "ecoli_k12_200kb.fna.gz"


def _features(gff: Path) -> list[list[str]]:
    return [l.rstrip("\n").split("\t") for l in gff.read_text().splitlines() if not l.startswith("#")]


def _calls(gff: Path) -> set[tuple]:
    return {(f[0], f[3], f[4], f[6]) for f in _features(gff)}


def _fasta(path: Path) -> dict[str, str]:
    out, name = {}, None
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                name = line[1:].split()[0]
                out[name] = []
            else:
                out[name].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def single(tmp_path_factory):
    d = tmp_path_factory.mktemp("single")
    assert main(["-q", "-i", str(FNA), "-o", str(d / "o.gff3"), "-a", str(d / "o.faa"),
                 "-d", str(d / "o.fna"), "--keep-intermediates", str(d / "keep")]) == 0
    return d


def _identity(a: set, b: set) -> float:
    return len(a & b) / max(len(a | b), 1)


def test_single_mode_matches_expected(single):
    # >= 99 %: the network runs in float32 and BLAS differs across platforms, which
    # can move a borderline start; the reference output was made on macOS arm64.
    got, exp = _calls(single / "o.gff3"), _calls(DATA / "ecoli_k12_200kb.expected.gff3")
    assert _identity(got, exp) >= 0.99, (len(got), len(exp), len(got & exp))


def test_meta_mode_matches_expected(tmp_path):
    out = tmp_path / "m.gff3"
    assert main(["-q", "-p", "meta", "-i", str(FNA), "-o", str(out)]) == 0
    assert _identity(_calls(out), _calls(DATA / "ecoli_k12_200kb.meta.expected.gff3")) >= 0.99
    assert "# mode: meta" in out.read_text()


def test_gff3_is_well_formed(single):
    text = (single / "o.gff3").read_text().splitlines()
    assert text[0] == "##gff-version 3"
    assert text[1] == "##sequence-region NC_000913.3_1-200000 1 200000"
    feats = _features(single / "o.gff3")
    assert len(feats) > 100
    ids = set()
    for f in feats:
        assert len(f) == 9
        seqid, src, typ, s, e, score, strand, phase, attrs = f
        assert (src, typ, phase) == ("tprok", "CDS", "0")
        assert strand in "+-"
        assert 1 <= int(s) < int(e) <= 200000
        assert (int(e) - int(s) + 1) % 3 == 0
        assert 0.0 <= float(score) <= 1.0
        kv = dict(x.split("=", 1) for x in attrs.split(";"))
        assert set(kv) == {"ID", "start_codon", "stop_codon", "conf", "p_start", "p_stop"}
        assert kv["ID"] not in ids
        ids.add(kv["ID"])


def test_coordinates_codons_and_proteins_agree(single):
    genome = _fasta(FNA)["NC_000913.3_1-200000"]
    prots = _fasta(single / "o.faa")
    nts = _fasta(single / "o.fna")
    feats = _features(single / "o.gff3")
    assert len(prots) == len(nts) == len(feats)
    for f in feats:
        s, e, strand = int(f[3]), int(f[4]), f[6]
        kv = dict(x.split("=", 1) for x in f[8].split(";"))
        cds = genome[s - 1:e] if strand == "+" else revcomp(genome[s - 1:e])
        assert nts[kv["ID"]] == cds
        assert cds[:3] == kv["start_codon"] and cds[-3:] == kv["stop_codon"]
        assert CODON_TABLE_11[cds[-3:]] == "*"
        prot = prots[kv["ID"]]
        assert len(prot) * 3 + 3 == len(cds)
        assert prot[0] == "M" and "*" not in prot


def test_known_gene_thrA(single):
    # thrA (b0002), NC_000913.3:337..2799 +, a long, unambiguous gene
    assert ("NC_000913.3_1-200000", "337", "2799", "+") in _calls(single / "o.gff3")


def test_intermediates_written(single):
    k = single / "keep"
    assert (k / "pass1.gff3").exists() and (k / "start_ranker.pt").exists()
    assert (k / "nn_probs" / "nn_probs_manifest.json").exists()


def test_stdin_to_stdout_and_clean_stdout():
    raw = gzip.decompress(FNA.read_bytes())
    r = subprocess.run([sys.executable, "-m", "tprok", "-p", "meta"], input=raw,
                       capture_output=True, check=True)
    out = r.stdout.decode().splitlines()
    assert out[0] == "##gff-version 3"
    # every stdout line is GFF: decoder diagnostics must go to stderr
    assert all(l.startswith("#") or l.count("\t") == 8 for l in out)
    assert b"pass 1" in r.stderr


def test_quiet_is_silent():
    r = subprocess.run([sys.executable, "-m", "tprok", "-q", "-p", "meta", "-i", str(FNA)],
                       capture_output=True, check=True)
    assert r.stderr == b""


def test_short_input_falls_back_to_meta(tmp_path):
    seq = _fasta(FNA)["NC_000913.3_1-200000"][:15000]
    fa = tmp_path / "short.fna"
    fa.write_text(">short\n" + seq + "\n")
    out = tmp_path / "s.gff3"
    assert main(["-q", "-i", str(fa), "-o", str(out)]) == 0
    assert "# mode: meta (requested single" in out.read_text()


def test_lowercase_input_is_handled(tmp_path):
    seq = _fasta(FNA)["NC_000913.3_1-200000"][:30000]
    up, lo = tmp_path / "up.fna", tmp_path / "lo.fna"
    up.write_text(">x\n" + seq + "\n")
    lo.write_text(">x\n" + seq.lower() + "\n")
    a, b = tmp_path / "a.gff3", tmp_path / "b.gff3"
    main(["-q", "-p", "meta", "-i", str(up), "-o", str(a)])
    main(["-q", "-p", "meta", "-i", str(lo), "-o", str(b)])
    assert _calls(a) == _calls(b) and len(_calls(a)) > 10


def test_bad_input_exits_cleanly(tmp_path):
    with pytest.raises(SystemExit, match="input not found"):
        main(["-i", str(tmp_path / "nope.fna")])
    empty = tmp_path / "empty.fna"
    empty.write_text("")
    with pytest.raises(SystemExit, match="no sequences"):
        main(["-q", "-i", str(empty)])
    dup = tmp_path / "dup.fna"
    dup.write_text(">a\nACGT\n>a\nACGT\n")
    with pytest.raises(SystemExit, match="duplicate"):
        main(["-q", "-i", str(dup)])
