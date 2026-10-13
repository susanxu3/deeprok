"""Error handling: malformed / truncated input, unwritable output, bad options.

Every case must end in a one-line ``deeprok: ...`` message (no traceback) and,
where possible, before the network runs. Most cases fail during input checks,
so this module is fast; the few that run the pipeline use ``-p meta`` on 30 kb.
"""
from __future__ import annotations

import bz2
import gzip
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from deeprok.cli import main
from deeprok.pipeline import InputError, read_fasta

DATA = Path(__file__).parent / "data"
FNA = DATA / "ecoli_k12_200kb.fna.gz"


@pytest.fixture(scope="module")
def seq() -> str:
    text = gzip.decompress(FNA.read_bytes()).decode()
    return "".join(text.split("\n", 1)[1].split())[:30000]


@pytest.fixture
def good(tmp_path, seq) -> Path:
    p = tmp_path / "good.fna"
    p.write_text(">ok\n" + "\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)) + "\n")
    return p


def _fails(argv, match):
    with pytest.raises(SystemExit) as e:
        main(["-q", "-p", "meta", *map(str, argv)])
    assert e.value.code not in (0, None)
    assert match in str(e.value.code), e.value.code


# ---- input ---------------------------------------------------------------

@pytest.mark.parametrize("compress,kind", [(gzip.compress, "gzip"), (bz2.compress, "bzip2")])
def test_truncated_compressed_input(tmp_path, good, compress, kind):
    blob = compress(good.read_bytes())
    p = tmp_path / "t.fna.gz"
    p.write_bytes(blob[: len(blob) // 2])
    _fails(["-i", p], f"{kind} stream ends early; the file is truncated")


def test_corrupt_gzip(tmp_path, good):
    blob = bytearray(gzip.compress(good.read_bytes()))
    blob[len(blob) // 2: len(blob) // 2 + 64] = os.urandom(64)
    p = tmp_path / "c.fna.gz"
    p.write_bytes(bytes(blob))
    _fails(["-i", p], "corrupt or truncated")


def test_compression_detected_by_content_not_extension(tmp_path, good):
    gz_no_ext = tmp_path / "plain_name.fna"
    gz_no_ext.write_bytes(gzip.compress(good.read_bytes()))
    plain_gz_ext = tmp_path / "not_really.fna.gz"
    plain_gz_ext.write_bytes(good.read_bytes())
    assert read_fasta(gz_no_ext) == read_fasta(plain_gz_ext) == read_fasta(good)


def test_truncated_plain_fasta_warns(tmp_path, good):
    p = tmp_path / "cut.fna"
    p.write_text(good.read_text()[:5000])           # cut mid-line, no final newline
    msgs = []
    read_fasta(p, log=msgs.append)
    assert any("does not end with a newline" in m for m in msgs)


@pytest.mark.parametrize("content,match", [
    ("", "no sequences found"),
    ("\n \n\n", "no sequences found"),
    (">a\n>b\n", "no sequences found"),
    ("ACGTACGT\n", "sequence before the first '>' header"),
    ("@r1\nACGT\n+\nIIII\n", "looks like FASTQ"),
    ("##gff-version 3\n", "looks like GFF"),
    ("LOCUS       NC_000913\n", "looks like GenBank"),
    (">p\nMKLVAEERSTWQHHPLDMNNK\n", "looks like a protein FASTA"),
    (">a\n        1 agcttttcat tctgactgca\n", "digits in the sequence"),
    (">a\nACGUACGU\n", "RNA sequence"),
    (">a\nACGT-ACGT\n", "invalid character(s) '-'"),
    (">\nACGT\n", "header with no sequence id"),
    (">a\nACGT\n>a\nACGT\n", "duplicate sequence id 'a'"),
])
def test_malformed_fasta(tmp_path, content, match):
    p = tmp_path / "bad.fna"
    p.write_text(content)
    _fails(["-i", p], match)


def test_non_ascii_and_binary(tmp_path):
    u = tmp_path / "u.fna"
    u.write_text(">a\nACGTÄACGT\n", encoding="utf-8")
    _fails(["-i", u], "non-ASCII bytes")
    b = tmp_path / "b.fna"
    b.write_bytes(b">a\n" + bytes(range(128, 256)))
    _fails(["-i", b], "non-ASCII bytes")


def test_zstd_is_named(tmp_path):
    p = tmp_path / "z.fna.zst"
    p.write_bytes(b"\x28\xb5\x2f\xfd" + b"\0" * 20)
    _fails(["-i", p], "zstd-compressed")


def test_input_path_problems(tmp_path):
    _fails(["-i", tmp_path / "missing.fna"], "input not found")
    _fails(["-i", tmp_path], "input is a directory")


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permissions")
def test_unreadable_input(tmp_path, good):
    good.chmod(0)
    try:
        _fails(["-i", good], "permission denied")
    finally:
        good.chmod(0o644)


def test_lenient_cases_still_run(tmp_path, seq):
    """CRLF line ends, spaces inside lines, IUPAC codes and empty records are fine."""
    p = tmp_path / "lenient.fna"
    body = "\r\n".join(seq[i:i + 60] for i in range(0, len(seq), 60)).replace("A", "R", 20)
    p.write_bytes(f">empty\r\n>x desc\r\n{body[:300]} {body[300:]}\r\n".encode())
    msgs = []
    out = read_fasta(p, log=msgs.append)
    assert list(out) == ["x"] and len(out["x"]) == len(seq)
    assert any("skipped 1 record(s) with no sequence: empty" in m for m in msgs)


def test_all_n_warns_and_reports_no_genes(tmp_path):
    p = tmp_path / "n.fna"
    p.write_text(">n\n" + "N" * 30000 + "\n")
    msgs = []
    read_fasta(p, log=msgs.append)
    assert any("no A/C/G/T" in m for m in msgs)


# ---- output and options -------------------------------------------------

def test_outputs_checked_before_the_run(tmp_path, good):
    _fails(["-i", good, "-o", tmp_path / "nodir" / "x.gff3"], "does not exist")
    _fails(["-i", good, "-a", tmp_path / "nodir" / "x.faa"], "does not exist")
    _fails(["-i", good, "-o", tmp_path], "is a directory")
    _fails(["-i", good, "-o", good], "would overwrite the input")
    _fails(["-i", good, "-o", tmp_path / "s", "-a", tmp_path / "s"], "both write to")
    f = tmp_path / "afile"
    f.write_text("x")
    _fails(["-i", good, "--keep-intermediates", f], "exists and is not a directory")


def test_failed_run_leaves_existing_output_untouched(tmp_path, good):
    out = tmp_path / "o.gff3"
    out.write_text("OLD\n")
    bad = tmp_path / "t.fna.gz"
    bad.write_bytes(gzip.compress(good.read_bytes())[:2000])
    _fails(["-i", bad, "-o", out], "truncated")
    assert out.read_text() == "OLD\n"
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_special_output_files(good):
    assert main(["-q", "-p", "meta", "-i", str(good), "-o", os.devnull, "-a", os.devnull]) == 0


@pytest.mark.parametrize("argv,match", [
    (["--threads", "0"], "must be at least 1"),
    (["--batch-size", "x"], "expected a positive integer"),
    (["--window", "10000"], ""),
    (["--max-gene-len", "200"], "at least 300"),
    (["-p", "fast"], ""),
])
def test_bad_options(good, argv, match, capsys):
    with pytest.raises(SystemExit) as e:
        main(["-q", "-i", str(good), *argv])
    assert e.value.code not in (0, None)
    assert match in capsys.readouterr().err + str(e.value.code)


def test_unavailable_device(good):
    import torch
    if torch.cuda.is_available():
        pytest.skip("CUDA present")
    _fails(["-i", good, "--device", "cuda"], "no CUDA GPU available")


# ---- process-level behaviour ---------------------------------------------

def _cli(*argv, **kw):
    return subprocess.run([sys.executable, "-m", "deeprok", *map(str, argv)], capture_output=True, **kw)


def test_no_traceback_on_stdin_errors(good):
    r = _cli("-p", "meta", input=gzip.compress(good.read_bytes())[:2000])
    assert r.returncode == 1 and b"Traceback" not in r.stderr
    assert r.stderr.strip().endswith(b"stdin: gzip stream ends early; the file is truncated "
                                     b"(incomplete download or copy?)")
    r = _cli("-p", "meta", input=b"")
    assert r.returncode == 1 and b"no sequences found in stdin" in r.stderr


def test_gzip_on_stdin(good):
    r = _cli("-q", "-p", "meta", input=gzip.compress(good.read_bytes()))
    assert r.returncode == 0 and r.stdout.startswith(b"##gff-version 3")


def test_broken_pipe_is_quiet(good):
    p = subprocess.Popen([sys.executable, "-m", "deeprok", "-q", "-p", "meta", "-i", str(good)],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.stdout.readline()
    p.stdout.close()                         # like `deeprok ... | head -1`
    err = p.stderr.read()
    assert p.wait(timeout=120) == 0 and b"Traceback" not in err and b"Exception" not in err


def test_interrupt_exits_130_and_keeps_old_output(tmp_path):
    out = tmp_path / "o.gff3"
    out.write_text("OLD\n")
    p = subprocess.Popen([sys.executable, "-m", "deeprok", "-i", str(FNA), "-o", str(out)],
                         stderr=subprocess.PIPE)
    # wait until the network is running, then Ctrl-C
    deadline = time.time() + 60
    buf = b""
    while b"nn batch" not in buf and time.time() < deadline:
        buf += p.stderr.read1(4096)
    p.send_signal(signal.SIGINT)
    err = buf + p.stderr.read()
    assert p.wait(timeout=60) == 130
    assert err.strip().endswith(b"deeprok: interrupted") and b"Traceback" not in err
    assert out.read_text() == "OLD\n"
    assert not [x for x in tmp_path.iterdir() if x.name.endswith(".tmp")]


def test_input_error_is_a_value_error():
    # library users catch ValueError, as annotate() documented before InputError existed
    assert issubclass(InputError, ValueError)
