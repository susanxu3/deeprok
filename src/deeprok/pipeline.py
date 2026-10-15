"""The prediction pipeline, in memory.

Same steps and settings as the research pipeline that produced the paper numbers
(``cluster/redecode_keephyp666_stats.sbatch``, job 29683), which ran them as five
scripts exchanging files:

  1. network forward pass over 50-kb phase-aligned tiles, centre-stitched
     -> per-base (L, 11) probabilities                 [save_nn_probs.py]
  2. pass 1: segmental decode, no ranker                [decode_from_nn_probs.py]
  3. genome-specific start ranker trained on the pass-1 calls
     (no reference used)                                [start_ranker.py]
  4. pass 2: segmental decode with the ranker scores    [decode_from_nn_probs.py]
  5. drop calls nested inside a longer antisense call   [filter_nested_antisense.py]

``mode="meta"`` stops after step 2 (the "no ranker" arm), for input too short to
self-train the ranker -- the same split as Prodigal's single / meta modes.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from argparse import Namespace
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

# Settings of the published pipeline. Changing any of these changes the calls.
TILE_WINDOW = 49995            # 50 kb, multiple of the pool size 3
PASS1 = dict(min_nt=90, start_emission_scale=50.0, alpha=0.001, max_overlap_depth=50)
PASS2 = dict(min_nt=90, start_emission_scale=10.0, alpha=0.001, max_overlap_depth=250,
             start_rank_lambda=100.0, start_rank_existence_lambda=50.0)
# The decoder cannot emit a gene longer than this, so it is a hard ceiling, not a
# prior. Across 944 GTDB genomes (2.99 M CDS) the longest CDS is 119 kb and only
# 147 exceed 20 kb, but a genome that has one cannot have it called correctly.
# Raising the cap costs nothing measurable: on the 4 curated genomes 20 kb -> 150 kb
# left every call bit-identical, and on 3 held-out genomes with 12 genes > 20 kb it
# changed only those genes (same call count, same runtime). The published runs used
# 20000; on their genomes the two settings give identical output.
MAX_GENE_LEN = 125000
OVERLAP_TAIL_SCALE = 10.0
RANKER = Namespace(model="cnn", width=24, kernel=7, epochs=40, batch=64, lr=3e-3, wd=1e-3,
                   tau=1.0, patience=6, device="cpu", seed=0, verbose=False)
RANKER_MIN_NT = 30
RANKER_CONF_MIN_MARGIN = 0.8
NESTED_ANTISENSE_CONTAINMENT = 0.6
# Below these the ranker is not trained and the pass-1 calls are reported (as in
# Prodigal, which refuses single mode on < 20 kb).
MIN_SINGLE_MODE_BP = 20000
MIN_RANKER_TRAIN_ORFS = 100

# Network channel layout (annotation_gtf.get_merged_multilabel_labels)
CH_CDS_PLUS, CH_CDS_MINUS = slice(1, 4), slice(4, 7)
CH_START_PLUS, CH_START_MINUS, CH_STOP_PLUS, CH_STOP_MINUS = 7, 8, 9, 10


@dataclass
class Gene:
    seqid: str
    start: int          # 0-based, inclusive (start codon end for '-' genes is `end`)
    end: int            # 0-based, exclusive
    strand: str
    start_codon: str
    stop_codon: str
    conf: float         # mean coding probability over the gene body
    p_start: float
    p_stop: float


@dataclass
class Result:
    genes: List[Gene]
    contigs: Dict[str, str]          # seqid -> sequence (upper case), FASTA order
    mode_used: str
    info: Dict[str, object] = field(default_factory=dict)


def _log_default(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def data_path(name: str):
    return resources.files("deeprok") / "data" / name


def model_card() -> dict:
    return json.loads(data_path("model_card.json").read_text())


def pick_device(requested: str = "auto"):
    import torch
    if requested == "cuda" and not torch.cuda.is_available():
        # Two different problems, two different fixes: a CPU-only build of torch
        # can never see a GPU, while a CUDA build with no visible device is a
        # driver or allocation problem.
        if getattr(torch.version, "cuda", None) is None:
            raise ValueError(
                f"--device cuda: this is a CPU-only build of PyTorch (torch {torch.__version__}), "
                "which cannot use a GPU. Install a CUDA build into this environment "
                "(conda install 'pytorch=*=cuda*'), or use "
                "--device cpu to run without a GPU")
        raise ValueError(
            f"--device cuda: PyTorch {torch.__version__} is built against CUDA "
            f"{torch.version.cuda} but no GPU is visible to it (check nvidia-smi and "
            "CUDA_VISIBLE_DEVICES); use --device cpu to run without a GPU")
    if requested == "mps" and not (getattr(torch.backends, "mps", None)
                                   and torch.backends.mps.is_available()):
        raise ValueError("--device mps: Apple-silicon GPU (MPS) not available; use --device cpu")
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS is never picked automatically: the benchmarks were run on CPU/CUDA. On
    # K-12 ``--device mps`` gave probabilities within 1e-8 of CPU and identical
    # calls (2026-10-12), so it is safe to request.
    return torch.device("cpu")


class InputError(ValueError):
    """The input is not a usable nucleotide FASTA; the message says why."""


# IUPAC nucleotide codes. Anything else in a sequence line is an error rather than
# silently becoming N: digits or spaces (GenBank / EMBL layout) would shift every
# downstream base out of frame, and amino-acid letters mean a protein FASTA.
_IUPAC = frozenset("ACGTNRYSWKMBDHV")
_AA_ONLY = frozenset("EFILPQ")       # amino acids that are not IUPAC nucleotide codes
_COMPRESSED = {b"\x1f\x8b": "gzip", b"BZh": "bzip2", b"\xfd7zXZ": "xz", b"\x28\xb5\x2f\xfd": "zstd"}


def _open_text(path: Path, label: str):
    """Open plain / gzip / bzip2 / xz by content (magic bytes), not by extension."""
    import bz2
    import gzip
    import lzma
    try:
        with open(path, "rb") as fh:
            magic = fh.read(4)
    except OSError as e:
        raise InputError(f"cannot read {label}: {e.strerror}") from None
    kind = next((k for m, k in _COMPRESSED.items() if magic.startswith(m)), "plain")
    if kind == "zstd":
        raise InputError(f"{label} is zstd-compressed; decompress it first (zstd -d)")
    opener = {"gzip": gzip.open, "bzip2": bz2.open, "xz": lzma.open, "plain": open}[kind]
    return opener(path, "rt", encoding="ascii", newline=None), kind


def read_fasta(path: Path, label: Optional[str] = None, log: Callable = _log_default) -> Dict[str, str]:
    """seqid (first header token) -> upper-case sequence, in file order.

    Upper-casing matters: the network was trained on upper-case RefSeq sequence
    (lower case switches on its soft-mask channel) and the decoder only recognises
    upper-case start/stop codons.

    Raises InputError with a one-line, user-facing reason for anything that is not
    a nucleotide FASTA: truncated or corrupt compressed files, FASTQ / GFF /
    GenBank / protein files, non-ASCII bytes, invalid characters, empty or
    duplicate ids, no sequence at all.
    """
    import lzma
    import zlib
    label = label or str(path)
    out: Dict[str, List[str]] = {}
    name, empty = None, []
    fh, kind = _open_text(path, label)
    lineno, last = 0, "\n"
    try:
        with fh:
            for lineno, line in enumerate(fh, 1):
                last = line
                if line.startswith(">"):
                    toks = line[1:].split()
                    if not toks:
                        raise InputError(f"{label} line {lineno}: header with no sequence id")
                    name = toks[0]
                    if name in out:
                        raise InputError(f"{label}: duplicate sequence id {name!r} (line {lineno})")
                    out[name] = []
                    continue
                s = "".join(line.split()).upper()
                if not s:
                    continue
                if name is None:
                    raise InputError(_not_fasta_reason(label, lineno, line))
                if not _IUPAC.issuperset(s):
                    raise InputError(_bad_char_reason(label, lineno, name, s))
                out[name].append(s)
    except UnicodeDecodeError:
        raise InputError(f"{label} line {lineno + 1}: not a text FASTA "
                         "(non-ASCII bytes; binary, corrupt or wrongly compressed file?)") from None
    except EOFError:
        raise InputError(f"{label}: {kind} stream ends early; the file is truncated "
                         "(incomplete download or copy?)") from None
    except (OSError, zlib.error, lzma.LZMAError) as e:   # BadGzipFile, corrupt data, I/O
        raise InputError(f"{label}: cannot read {kind} data ({e}); "
                         "the file is corrupt or truncated") from None

    if kind == "plain" and not last.endswith("\n"):
        # A plain-text FASTA cut short is otherwise indistinguishable from a complete
        # one; a missing final newline is the usual trace of an interrupted copy.
        log(f"deeprok: warning: {label} does not end with a newline; if it was downloaded "
            "or copied, check that it is complete")
    seqs = {}
    for n, parts in out.items():
        if parts:
            seqs[n] = "".join(parts)
        else:
            empty.append(n)
    if empty:
        log(f"deeprok: warning: skipped {len(empty)} record(s) with no sequence: "
            + ", ".join(empty[:5]) + (" ..." if len(empty) > 5 else ""))
    if not seqs:
        raise InputError(f"no sequences found in {label}")
    acgt = sum(len(s) - s.count("N") - sum(s.count(c) for c in "RYSWKMBDHV") for s in seqs.values())
    if acgt == 0:
        log(f"deeprok: warning: {label} contains no A/C/G/T (all N or ambiguity codes); "
            "no genes can be called")
    return seqs


def _not_fasta_reason(label: str, lineno: int, line: str) -> str:
    head = line.strip()[:40]
    hint = ("looks like FASTQ; convert to FASTA first" if head.startswith("@") else
            "looks like GFF" if head.startswith("##gff") or head.count("\t") >= 8 else
            "looks like GenBank; convert to FASTA first" if head.startswith("LOCUS") else
            "sequence before the first '>' header")
    return f"{label} line {lineno}: not a FASTA file ({hint}): {head!r}"


def _bad_char_reason(label: str, lineno: int, name: str, s: str) -> str:
    bad = sorted(set(s) - _IUPAC)
    shown = ", ".join(repr(c) for c in bad[:6])
    if any(c.isdigit() for c in bad):
        hint = "digits in the sequence: GenBank/EMBL-style numbered lines?"
    elif set(bad) & _AA_ONLY:
        hint = "amino-acid letters: this looks like a protein FASTA; deeprok needs nucleotides"
    elif "U" in bad:
        hint = "U: RNA sequence? convert U to T"
    else:
        hint = "only IUPAC nucleotide codes (ACGTN RYSWKMBDHV) are allowed"
    return f"{label} line {lineno}, sequence {name!r}: invalid character(s) {shown} ({hint})"


def predict_probs(contigs: Dict[str, str], *, device, batch_size: int,
                  window: int = TILE_WINDOW, log: Callable = _log_default) -> Dict[str, np.ndarray]:
    """Step 1: per-base (L, 11) sigmoid probabilities for every contig."""
    import torch
    from ._core.genome_sequences import one_hot_table
    from ._core.inference import _load_checkpoint, tile_predict_v4

    if window % 3:
        raise ValueError(f"window must be a multiple of 3, got {window}")
    if device.type == "cpu":
        # oneDNN's LSTM kernel fails with "could not create a primitive" on ~1 genome
        # in 5 (cluster jobs 29659 and earlier); every production run disabled it.
        torch.backends.mkldnn.enabled = False
    with resources.as_file(data_path("model.pt")) as ck:
        model = _load_checkpoint(str(ck), device, pool_size=3, numb_conv=3, numb_lstm=1)
    model.eval()
    # Half-window step rounded UP to a multiple of 3: tile starts must stay
    # phase-aligned with the triplet pooling (49995 -> 24999, as in job 29689).
    step = -(-(window // 2) // 3) * 3
    out = {}
    for name, seq in contigs.items():
        t0 = time.time()
        oh = one_hot_table[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]
        with torch.no_grad():
            p = tile_predict_v4(model, oh, window, step, device, batch_size,
                                full_sequence=False, stitch_mode="center")
        if p.shape != (len(seq), 11):
            raise RuntimeError(f"{name}: probabilities have shape {p.shape}, expected ({len(seq)}, 11)")
        out[name] = p.astype(np.float32)
        log(f"  network  {name}: {len(seq):,} bp in {time.time() - t0:.1f} s")
    return out


def _decode(contigs, probs, stats, overlap_probs, *, min_nt, start_emission_scale, alpha,
            max_overlap_depth, workers, max_gene_len=MAX_GENE_LEN, rank=None, start_rank_lambda=0.0,
            start_rank_existence_lambda=None) -> List[dict]:
    """One segmental decode over all contigs (decode_genome_from_nn_probs, in memory)."""
    from ._core.decode_from_nn_probs import build_length_model
    from ._core.hmm_duration_decode import (annotate_strategy_a_segmental,
                                            default_start_type_bonus, segmental_overlap_range)
    length_model = build_length_model(stats, overlap_probs, min_gene_len=min_nt,
                                      max_gene_len=max_gene_len,
                                      max_overlap_depth=max_overlap_depth,
                                      overlap_tail_scale=OVERLAP_TAIL_SCALE)
    start_bonus = default_start_type_bonus(stats)
    overlaps = segmental_overlap_range(max_overlap_depth)
    genes: List[dict] = []
    for name, seq in contigs.items():
        qp, qm = (rank or {}).get(name, (None, None))
        genes.extend(annotate_strategy_a_segmental(
            probs[name], seq, name, stats, length_model,
            alpha=alpha, use_codon_c=True,
            same_strand_max_ov=max_overlap_depth, opposite_max_ov=max_overlap_depth,
            start_emission_scale=start_emission_scale, min_nt=min_nt,
            allowed_overlaps=overlaps, start_type_bonus=start_bonus,
            overlap_probs=overlap_probs, duration_lambda=0.0,
            gene_length_lambda=0.0, gene_length_center=False, gene_entry_bonus=0.0,
            boundary_emit_floor=None, overlap_emission_scale=1.0,
            conflict_resolution="length",
            rbs_bonus_plus=None, rbs_bonus_minus=None,
            start_rank_q_plus=qp, start_rank_q_minus=qm,
            start_rank_lambda=start_rank_lambda, start_rank_center=True,
            start_rank_existence_lambda=start_rank_existence_lambda,
            segmental_overlap_score="empirical_multihot",
            empirical_connector_prior_lambda=1.0,
            segmental_workers=workers,
        ))
    return genes


def _gff_order(contigs, genes: List[dict]) -> List[tuple]:
    """Pass-1 calls as (seqid, lo, hi, strand) in the order the research ranker read
    them back from the GFF: contigs in FASTA order, genes by (start, end). The order
    feeds the ranker's random candidate subsampling and train/val split."""
    by: Dict[str, List[dict]] = {}
    for g in genes:
        by.setdefault(g["seqid"], []).append(g)
    out = []
    for sid in contigs:
        for g in sorted(by.get(sid, []), key=lambda x: (x["start"], x["end"])):
            out.append((sid, int(g["start"]), int(g["end"]), g["strand"]))
    return out


def train_ranker(contigs, probs, pass1: List[dict], ckpt_out: Path, log=_log_default) -> Optional[int]:
    """Step 3. Returns the number of training ORFs, or None if too few to train."""
    import torch
    from ._core import start_ranker as R
    cs = R.build_candidates(contigs, _gff_order(contigs, pass1), min_nt=RANKER_MIN_NT,
                            probs=probs, rng=np.random.default_rng(RANKER.seed))
    trainable = R.confidence(cs, min_len=0, min_margin=RANKER_CONF_MIN_MARGIN)
    tr = np.flatnonzero(trainable)
    if tr.size < MIN_RANKER_TRAIN_ORFS:
        log(f"  ranker   only {tr.size} confident pass-1 genes (< {MIN_RANKER_TRAIN_ORFS}); not trained")
        return None
    t0 = time.time()
    # Training is the longest silent step in the run (minutes on a whole genome,
    # CPU on every device), so say what is happening before starting it.
    log(f"  ranker   training the start-site ranker on {tr.size} confident pass-1 "
        f"genes (CPU; the slowest step)")
    model, inner, eps = R.train_fold(cs, tr, None, RANKER, RANKER.seed * 100)
    torch.save({"model": RANKER.model, "width": RANKER.width, "kernel": RANKER.kernel,
                "state": model.state_dict(), "up_nt": R.UP_NT, "down_nt": R.DOWN_NT,
                "fold": 0, "inner_top1": inner}, ckpt_out)
    log(f"  ranker   done: {eps} epochs, held-out top-1 {inner:.3f} "
        f"({time.time() - t0:.0f} s)")
    return int(tr.size)


def ranker_scores(contigs, ckpt: Path) -> Dict[str, tuple]:
    from ._core.decode_from_nn_probs import build_start_rank_q
    from ._core.genome_sequences import reverse_complement_dna
    return {name: (build_start_rank_q(seq, ckpt), build_start_rank_q(reverse_complement_dna(seq), ckpt))
            for name, seq in contigs.items()}


def drop_nested_antisense(genes: List[dict]) -> List[dict]:
    """Step 5 (filter_nested_antisense.py on the in-memory calls)."""
    from ._core.filter_nested_antisense import drop_indices
    recs = [{"i": i, "seqid": g["seqid"], "start": int(g["start"]) + 1, "end": int(g["end"]),
             "strand": g["strand"]} for i, g in enumerate(genes)]
    dropped = drop_indices(recs, NESTED_ANTISENSE_CONTAINMENT)
    return [g for i, g in enumerate(genes) if i not in dropped]


def _to_gene(g: dict, P: np.ndarray) -> Gene:
    lo, hi, st = int(g["start"]), int(g["end"]), g["strand"]
    if st == "+":
        ps, pe = float(P[lo:lo + 3, CH_START_PLUS].max()), float(P[hi - 3:hi, CH_STOP_PLUS].max())
        body = P[lo + 3:hi - 3, CH_CDS_PLUS].max(1)
    else:
        ps, pe = float(P[hi - 3:hi, CH_START_MINUS].max()), float(P[lo:lo + 3, CH_STOP_MINUS].max())
        body = P[lo + 3:hi - 3, CH_CDS_MINUS].max(1)
    return Gene(seqid=g["seqid"], start=lo, end=hi, strand=st,
                start_codon=str(g.get("start_codon", "")), stop_codon=str(g.get("stop_codon", "")),
                conf=float(body.mean()) if body.size else float("nan"), p_start=ps, p_stop=pe)


def annotate(fasta: Path, *, mode: str = "single", device: str = "auto", threads: int = 4,
             batch_size: Optional[int] = None, window: int = TILE_WINDOW,
             max_gene_len: int = MAX_GENE_LEN,
             workers: int = 2, keep_dir: Optional[Path] = None,
             log: Callable = _log_default, input_label: Optional[str] = None) -> Result:
    """Predict protein-coding genes in every sequence of ``fasta``."""
    import torch
    from ._core.hmm_prokaryotic import load_overlap_transition_probs

    if mode not in ("single", "meta"):
        raise ValueError(f"mode must be 'single' or 'meta', got {mode!r}")
    torch.set_num_threads(max(1, threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    t_all = time.time()
    contigs = read_fasta(Path(fasta), label=input_label, log=log)
    total = sum(map(len, contigs.values()))
    log(f"deeprok: {len(contigs)} sequence(s), {total:,} bp")
    dev = pick_device(device)
    bs = batch_size or (32 if dev.type == "cuda" else 4)
    log(f"  device   {dev} (batch {bs}, {threads} threads)")

    probs = predict_probs(contigs, device=dev, batch_size=bs, window=window, log=log)
    stats = json.loads(data_path("priors.json").read_text())
    overlap_probs = load_overlap_transition_probs(None)

    t0 = time.time()
    log(f"  pass 1   decoding, no ranker; overlaps up to {PASS1['max_overlap_depth']} nt")
    pass1 = _decode(contigs, probs, stats, overlap_probs, workers=workers,
                    max_gene_len=max_gene_len, **PASS1)
    log(f"  pass 1   {len(pass1)} genes ({time.time() - t0:.0f} s)")
    if keep_dir is not None:
        _keep(keep_dir, contigs, probs, pass1)

    mode_used, n_train = mode, None
    genes = pass1
    if mode == "single" and total < MIN_SINGLE_MODE_BP:
        log(f"  note     input is {total:,} bp (< {MIN_SINGLE_MODE_BP:,}); too short to train the "
            f"start ranker, reporting pass-1 calls (meta mode)")
        mode_used = "meta"
    elif mode == "single":
        with tempfile.TemporaryDirectory() as td:
            ck = (keep_dir / "start_ranker.pt") if keep_dir is not None else Path(td) / "start_ranker.pt"
            n_train = train_ranker(contigs, probs, pass1, ck, log=log)
            if n_train is None:
                mode_used = "meta"
            else:
                rank = ranker_scores(contigs, ck)
                t0 = time.time()
                log(f"  pass 2   decoding with ranker; overlaps up to "
                    f"{PASS2['max_overlap_depth']} nt")
                genes = _decode(contigs, probs, stats, overlap_probs, workers=workers, rank=rank,
                                max_gene_len=max_gene_len, **PASS2)
                n_before = len(genes)
                genes = drop_nested_antisense(genes)
                log(f"  pass 2   {n_before} genes, {n_before - len(genes)} dropped for lying "
                    f"inside a longer gene on the opposite strand -> {len(genes)} "
                    f"({time.time() - t0:.0f} s)")

    order = {sid: i for i, sid in enumerate(contigs)}
    genes = sorted(genes, key=lambda g: (order[g["seqid"]], g["start"], g["end"], g["strand"]))
    out = [_to_gene(g, probs[g["seqid"]]) for g in genes]
    log(f"deeprok: {len(out)} genes in {time.time() - t_all:.0f} s (mode {mode_used})")
    return Result(genes=out, contigs=contigs, mode_used=mode_used,
                  info={"device": str(dev), "ranker_train_orfs": n_train, "total_bp": total})


def _keep(d: Path, contigs, probs, pass1) -> None:
    from ._core.viterbi_decode import write_gff3
    d.mkdir(parents=True, exist_ok=True)
    pd = d / "nn_probs"
    pd.mkdir(exist_ok=True)
    seqs = {}
    for i, (name, p) in enumerate(probs.items()):
        fn = f"contig{i:04d}.npy"
        np.save(pd / fn, p)
        seqs[name] = {"file": fn, "length": int(p.shape[0]), "shape": list(p.shape), "dtype": "float32"}
    (pd / "nn_probs_manifest.json").write_text(json.dumps({"version": 1, "sequences": seqs}, indent=2))
    write_gff3(str(d / "pass1.gff3"), pass1, source="deeprok_pass1")
