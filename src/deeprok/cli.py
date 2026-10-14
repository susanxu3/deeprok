"""Command-line interface: ``deeprok -i genome.fna -o genes.gff3 -a proteins.faa``."""
from __future__ import annotations

import argparse
import multiprocessing
import os
import shlex
import shutil
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

from . import __version__
from .pipeline import MAX_GENE_LEN, TILE_WINDOW

EPILOG = """\
examples:
  deeprok -i genome.fna -o genes.gff3 -a proteins.faa
  deeprok -i genome.fna.gz -o genes.gff3 -d genes.fna --threads 8
  deeprok -i small_contigs.fna -o genes.gff3 -p meta
  cat genome.fna | deeprok > genes.gff3

modes:
  single  (default) two passes: the second uses a start-site ranker trained on
          this genome's own first-pass calls (needs >= 20 kb of sequence)
  meta    first pass only; for short sequences, plasmids, or mixed bins

Outputs are complete CDS only (start codon to stop codon, translation table 11).
"""


def _positive_int(text: str) -> int:
    try:
        v = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if v < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {v}")
    return v


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="deeprok",
        description="Neural prokaryotic gene finder: CNN+BiLSTM base-level predictions "
                    "decoded by a segmental ORF HMM.",
        epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    io = ap.add_argument_group("input / output")
    io.add_argument("-i", "--input", metavar="FASTA", default="-",
                    help="genome FASTA (plain, .gz or .bz2); '-' or omitted = stdin")
    io.add_argument("-o", "--output", metavar="GFF3", default="-",
                    help="gene predictions, GFF3 (default: stdout)")
    io.add_argument("-a", "--proteins", metavar="FAA", help="write protein translations (FASTA)")
    io.add_argument("-d", "--nucleotides", metavar="FNA", help="write CDS nucleotide sequences (FASTA)")
    io.add_argument("--keep-intermediates", metavar="DIR", type=Path,
                    help="also save network probabilities (.npy), pass-1 calls and the ranker model")
    run = ap.add_argument_group("run")
    run.add_argument("-p", "--mode", choices=["single", "meta"], default="single",
                     help="single (default) or meta, see below")
    run.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"],
                     help="where to run the network (default: cuda if available, else cpu)")
    run.add_argument("--threads", type=_positive_int, default=4, metavar="N",
                     help="CPU threads (default 4)")
    run.add_argument("--batch-size", type=_positive_int, metavar="N",
                     help="tiles per network batch (default 32 on GPU, 4 on CPU)")
    run.add_argument("--max-gene-len", type=int, default=MAX_GENE_LEN, metavar="NT",
                     help="longest gene that can be called, in nt (default %(default)s)")
    run.add_argument("--window", type=int, default=TILE_WINDOW, metavar="NT",
                     help="network tile length in nt, multiple of 3 (default 49995)")
    ap.add_argument("-q", "--quiet", action="store_true", help="no progress messages on stderr")
    ap.add_argument("-v", "--version", action="version",
                    version=f"deeprok {__version__}")
    return ap


@contextmanager
def _stdout_to_stderr(silence: bool = False):
    """Point file descriptor 1 at stderr (or /dev/null) while the pipeline runs.

    The decoder code prints diagnostics with print(), including from its worker
    processes, which inherit fd 1; redirecting sys.stdout alone would not catch
    those, and any of it on stdout would corrupt a GFF written to stdout.

    fd 1 is a pipe or a file here, not a tty, so sys.stdout would be block
    buffered and the decoder's lines would reach the terminal long after the
    progress lines they belong under (a pass-1 diagnostic printing below the
    "pass 2" heading). Line buffering while the redirect is in place keeps the
    two streams in order.
    """
    sys.stdout.flush()
    saved = os.dup(1)
    target = os.open(os.devnull, os.O_WRONLY) if silence else os.dup(2)
    os.dup2(target, 1)
    os.close(target)
    was_line_buffered = sys.stdout.line_buffering
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except (AttributeError, ValueError):       # not a TextIOWrapper
        was_line_buffered = None
    try:
        yield
    finally:
        sys.stdout.flush()
        if was_line_buffered is not None:
            sys.stdout.reconfigure(line_buffering=was_line_buffered)
        os.dup2(saved, 1)
        os.close(saved)


def _check_input(path: Path) -> None:
    if not path.exists():
        sys.exit(f"deeprok: input not found: {path}")
    if path.is_dir():
        sys.exit(f"deeprok: input is a directory, expected a FASTA file: {path}")
    if not os.access(path, os.R_OK):
        sys.exit(f"deeprok: input is not readable (permission denied): {path}")


def _check_outputs(args) -> None:
    """Fail before the (minutes-long) run, not after it, on unwritable outputs."""
    files = [(flag, p) for flag, p in (("-o", args.output), ("-a", args.proteins),
                                       ("-d", args.nucleotides)) if p and p != "-"]
    seen = {}
    inp = Path(args.input).resolve() if args.input != "-" else None
    for flag, p in files:
        p = Path(p)
        if p.is_dir():
            sys.exit(f"deeprok: {flag} {p} is a directory")
        if _is_special(p):            # /dev/null, a named pipe, /dev/stdout ...
            if not os.access(p, os.W_OK):
                sys.exit(f"deeprok: {flag} {p}: permission denied")
            continue
        parent = p.parent if str(p.parent) else Path(".")
        if not parent.is_dir():
            sys.exit(f"deeprok: {flag} {p}: directory {parent} does not exist")
        if not os.access(parent, os.W_OK) or (p.exists() and not os.access(p, os.W_OK)):
            sys.exit(f"deeprok: {flag} {p}: permission denied")
        r = p.resolve()
        if r == inp:
            sys.exit(f"deeprok: {flag} {p} would overwrite the input")
        if r in seen:
            sys.exit(f"deeprok: {seen[r]} and {flag} both write to {p}")
        seen[r] = flag
    k = args.keep_intermediates
    if k is not None:
        if k.exists() and not k.is_dir():
            sys.exit(f"deeprok: --keep-intermediates {k} exists and is not a directory")
        if not k.exists() and not (k.parent.is_dir() and os.access(k.parent, os.W_OK)):
            sys.exit(f"deeprok: --keep-intermediates {k}: cannot create it "
                     f"(parent {k.parent} missing or not writable)")


def _is_special(p: Path) -> bool:
    """Exists and is not a regular file (device, FIFO): write in place, never rename onto it."""
    import stat
    try:
        return not stat.S_ISREG(os.stat(p).st_mode)
    except OSError:
        return False


@contextmanager
def _atomic_write(path: str):
    """Write to a temporary file next to ``path`` and rename on success, so an
    interrupted or failed run never leaves a truncated output behind."""
    p = Path(path)
    if _is_special(p):
        with open(p, "w") as fh:
            yield fh
        return
    fd, tmp = tempfile.mkstemp(prefix=f".{p.name}.", suffix=".tmp", dir=p.parent if str(p.parent) else ".")
    try:
        with os.fdopen(fd, "w") as fh:
            yield fh
        os.replace(tmp, p)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def main(argv=None) -> int:
    try:
        return _main(argv)
    except KeyboardInterrupt:
        print("deeprok: interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        # reader of stdout went away (e.g. `deeprok ... | head`); not an error
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())   # silence the flush at interpreter exit
        return 0
    except MemoryError:
        sys.exit("deeprok: out of memory (a 9-Mb genome needs ~5 GB; try --batch-size 1)")
    except OSError as e:
        where = f": {e.filename}" if getattr(e, "filename", None) else ""
        sys.exit(f"deeprok: I/O error{where}: {e.strerror or e}")


def _main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    # The decoder runs the two strands in worker processes. After torch has started
    # its OpenMP threads, fork() can deadlock those workers on Linux (cluster job
    # 25984); spawn avoids it and is already the macOS default.
    multiprocessing.set_start_method("spawn", force=True)

    from .pipeline import annotate, model_card
    from .writers import write_faa, write_fna, write_gff3

    log = (lambda msg: None) if args.quiet else (lambda msg: print(msg, file=sys.stderr, flush=True))
    if args.max_gene_len < 300:
        sys.exit("deeprok: --max-gene-len must be at least 300")
    if args.window % 3 or args.window < 9990:
        sys.exit("deeprok: --window must be a multiple of 3 and at least 9990")
    if args.input != "-":
        _check_input(Path(args.input))
    _check_outputs(args)

    with ExitStack() as stack:
        if args.input == "-":
            if sys.stdin.isatty():
                build_parser().print_usage(sys.stderr)
                sys.exit("deeprok: no input (use -i FASTA or pipe a FASTA on stdin)")
            td = stack.enter_context(tempfile.TemporaryDirectory())
            fasta = Path(td) / "stdin.fna"
            with fasta.open("wb") as fh:
                shutil.copyfileobj(sys.stdin.buffer, fh)
            label = "stdin"
        else:
            fasta, label = Path(args.input), args.input

        try:
            with _stdout_to_stderr(silence=args.quiet):
                res = annotate(fasta, mode=args.mode, device=args.device, threads=args.threads,
                               batch_size=args.batch_size, window=args.window,
                               max_gene_len=args.max_gene_len,
                               keep_dir=args.keep_intermediates, log=log, input_label=label)
        except ValueError as e:
            sys.exit(f"deeprok: {e}")

        card = model_card()
        header = [
            ("deeprok version", __version__),
            ("model", f"{card['name']} (sha256 {card['sha256'][:12]})"),
            ("mode", res.mode_used + ("" if res.mode_used == args.mode
                                      else f" (requested {args.mode}; see stderr)")),
            ("translation table", "11"),
            ("max gene length", f"{args.max_gene_len} nt"),
            ("network window", f"{args.window} nt"),
            ("command", "deeprok " + " ".join(shlex.quote(a) for a in (argv if argv is not None
                                                                      else sys.argv[1:]))),
        ]
        if args.proteins:
            with _atomic_write(args.proteins) as fh:
                write_faa(fh, res.genes, res.contigs)
        if args.nucleotides:
            with _atomic_write(args.nucleotides) as fh:
                write_fna(fh, res.genes, res.contigs)
        if args.output == "-":
            write_gff3(sys.stdout, res.genes, res.contigs, header)
            sys.stdout.flush()
        else:
            with _atomic_write(args.output) as fh:
                write_gff3(fh, res.genes, res.contigs, header)
    return 0


if __name__ == "__main__":
    sys.exit(main())
