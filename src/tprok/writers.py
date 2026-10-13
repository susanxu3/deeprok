"""Output formats: GFF3, protein FASTA, nucleotide FASTA."""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, TextIO, Tuple

from .pipeline import Gene

SOURCE = "tprok"

# NCBI translation table 11 (bacteria, archaea, plant plastids), TCAG order.
_B = "TCAG"
_AA11 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE_11 = {a + b + c: _AA11[16 * i + 4 * j + k]
                  for i, a in enumerate(_B) for j, b in enumerate(_B) for k, c in enumerate(_B)}
_COMP = str.maketrans("ACGTN", "TGCAN")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def cds_sequence(g: Gene, contig: str) -> str:
    """Nucleotide sequence of the CDS, 5'->3' on its own strand, stop codon included."""
    s = contig[g.start:g.end]
    return s if g.strand == "+" else revcomp(s)


def translate(nt: str) -> str:
    """Protein for a complete CDS. The start codon is read as Met whatever its
    sequence (GTG/TTG initiate with fMet), the terminal stop is not written, and a
    codon containing a non-ACGT base becomes X."""
    aa = [CODON_TABLE_11.get(nt[i:i + 3], "X") for i in range(0, len(nt) - len(nt) % 3, 3)]
    if aa:
        aa[0] = "M"
    if aa and aa[-1] == "*":
        aa.pop()
    return "".join(aa)


# GFF3 spec: column 1 may use [a-zA-Z0-9.:^*$@!+_?-|], everything else %-escaped;
# attribute values must escape ; = & , and control characters.
_SEQID_OK = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.:^*$@!+_?-|")


def _esc_seqid(s: str) -> str:
    return "".join(c if c in _SEQID_OK else f"%{ord(c):02X}" for c in s)


def _esc_attr(s: str) -> str:
    return "".join(f"%{ord(c):02X}" if c in ";=&,%\t\n\r" or ord(c) < 32 else c for c in s)


def gene_ids(genes: Iterable[Gene]) -> List[str]:
    """``<seqid>_<n>``, n counting from 1 within each sequence (in output order)."""
    n: Dict[str, int] = {}
    out = []
    for g in genes:
        n[g.seqid] = n.get(g.seqid, 0) + 1
        out.append(f"{g.seqid}_{n[g.seqid]}")
    return out


def _fmt(x: float) -> str:
    return "nan" if math.isnan(x) else f"{x:.3f}"


def write_gff3(fh: TextIO, genes: List[Gene], contigs: Dict[str, str],
               header: List[Tuple[str, str]]) -> None:
    fh.write("##gff-version 3\n")
    for sid, seq in contigs.items():
        fh.write(f"##sequence-region {_esc_seqid(sid)} 1 {len(seq)}\n")
    for k, v in header:
        fh.write(f"# {k}: {v}\n")
    for gid, g in zip(gene_ids(genes), genes):
        attrs = ";".join([
            f"ID={_esc_attr(gid)}",
            f"start_codon={g.start_codon}",
            f"stop_codon={g.stop_codon}",
            f"conf={_fmt(g.conf)}",
            f"p_start={_fmt(g.p_start)}",
            f"p_stop={_fmt(g.p_stop)}",
        ])
        fh.write("\t".join([_esc_seqid(g.seqid), SOURCE, "CDS", str(g.start + 1), str(g.end),
                            _fmt(g.conf), g.strand, "0", attrs]) + "\n")


def _header_line(gid: str, g: Gene) -> str:
    # Prodigal-style: ">ID # start # end # strand # key=value;..." (1-based, inclusive)
    return (f">{gid} # {g.start + 1} # {g.end} # {1 if g.strand == '+' else -1} # "
            f"seqid={g.seqid};start_codon={g.start_codon};conf={_fmt(g.conf)}")


def _wrap(s: str, width: int = 60) -> str:
    return "\n".join(s[i:i + width] for i in range(0, len(s), width)) or ""


def write_faa(fh: TextIO, genes: List[Gene], contigs: Dict[str, str]) -> None:
    for gid, g in zip(gene_ids(genes), genes):
        fh.write(_header_line(gid, g) + "\n" + _wrap(translate(cds_sequence(g, contigs[g.seqid]))) + "\n")


def write_fna(fh: TextIO, genes: List[Gene], contigs: Dict[str, str]) -> None:
    for gid, g in zip(gene_ids(genes), genes):
        fh.write(_header_line(gid, g) + "\n" + _wrap(cds_sequence(g, contigs[g.seqid])) + "\n")
