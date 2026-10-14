# Modes, overlaps and tuning

## Modes

**single** (default) decodes the genome, trains a start-site ranker on that
genome's own first-pass genes (pooled over all its contigs), and decodes a
second time with the ranker's scores. It needs ≥ 20 kb of sequence and ≥ 100
confident first-pass genes; below that deeprok falls back to `meta`
automatically and says so on stderr and in the GFF3 header.

**meta** runs the first pass only. Use it for short sequences, plasmids on their
own, or mixed metagenomic bins where one ranker per input would be wrong.

## Overlapping genes

Overlapping genes are allowed, and the decoder prefers the overlap when the
sequence supports it: every same-strand overlap from 1 nt up to the limit below
is an explicit state with its own prior, so an overlapping pair competes on
score against the non-overlapping alternative rather than being filtered out
afterwards. The limit differs between the two passes, and only the second one
decides the output:

| | overlap limit (same strand and opposite strand) |
|---|---|
| pass 1 (no ranker; `meta` mode output) | 50 nt |
| pass 2 (with the start ranker; the reported genes in `single` mode) | **250 nt** |

Each pass prints what it dropped:

```
  pass 1   decoding, no ranker; overlaps up to 50 nt
    overlap filter: +2063/-2170 -> +2059/-2166  (dropped: same-strand >50 nt 0,
    opposite-strand start collision 1, opposite-strand >50 nt 7)
```

A gene counted as `>50 nt` in pass 1 is usually called again in pass 2, where
the limit is five times higher. Read these counts as decode diagnostics, not as
the final gene set.

One overlap is rejected at any depth: **start collision**, where two genes on
opposite strands overlap and their start codons do as well. A gene's start codon
occupies the first three bases of the gene as read on its own strand, so this is
the back-to-back arrangement in which a `+` gene and a `−` gene begin within two
bases of each other and are transcribed away from each other. Both cannot
initiate translation there — the two ribosome binding sites would need the same
nucleotides — so the shorter gene is dropped. This rule has no threshold.

After pass 2, a call is also dropped if nearly all of its own length lies inside
a single, strictly longer gene on the opposite strand. These embedded antisense
ORFs are the dominant false-positive class when the decoder is tuned to recover
small genes.

Neither overlap limit is exposed on the command line; both are properties of the
published pipeline (`pipeline.py`, `PASS1`/`PASS2`).

## `--max-gene-len` (default 125000)

A hard ceiling in the decoder, so a gene longer than this cannot be emitted at
all, and a real one is instead reported with its stop codon correct and its
start cut back to the limit.

Across 944 GTDB genomes (3.0 M CDS) the longest CDS is 119 kb, and 147 CDS in 89
genomes (9 % of genomes) exceed 20 kb — giant surface, adhesin and NRPS/PKS
proteins. The default covers all of them. Raising it is close to free: on the 4
curated benchmark genomes 20 kb → 150 kb left every call bit-identical, and on 3
held-out genomes holding 12 CDS above 20 kb the call count and runtime were
unchanged while 10 of the 12 moved from truncated to exactly right. Lowering it
only truncates genes, so do that only to reproduce an older run (the published
runs used 20000, which on their genomes gives identical output).

## `--window` (default 49995)

The tile length the network scores at once. Memory is flat in genome size
because of the tiling, so a larger window costs RAM without buying accuracy: 50
kb tiles already reproduce whole-chromosome inference to within 0.008 in
probability (no base differs by more than 0.05), while 10 kb tiles differ by up
to 0.85 at tile seams. Below ~10 kb, accuracy degrades.

Both options are recorded in the GFF3 header, and neither normally needs
changing.
