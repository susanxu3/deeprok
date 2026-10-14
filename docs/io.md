# Input, output and errors

## Input

* Nucleotide FASTA, one or more records; plain, gzip, bzip2 or xz (detected from
  the file content, not the extension; also on stdin). The sequence id is the
  first word of the header line and must be unique. Records with no sequence are
  skipped with a warning.
* Upper or lower case (lower case is upper-cased; soft-masking is ignored).
  Windows line ends and spaces inside sequence lines are fine. IUPAC ambiguity
  codes (N R Y S W K M B D H V) are read as N. There is no hard rule against a
  gene spanning a run of N (Prodigal's `-m`); the network decides, and in a test
  a 5-kb all-N contig produced no calls.
* Bacterial or archaeal genomes, translation table 11. Genomes using table 4
  (*Mycoplasma*, *Spiroplasma*: TGA = Trp) are not supported yet.

## GFF3 (`-o`)

```
##gff-version 3
##sequence-region NC_000913.3 1 4641652
# deeprok version: 0.1.0
# model: prok666_keephyp (sha256 e608f60a1965)
# mode: single
# translation table: 11
# max gene length: 125000 nt
# network window: 49995 nt
# command: deeprok -i genome.fna -o genes.gff3
NC_000913.3  deeprok  CDS  337  2799  0.996  +  0  ID=NC_000913.3_1;start_codon=ATG;stop_codon=TGA;conf=0.996;p_start=1.000;p_stop=1.000
```

One `CDS` line per gene, sorted by sequence (input order) then position.

| Column | Content |
|---|---|
| 1 seqid | FASTA sequence id (GFF3-escaped) |
| 2 source | `deeprok` |
| 3 type | `CDS` |
| 4–5 start, end | 1-based, inclusive; `end − start + 1` is a multiple of 3; the stop codon is included |
| 6 score | `conf`, the gene's mean coding probability (0–1) |
| 7 strand | `+` / `-` |
| 8 phase | `0` (every CDS is complete) |
| 9 attributes | below |

| Attribute | Meaning |
|---|---|
| `ID` | `<seqid>_<n>`, n = 1, 2, … within each sequence |
| `start_codon`, `stop_codon` | codons as read on the gene's strand (ATG/GTG/TTG; TAA/TAG/TGA) |
| `conf` | mean over the gene body of the network's coding probability on the gene's strand. On the four curated benchmark genomes, 99 % of calls with `conf` > 0.9 had their stop codon in the reference annotation, vs 35 % of calls with `conf` < 0.5 |
| `p_start` | network probability of a start codon at this start |
| `p_stop` | network probability of a stop codon at this stop |

Only complete genes (start codon to stop codon inside the sequence) are
reported; genes running off a contig end are not called. RNA genes are not
predicted.

## Protein FASTA (`-a`) and CDS FASTA (`-d`)

```
>NC_000913.3_1 # 337 # 2799 # 1 # seqid=NC_000913.3;start_codon=ATG;conf=0.996
MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAMIEKTISGQDA
```

Header: `ID # start # end # strand(1/-1) # attributes`, the same IDs and
coordinates as the GFF3 (the layout Prodigal uses). Proteins are translated with
table 11; the first codon is always written as `M` (GTG and TTG starts initiate
with methionine); the terminal stop is not written. `-d` writes the CDS on its
own strand, stop codon included. Lines are wrapped at 60 characters.

## Intermediates (`--keep-intermediates DIR`)

| File | Content |
|---|---|
| `nn_probs/contigNNNN.npy` + `nn_probs_manifest.json` | per-base network probabilities, float32 `(L, 11)`: 0 intergenic · 1–3 coding frames + · 4–6 coding frames − · 7/8 start +/− · 9/10 stop +/− |
| `pass1.gff3` | first-pass calls (= `meta` mode output) |
| `start_ranker.pt` | the genome-specific start ranker (PyTorch state dict) |

## Errors

Input is checked before the network runs, and every problem ends with a one-line
`deeprok: …` message on stderr and a non-zero exit code:

| Problem | Message (abridged) |
|---|---|
| truncated `.gz` / `.bz2` / `.xz` (incomplete download) | `gzip stream ends early; the file is truncated` |
| corrupt compressed data | `cannot read gzip data (…); the file is corrupt or truncated` |
| FASTQ, GFF, GenBank, or sequence before the first `>` | `line 1: not a FASTA file (looks like FASTQ; …)` |
| protein FASTA | `invalid character(s) 'E', 'L' … (amino-acid letters: this looks like a protein FASTA)` |
| GenBank/EMBL-style numbered sequence lines | `invalid character(s) '1' (digits in the sequence …)` |
| other non-IUPAC characters (`-`, `U`, `*`, non-ASCII) | `line N, sequence 'id': invalid character(s) …` |
| empty file, headers only, empty or duplicate ids | `no sequences found` / `duplicate sequence id` |
| missing / unreadable input; output directory missing or not writable; `-o` = input | named in the message |
| `--device cuda` without a GPU | `no CUDA GPU available …; use --device cpu` |

A plain-text FASTA cut short cannot be told from a complete one; deeprok warns
when the file does not end with a newline, the usual trace of an interrupted
copy. Output files are written to a temporary file and renamed when complete, so
a failed or interrupted run (Ctrl-C exits with 130) never leaves a truncated
GFF/FASTA and does not touch an existing one.

Exit codes: `0` success · `1` input, output or runtime error · `2` invalid
command-line options · `130` interrupted.
