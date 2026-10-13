# tprok

*(working name)* — a neural gene finder for bacterial and archaeal genomes.

A CNN + BiLSTM network scores every base of the genome (coding frame, start
codon, stop codon, intergenic). A segmental ORF HMM turns those scores into
complete genes. A start-site ranker is then trained on the input genome's own
first-pass calls and used to refine the translation starts — no reference
annotation and no internet connection needed.

```
tprok -i genome.fna -o genes.gff3 -a proteins.faa
```

## Install

Conda (recommended; CPU):

```bash
git clone <repo> && cd tprok
conda env create -f environment.yml      # python 3.12, pytorch 2.14.1 (cpu), numpy 2.5.4
conda activate tprok
pip install --no-deps .
tprok --version
```

The model weights (4.6 MB) ship inside the package; nothing is downloaded at run
time. `environment.yml` pins the exact versions the release was tested with;
`conda/lock/<platform>.txt` (`linux-64`, `osx-arm64`) is an explicit lock file for a
bit-for-bit identical environment (`conda create -n tprok --file conda/lock/linux-64.txt`).
A conda recipe is in `conda/recipe/` (`conda build conda/recipe -c conda-forge`
builds a noarch package whose tests run tprok on the bundled genome slice).

pip only: `pip install .` (pulls `torch` and `numpy` from PyPI; on Linux, install the
CPU wheel of torch first if you don't want the ~2 GB CUDA build).

**GPU**: optional; only the network step uses it, the rest of the pipeline runs on
CPU either way. With a CUDA build of PyTorch, `--device auto` (the default) uses
the GPU; a 5 Mb genome then takes ~10 s for that step instead of ~2 min. On Apple
silicon, `--device mps` runs it on the Mac GPU (K-12: 32 s instead of 133 s, with
identical calls). `auto` never picks MPS; ask for it explicitly.

## Usage

```
tprok -i FASTA [-o GFF3] [-a FAA] [-d FNA] [-p single|meta]
      [--threads N] [--device auto|cpu|cuda|mps] [--batch-size N] [--window NT]
      [--keep-intermediates DIR] [-q] [-v]
```

| Option | Meaning | Default |
|---|---|---|
| `-i, --input` | genome FASTA; plain, `.gz` or `.bz2`; `-` = stdin | stdin |
| `-o, --output` | gene predictions, GFF3 | stdout |
| `-a, --proteins` | protein sequences, FASTA | — |
| `-d, --nucleotides` | CDS nucleotide sequences, FASTA | — |
| `-p, --mode` | `single` or `meta` (below) | `single` |
| `--threads` | CPU threads | 4 |
| `--device` | network device; `auto` = CUDA if available, else CPU | `auto` |
| `--batch-size` | network tiles per batch | 32 GPU / 4 CPU |
| `--window` | network tile length (nt, multiple of 3, ≥ 9990) | 49995 |
| `--keep-intermediates` | also save base-level probabilities, pass-1 calls, ranker model | — |
| `-q, --quiet` | no progress output | |
| `-v, --version` | print version | |

Progress goes to stderr; stdout carries only the GFF3 (when `-o` is not given), so
`tprok -i g.fna > g.gff3` and pipes are safe.

### Modes

* **single** (default) — one genome, complete or draft. Two decoding passes: the
  second uses a start-site ranker trained on this genome's first-pass genes,
  pooled over all its contigs. Needs ≥ 20 kb of sequence and ≥ 100 confident
  first-pass genes; below that tprok falls back to `meta` automatically and says
  so on stderr and in the GFF header.
* **meta** — first pass only. For short sequences, plasmids on their own, or
  mixed metagenomic bins where one ranker per input would be wrong.

## Input

* Nucleotide FASTA, one or more records; plain, gzip, bzip2 or xz (detected from
  the file content, not the extension; also on stdin). The sequence id is the
  first word of the header line and must be unique. Records with no sequence are
  skipped with a warning.
* Upper or lower case (lower case is upper-cased; soft-masking is ignored).
  Windows line ends and spaces inside sequence lines are fine. IUPAC ambiguity
  codes (N R Y S W K M B D H V) are read as N. There is no hard rule against a
  gene spanning a run of N (Prodigal's `-m`); the network decides, and in a test a
  5-kb all-N contig produced no calls.
* Bacterial or archaeal genomes, translation table 11. Genomes using table 4
  (*Mycoplasma*, *Spiroplasma*: TGA = Trp) are not supported yet.

### Errors

Input is checked before the network runs, and every problem ends with a
one-line `tprok: …` message on stderr and a non-zero exit code:

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

A plain-text FASTA cut short cannot be told from a complete one; tprok warns
when the file does not end with a newline, the usual trace of an interrupted
copy. Output files are written to a temporary file and renamed when complete,
so a failed or interrupted run (Ctrl-C exits with 130) never leaves a truncated
GFF/FASTA and does not touch an existing one.

Exit codes: `0` success · `1` input, output or runtime error · `2` invalid
command-line options · `130` interrupted.
## Output

### GFF3 (`-o`)

```
##gff-version 3
##sequence-region NC_000913.3 1 4641652
# tprok version: 0.1.0
# model: prok666_keephyp (sha256 e608f60a1965)
# mode: single
# translation table: 11
# command: tprok -i genome.fna -o genes.gff3
NC_000913.3  tprok  CDS  337  2799  0.996  +  0  ID=NC_000913.3_1;start_codon=ATG;stop_codon=TGA;conf=0.996;p_start=1.000;p_stop=1.000
```

One `CDS` line per gene, sorted by sequence (input order) then position.

| Column | Content |
|---|---|
| 1 seqid | FASTA sequence id (GFF3-escaped) |
| 2 source | `tprok` |
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

Only complete genes (start codon to stop codon inside the sequence) are reported;
genes running off a contig end are not called. RNA genes are not predicted.

### Protein FASTA (`-a`) and CDS FASTA (`-d`)

```
>NC_000913.3_1 # 337 # 2799 # 1 # seqid=NC_000913.3;start_codon=ATG;conf=0.996
MRVLKFGGTSVANAERFLRVADILESNARQGQVATVLSAPAKITNHLVAMIEKTISGQDA
```

Header: `ID # start # end # strand(1/-1) # attributes`, the same IDs and
coordinates as the GFF3 (the layout Prodigal uses). Proteins are translated with
table 11; the first codon is always written as `M` (GTG and TTG starts initiate
with methionine); the terminal stop is not written. `-d` writes the CDS on its
own strand, stop codon included. Lines are wrapped at 60 characters.

### Intermediates (`--keep-intermediates DIR`)

| File | Content |
|---|---|
| `nn_probs/contigNNNN.npy` + `nn_probs_manifest.json` | per-base network probabilities, float32 `(L, 11)`: 0 intergenic · 1–3 coding frames + · 4–6 coding frames − · 7/8 start +/− · 9/10 stop +/− |
| `pass1.gff3` | first-pass calls (= `meta` mode output) |
| `start_ranker.pt` | the genome-specific start ranker (PyTorch state dict) |

## Resources

Measured on the full pipeline (single mode):

| Genome | Hardware | Time | Peak RAM |
|---|---|---|---|
| *E. coli* K-12, 4.6 Mb | MacBook M3 Pro, CPU, 4 threads | 6.1 min (network 133 s) | 2.1 GB |
| *E. coli* K-12, 4.6 Mb | MacBook M3 Pro, `--device mps` | 4.7 min (network 32 s) | 1.7 GB |
| *E. coli* K-12, 4.6 Mb | NVIDIA H200, CUDA, batch 32 (network step only, research code, job 29688) | network 10–13 s | — |

The network runs on 50-kb tiles, so its memory does not grow with genome size
(whole-chromosome inference needed ~3.5 GB per Mb); the decoder then dominates,
at up to ~5 GB for a 9-Mb genome. On CPU about a third of the time is the network
and half the start-ranker training (CPU on every device; ~3 min for K-12).

## Coming from Prodigal or GeneMarkS-2

| Prodigal | GeneMarkS-2 | tprok |
|---|---|---|
| `-i genome.fna` | `--seq genome.fna` | `-i genome.fna` |
| `-o out.gff -f gff` | `--output out.gff --format gff` | `-o out.gff3` (GFF3 is the only format) |
| `-a proteins.faa` | `--faa proteins.faa` | `-a proteins.faa` |
| `-d genes.fna` | `--fnn genes.fna` | `-d genes.fna` |
| `-p single` / `-p meta` | — | `-p single` / `-p meta` |
| `-q` | — | `-q` |
| `-g 11` | `--gcode 11` | table 11 only |
| `-c` (closed ends) | — | always closed: only complete genes |
| — | `--genome-type auto` | not needed: one model for bacteria and archaea |

## Reproducibility

The model checkpoint, its training run and the decoding priors are recorded in
`src/tprok/data/model_card.json` and summarised in every GFF3 header. With the
same inputs and environment the output is deterministic. Across platforms the
first pass is identical; the start ranker's training can move a handful of
starts (on *E. coli* K-12 the macOS arm64 CPU run and the Linux cluster run used
for the paper share 4,233 of ~4,250 calls; 706 vs 705 errors against the
reference).

## Development

* `src/tprok/_core/` is **generated**: verbatim copies of the research code
  (decoder, model, ranker) with package-relative imports. Re-create it with
  `python package/dev/sync_core.py` from the research repository;
  `tests/test_units.py::test_core_in_sync` fails when it is stale.
* `python package/dev/stage_model.py --checkpoint … --priors … --name … --training-run …`
  installs a different checkpoint into `src/tprok/data/` and rewrites the model card.
* Tests: `pytest` (54 tests, ≈ 1 min on CPU; runs the full pipeline on the bundled
  200-kb *E. coli* slice; `tests/test_errors.py` covers malformed input, outputs,
  options, Ctrl-C and broken pipes).

## Citation

TBD.

## License

TBD.
