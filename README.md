# deeprok - a neural gene finder for bacterial and archaeal genomes.

```bash
deeprok -i genome.fna -o genes.gff3 
```

## Introduction

deeprok predicts protein-coding genes in bacterial and archaeal genomes
*ab initio* — from sequence alone, with no reference annotation and no homology
search. deeprok is trained on 666 diverse family representatives
spanning the bacterial and archaeal tree.

Use deeprok to annotate new isolate assemblies, complete or draft. Compared to
other gene prediction methods, deeprok can
output fewer false positives on diverse prokaryotic clades through a two-pass neural approach. 
A 4.6 Mb genome takes ~6 min on four laptop CPU threads,
under 5 with the Mac GPU, and ~10 s for the network step on a CUDA GPU.

## How it works

A neural network (CNN + BiLSTM) reads the genome and predicts, for every base,
whether it is protein-coding and in which reading frame, and whether it is part
of a start or a stop codon. deeprok then enumerates every open reading frame in
the sequence and scores each one from those per-base predictions, using an HMM
that also scores the gaps and overlaps between neighbouring ORFs. The genes it
reports are the highest-scoring combination of non-conflicting ORFs across the
whole sequence, rather than each ORF judged on its own.

Picking the right start codon is the hardest part, so a second pass handles it
separately. deeprok takes the genes it was most confident about in the first
pass and uses them as training examples for a second, smaller CNN, which learns
what a real translation start looks like *in this particular genome* — the
sequence around it, including its ribosome binding site. That ranker then scores
every alternative ATG, GTG and TTG that shares a stop codon with each gene, and
the genome is decoded a second time using those scores. Because the training
examples come from the genome being annotated, nothing external is needed.

## Install

```bash
conda install -c conda-forge -c bioconda deeprok
```

> **Not published yet** — the bioconda recipe is under review
> ([bioconda-recipes#70340](https://github.com/bioconda/bioconda-recipes/pull/70340)).
> Until it is merged, use the from-source install below and ignore this block.

The model weights (4.6 MB) ship inside the package; nothing is downloaded at run
time.

### From source

```bash
git clone <repo> && cd deeprok
conda env create -f environment.yml
conda activate deeprok
pip install --no-deps .
deeprok --version
```

The same file works everywhere: conda installs a CUDA build of PyTorch where it
sees an NVIDIA driver and a CPU build otherwise.

Lock files, pip-only installs and GPU details are in
**[docs/install.md](docs/install.md)**.

## Usage

```
deeprok -i FASTA [-o GFF3] [-a FAA] [-d FNA] [-p single|meta]
        [--threads N] [--device auto|cpu|cuda|mps] [--batch-size N]
        [--max-gene-len NT] [--window NT] [--keep-intermediates DIR] [-q] [-v]
```

| Option | Meaning | Default |
|---|---|---|
| `-i, --input` | genome FASTA; plain, `.gz` or `.bz2`; `-` = stdin | stdin |
| `-o, --output` | gene predictions, GFF3 | stdout |
| `-a, --proteins` | protein sequences, FASTA | — |
| `-d, --nucleotides` | CDS nucleotide sequences, FASTA | — |
| `-p, --mode` | `single` or `meta` | `single` |
| `--threads` | CPU threads | 4 |
| `--device` | network device; `auto` = CUDA if available, else CPU | `auto` |
| `--batch-size` | network tiles per batch | 32 GPU / 4 CPU |
| `--max-gene-len` | longest gene that can be called, nt | 125000 |
| `--window` | network tile length (nt, multiple of 3, ≥ 9990) | 49995 |
| `--keep-intermediates` | also save base-level probabilities, pass-1 calls, ranker model | — |
| `-q, --quiet` | no progress output | |
| `-v, --version` | print version | |

Progress goes to stderr; stdout carries only the GFF3 (when `-o` is not given),
so `deeprok -i g.fna > g.gff3` and pipes are safe.

**`single`** (the default) decodes the genome, trains a start-site ranker on its
own first-pass genes, then decodes again. **`meta`** stops after the first pass —
for short sequences, lone plasmids, or mixed metagenomic bins. Inputs under 20 kb
fall back to `meta` automatically.

Overlapping genes, `--max-gene-len` and `--window` are explained in
**[docs/options.md](docs/options.md)**. 

## Output

```
NC_000913.3  deeprok  CDS  337  2799  0.996  +  0  ID=NC_000913.3_1;start_codon=ATG;stop_codon=TGA;conf=0.996;p_start=1.000;p_stop=1.000
```

One `CDS` line per gene, 1-based inclusive, stop codon included in the
coordinates, sorted by sequence then position. The score column is `conf`, the
gene's mean coding probability; `p_start` and `p_stop` are the network's
probabilities for the chosen boundaries. Only complete genes are reported, and
RNA genes are not predicted. A header block records the version, model, mode and
options used.

`-a` and `-d` write protein and CDS FASTA with Prodigal-style headers.

Full field reference, the `--keep-intermediates` files, accepted input and every
error message: **[docs/io.md](docs/io.md)**.

## Resources

Full pipeline, `single` mode, *E. coli* K-12 (4.6 Mb):

| Hardware | Time | Peak RAM |
|---|---|---|
| MacBook M3 Pro, CPU, 4 threads | 6.1 min (network 133 s) | 2.1 GB |
| MacBook M3 Pro, `--device mps` | 4.7 min (network 32 s) | 1.7 GB |
| NVIDIA H200, CUDA, batch 32 | network step 10–13 s | — |

The network runs on 50-kb tiles, so its memory does not grow with genome size;
the decoder then dominates, at up to ~5 GB for a 9-Mb genome. On CPU about a
third of the time is the network and half the start-ranker training (CPU on
every device; ~3 min for K-12).



## Reproducibility

The model checkpoint, its training run and the decoding priors are recorded in
`src/deeprok/data/model_card.json` and summarised in every GFF3 header. With the
same inputs and environment the output is deterministic. Across platforms the
first pass is identical; the start ranker's training can move a handful of starts
(on *E. coli* K-12 the macOS arm64 CPU run and the Linux cluster run used for the
paper share 4,233 of ~4,250 calls; 706 vs 705 errors against the reference).



## Citation

TBD.

## License

MIT — see [LICENSE](LICENSE).
