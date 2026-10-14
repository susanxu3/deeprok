# deeprok

A neural gene finder for bacterial and archaeal genomes.

A CNN + BiLSTM network scores every base of the genome (coding frame, start
codon, stop codon, intergenic). A segmental ORF HMM turns those scores into
complete genes. A start-site ranker is then trained on the input genome's own
first-pass calls and used to refine the translation starts — no reference
annotation and no internet connection needed.

```bash
deeprok -i genome.fna -o genes.gff3 -a proteins.faa
```

## Install

```bash
git clone <repo> && cd deeprok
conda env create -f environment.yml
conda activate deeprok
pip install --no-deps .
deeprok --version
```

The model weights (4.6 MB) ship inside the package; nothing is downloaded at run
time. Lock files, the conda recipe, pip-only installs, GPU setup and networks
that block `anaconda.org` are covered in **[docs/install.md](docs/install.md)**.

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
**[docs/options.md](docs/options.md)**. Neither of the two numeric options
normally needs changing.

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

## Coming from Prodigal or GeneMarkS-2

| Prodigal | GeneMarkS-2 | deeprok |
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
`src/deeprok/data/model_card.json` and summarised in every GFF3 header. With the
same inputs and environment the output is deterministic. Across platforms the
first pass is identical; the start ranker's training can move a handful of starts
(on *E. coli* K-12 the macOS arm64 CPU run and the Linux cluster run used for the
paper share 4,233 of ~4,250 calls; 706 vs 705 errors against the reference).

## Development

* `src/deeprok/_core/` is **generated**: verbatim copies of the research code
  (decoder, model, ranker) with package-relative imports. Re-create it with
  `python package/dev/sync_core.py` from the research repository;
  `tests/test_units.py::test_core_in_sync` fails when it is stale.
* `python package/dev/stage_model.py --checkpoint … --priors … --name … --training-run …`
  installs a different checkpoint into `src/deeprok/data/` and rewrites the model card.
* Tests: `pytest` (56 tests, ≈ 2 min on CPU; runs the full pipeline on the bundled
  200-kb *E. coli* slice; `tests/test_errors.py` covers malformed input, outputs,
  options, Ctrl-C and broken pipes).

## Citation

TBD.

## License

MIT — see [LICENSE](LICENSE).
