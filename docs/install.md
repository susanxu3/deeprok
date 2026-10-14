# Installing deeprok

The model weights (4.6 MB) ship inside the package; nothing is downloaded at run
time.

## Conda (recommended)

```bash
git clone <repo> && cd deeprok
conda env create -f environment.yml      # python 3.12, pytorch 2.14.1 (cpu), numpy 2.5.4
conda activate deeprok
pip install --no-deps .
deeprok --version
```

`environment.yml` pins the versions the release was tested with. For a
bit-for-bit identical environment use the explicit lock files instead
(`linux-64`, `osx-arm64`):

```bash
conda create -n deeprok --file conda/lock/linux-64.txt
```

A conda recipe is in `conda/recipe/`; `conda build conda/recipe -c conda-forge`
builds a noarch package whose tests run deeprok on the bundled genome slice.

## pip only

```bash
pip install .
```

This pulls `torch` and `numpy` from PyPI. On Linux, install the CPU wheel of
torch first if you don't want the ~2 GB CUDA build.

## Networks that block anaconda.org

Mass General Brigham blocks it for licensing reasons (MGB KB0043690). Point
conda at the mirror:

```bash
conda config --set channel_alias https://anaconda.mgb.org/        # for environment.yml
sed -i.bak 's#conda.anaconda.org#anaconda.mgb.org#' conda/lock/*.txt   # for the lock files
```

`channel_alias` does not rewrite the lock files, because those carry full URLs —
hence the second line (written so it works with both GNU and BSD/macOS sed). The
mirror's certificate is valid, so `ssl_verify: false` (which that article
suggests) is not needed. pip and PyPI are unaffected.

## GPU

Optional, and only the network step uses it — the rest of the pipeline runs on
CPU either way.

| | |
|---|---|
| CUDA build of PyTorch | `--device auto` (the default) uses the GPU; a 5 Mb genome takes ~10 s for the network step instead of ~2 min |
| Apple silicon | `--device mps` runs it on the Mac GPU (K-12: 32 s instead of 133 s, with identical calls). `auto` never picks MPS; ask for it explicitly |
