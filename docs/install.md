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

## GPU

Optional, and only the network step uses it — the rest of the pipeline runs on
CPU either way.

**`environment.yml` installs a CPU-only build of PyTorch, so `--device cuda`
fails in that environment.** For an NVIDIA GPU, build the environment from
`environment-nvidia.yml` instead (Linux only — conda-forge has no CUDA builds for
macOS):

```bash
conda env create -f environment-nvidia.yml
conda activate deeprok-nvidia
pip install --no-deps .
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

`--device auto` (the default) then uses the GPU, and a 5 Mb genome takes ~10 s
for the network step instead of ~2 min. If `torch.cuda.is_available()` prints
`False`, you are still in a CPU build; `python -c "import torch; print(torch.version.cuda)"`
printing `None` confirms it.

The two files are a choice per machine, not per run: a CUDA build of PyTorch
includes CPU support, so `deeprok-nvidia` also serves `--device cpu` and there is
no reason to keep both environments on the same machine. Conda cannot install
both builds into one environment in any case — they are the same package name,
so the solver treats them as a conflict.

Already built the CPU environment and only now moved to a GPU node? Swap the
build in place instead of creating a second environment:

```bash
conda install -n deeprok 'pytorch=2.14.1=cuda*'
```

On Apple silicon, `environment.yml` is the right file: `--device mps` runs the
network on the Mac GPU (K-12: 32 s instead of 133 s, with identical calls).
`auto` never picks MPS; ask for it explicitly.
