# Installing deeprok

The model weights (4.6 MB) ship inside the package; nothing is downloaded at run
time.

## Conda (recommended)

```bash
git clone <repo> && cd deeprok
conda env create -f environment.yml      # python 3.12, pytorch 2.14.1, numpy 2.5.4
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

There is nothing to configure. `environment.yml` leaves the PyTorch build
unpinned, so conda installs a CUDA build wherever it can see an NVIDIA driver,
and `--device auto` (the default) then uses the GPU: ~10 s for the network step
on a 5 Mb genome instead of ~2 min. To check:

```bash
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

If that prints `False` on a machine that has a GPU, you have a CPU-only build of
PyTorch; `python -c "import torch; print(torch.version.cuda)"` printing `None`
confirms it. The usual cause is creating the environment where the driver is not
visible — a cluster login node, for instance, since conda decides from what it
sees at install time. Swap the build in place rather than rebuilding:

```bash
conda install -n deeprok 'pytorch=*=cuda*'
```

A CUDA build of PyTorch serves `--device cpu` just as well, so there is never a
reason to keep two environments; conda could not install both builds side by
side in any case, as they are the same package name.

On Apple silicon there are no CUDA builds and none are needed: `--device mps`
runs the network on the Mac GPU (K-12: 32 s instead of 133 s, with identical
calls). `auto` never picks MPS; ask for it explicitly.
