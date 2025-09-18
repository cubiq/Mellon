# Mellon

Mellon is a client/server application to easily interface with ML tools with a focus on [Diffusers](https://github.com/huggingface/diffusers).

> [!CAUTION]
> Mellon is in an early development stage and it is not ready for production. DO NOT USE IT unless you know what you are doing.


## Installation

```bash
git clone https://github.com/cubiq/Mellon.git
cd Mellon
python -m venv .venv --prompt=Mellon
source .venv/bin/activate
pip install -U pip
pip install -U wheel
pip install -U setuptools
```

Install [pytorch](https://pytorch.org/get-started/locally/) for your platform. Generally with:

```bash
pip install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Finally install the dependencies.

```bash
pip install -U -r requirements.txt
```

Run the application with `python main.py` or --on Linux-- with the script `./run.sh`.

To install [Flash Attention](https://github.com/Dao-AILab/flash-attention) at the moment we have to use v`2.8.0.post2`:

```bash
pip install flash-attn==2.8.0.post2 --no-build-isolation
```

Optionally you can also install [SageAttention](https://github.com/thu-ml/SageAttention) but you need v2 and it has to be compiled from source. Be sure to have the environment activated, clone the official repository somewhere outside of Mellon directory and compile it like so (it can take a while):

```bash
cd ~/some/directory
git clone https://github.com/thu-ml/SageAttention.git
cd SageAttention 
export EXT_PARALLEL=4 NVCC_APPEND_FLAGS="--threads 8" MAX_JOBS=32 # parallel compiling
pip install -e .
```

Remember you need `python3.x-devel` installed on your system to compile from source.

**Note:** Mellon has been tested on python **3.12**.

### Additional functionality

Additional libraries can be installed with:

```bash
pip install -U -r requirements_extras.txt
```

You are very likely going to need them but in the spirit of keeping the core dependencies to the bare minimum I'm putting extra features in an optional requirement file. You can also check `requirements_extras.txt` and cherry pick what you need.

### Quantization

Quantization often requires additional packages. Check the `requirements_quant.txt` file for a list of available quantizations. If you want to install them all execute:

```bash
pip install -U -r requirements_quant.txt
```

For **[Nunchaku](https://github.com/nunchaku-tech/nunchaku)** follow the installation [instructions](https://nunchaku.tech/docs/nunchaku/installation/installation.html) for your platform.

## Upgrading

To upgrade Mellon just pull the repository:

```
cd Mellon
git pull
```

To upgrade the python libraries, activate the environment and follow the pip install order as per the installation

```
source .venv/bin/activate
pip install -U pip
pip install -U wheel
pip install -U setuptools
...
...
```