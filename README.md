# Mellon

Mellon is a client/server application to easily interface with ML tools with a focus on [Diffusers](https://github.com/huggingface/diffusers).

> [!CAUTION]
> Mellon is in an early development stage and it is not ready for production. DO NOT USE IT unless you know what you are doing.


## Installation

```bash
git clone https://github.com/cubiq/Mellon.git
cd Mellon
python -m venv venv --prompt=Mellon
source venv/bin/activate
pip install --upgrade pip
pip install wheel
```

Install [pytorch](https://pytorch.org/get-started/locally/) for your platform. Generally with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Finally install the dependencies.

```bash
pip install -r requirements.txt
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

### Quantization

If you want to use quantization you need to also install the followings.

**[BitsandBytes](https://github.com/bitsandbytes-foundation/bitsandbytes)**
```bash
pip install bitsandbytes
```

For **[Nunchaku](https://github.com/nunchaku-tech/nunchaku)** follow the installation [instructions](https://nunchaku.tech/docs/nunchaku/installation/installation.html) for your platform.