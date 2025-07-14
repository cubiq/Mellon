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

Mellon has been tested on python **3.12**.
