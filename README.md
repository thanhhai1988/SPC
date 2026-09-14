
# SPC

**Stable Prototype Classifier (SPC)**

This repository contains the implementation and experiments for Stable Prototype Classifier (SPC).

## Project structure

## Requirements

The code is developed for a GPU environment.

Recommended:

- Python 3.10+
- CUDA-enabled NVIDIA GPU
- Sufficient GPU memory for Qwen3-VL-Embedding-2B

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

## Installation

### 1. Clone this repository

```bash
git clone https://github.com/thanhhai1988/SPC.git
cd SPC
```

### 2. Create a virtual environment

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Qwen3-VL-Embedding

SPC uses Qwen3-VL-Embedding for extracting image and text embeddings for classification

Clone the Qwen3-VL-Embedding repository:

```bash
git clone https://github.com/QwenLM/Qwen3-VL-Embedding.git
```
### Note: Delete @torch.no_grad in Qwen Embedder class to train classifier
The Qwen3-VL-Embedding code is expected to be available locally so that imports such as

```python
from Qwen3_VL_Embedding.src.models.qwen3_vl_embedding import Qwen3VLEmbedder
```

can be resolved.

### Download the pretrained model

Install/enable the Hugging Face CLI and download the model:

```bash
hf download Qwen/Qwen3-VL-Embedding-2B \
    --local-dir ./models/Qwen3-VL-Embedding-2B
```

The model files should not be committed to GitHub.


## Qwen preparation notebook

The notebook

```text
notebooks/Qwen_prepare.ipynb
```
contains the preparation and prototype-related code used with Qwen3-VL-Embedding.

It includes:

- installing the required packages;
- cloning Qwen3-VL-Embedding;
- downloading Qwen3-VL-Embedding-2B;
- loading the Qwen embedding model;

## Running the project

After completing the environment and model setup, run the appropriate experiment

For example:

```bash
python spc_main.py
```

The exact command and arguments depend on the experiment configuration implemented in the repository.

## License

This repository is intended for research purposes.
