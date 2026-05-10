# NER experiments on English EWT-style IOB2 data

Course/research codebase for **named entity recognition** comparing:

| Script | Model |
|--------|--------|
| `baseline.py` | **BiLSTM-CRF** (train from scratch) |
| `distilbert.py` | **DistilBERT** token classifier (Hugging Face fine-tuning) |
| `llama32_zeroshot_ner.py` | **Llama 3.2 (1B) instruct**, zero-shot labeling |
| `llama32.py` | **Llama 3.2** fine-tuned NER (LoRA recommended on laptop GPUs) |

Supporting utilities: `span_f1.py` (span-level evaluation), `split_data.py` / `analysis.py` (dataset splits), figure scripts under `figures/`.

## Requirements

- **Python 3.9+** (tested on macOS with Apple Silicon + MPS).
- GPU optional: **CUDA** or **Apple MPS** speeds up DistilBERT and Llama; BiLSTM will use MPS/CUDA when available.

### Setup

```bash
cd /path/to/project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch: if `pip install torch` is not enough for your machine, follow [pytorch.org](https://pytorch.org/get-started/locally/) and then install the rest from `requirements.txt`.

## Data layout

- Standard CoNLL-style **IOB2** files (tab-separated; column 3 = BIO tag):  
  `en_ewt-ud-train.iob2`, `en_ewt-ud-dev.iob2`, `en_ewt-ud-test-masked.iob2`
- **Stress splits** (train/dev pairs): `splits/<split_name>/train.iob2`, `dev.iob2`

Place files in this directory or pass explicit `--train` / `--dev` paths.

## Running the four models

### 1. BiLSTM-CRF (`baseline.py`)

Uses **MPS → CUDA → CPU** automatically when `--cpu` is not set.

```bash
python3 baseline.py --train en_ewt-ud-train.iob2 --dev en_ewt-ud-dev.iob2 --out predictions/bilstm_standard.iob2
```

### 2. DistilBERT (`distilbert.py`)

Default checkpoint is `distilbert-base-uncased`. On Apple Silicon, MPS is used unless `--cpu`.

```bash
python3 distilbert.py --mps \
  --train en_ewt-ud-train.iob2 --dev en_ewt-ud-dev.iob2 \
  --out predictions/distilbert_standard.iob2
```

Use `--checkpoint bert-base-uncased` for full BERT instead of DistilBERT.

### 3. Llama zero-shot (`llama32_zeroshot_ner.py`)

No training; downloads model weights from Hugging Face on first run.

```bash
python3 llama32_zeroshot_ner.py --mps \
  --dev en_ewt-ud-dev.iob2 \
  --out predictions_llama32_zeroshot.iob2
```

Long runs: append output with resume support is documented in the script docstring (`nohup`, `--fresh`).

### 4. Llama fine-tuning (`llama32.py`)

Requires **`peft`** for `--lora` (recommended on Mac). Checkpoints are written under `./results_llama32_<run>/` (see `.gitignore`).

Example (standard split):

```bash
python3 llama32.py --mps --lora \
  --train en_ewt-ud-train.iob2 --dev en_ewt-ud-dev.iob2 \
  --out predictions/llama_finetuned_standard.iob2 \
  --epochs 3 --train-batch-size 2 --eval-batch-size 2 --max-seq-length 96
```

Train/eval all bundled splits:

```bash
python3 llama32.py --mps --lora --run-all-splits --include-standard \
  --epochs 3 --train-batch-size 2 --eval-batch-size 2 --max-seq-length 96
```

## Evaluation (span F1)

```bash
python3 span_f1.py GOLD.iob2 PRED.iob2
python3 span_f1.py GOLD.iob2 PRED.iob2 --per-type
```

## Figures

```bash
python3 generate_figures.py
python3 generate_pertype_figures.py
```

PDFs are written to `figures/` (same folder as this README).

## What to upload to GitHub

- **Do commit:** source `.py` files, `requirements.txt`, `README.md`, `.gitignore`, IOB2 data / `splits/` if allowed by your course policy, small `figures/*.pdf`, representative **prediction** `.iob2` files if small enough.
- **Do not commit:** `results_*` training folders (often **hundreds of MB to GB**), optimizer states, downloaded HF caches (`.cache/`). Recipients can re-run training or you can share checkpoints via cloud storage / Git LFS separately.

## Optional utilities

- `split_data.py`, `analysis.py` — build or analyse experimental splits (see docstrings).
- `conlleval.pl` — legacy CoNLL evaluation script (optional).
