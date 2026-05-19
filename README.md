# NER experiments on English EWT-style IOB2 data

Named entity recognition codebase for the NLP project, comparing four models across controlled distribution-shift splits of the English Web Treebank (EWT).

| Script | Model |
| --- | --- |
| `baseline.py` | BiLSTM-CRF (trained from scratch) |
| `distilbert.py` | DistilBERT token classifier (Hugging Face fine-tuning) |
| `llama32_zeroshot_ner.py` | Llama 3.2 (1B) instruct, zero-shot labelling |
| `llama32.py` | Llama 3.2 fine-tuned NER with LoRA |

Supporting utilities: `span_f1.py` (span-level evaluation), `split_data.py` / `analysis.py` (dataset splits), `generate_figures.py` / `generate_pertype_figures.py` (PDFs under `figures/`).

## Requirements

Python 3.9+, tested on macOS with Apple Silicon (MPS). GPU optional: CUDA or Apple MPS speeds up DistilBERT and Llama; BiLSTM uses MPS/CUDA when available.

## Setup

```bash
cd /path/to/project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

If `pip install torch` is insufficient for your hardware, follow [pytorch.org](https://pytorch.org/get-started/locally/) first, then install the rest from `requirements.txt`. Also run `python -m spacy download en_core_web_sm` for Split 3 (context-shift) to work correctly.

## Data layout

Standard CoNLL-style IOB2 files: `en_ewt-ud-train.iob2`, `en_ewt-ud-dev.iob2`, `en_ewt-ud-test-masked.iob2`

Stress splits (train/dev pairs): `splits/<split_name>/train.iob2`, `dev.iob2`

Place files in this directory or pass explicit `--train` / `--dev` paths.

## Running the models

**BiLSTM-CRF**

```bash
python3 baseline.py --train en_ewt-ud-train.iob2 --dev en_ewt-ud-dev.iob2 --out predictions/bilstm_standard.iob2
```

**DistilBERT**

```bash
python3 distilbert.py --mps \
  --train en_ewt-ud-train.iob2 --dev en_ewt-ud-dev.iob2 \
  --out predictions/distilbert_standard.iob2
```

**Llama zero-shot**

```bash
python3 llama32_zeroshot_ner.py --mps \
  --dev en_ewt-ud-dev.iob2 \
  --out predictions_llama32_zeroshot.iob2
```

**Llama fine-tuned (LoRA)**

```bash
python3 llama32.py --mps --lora \
  --train en_ewt-ud-train.iob2 --dev en_ewt-ud-dev.iob2 \
  --out predictions/llama_finetuned_standard.iob2 \
  --epochs 3 --train-batch-size 2 --eval-batch-size 2 --max-seq-length 96
```

To run all four splits at once:

```bash
python3 llama32.py --mps --lora --run-all-splits --include-standard \
  --epochs 3 --train-batch-size 2 --eval-batch-size 2 --max-seq-length 96
```

## Evaluation

```bash
python3 span_f1.py GOLD.iob2 PRED.iob2
python3 span_f1.py GOLD.iob2 PRED.iob2 --per-type
```

## Figures

```bash
python3 generate_figures.py
python3 generate_pertype_figures.py
```

PDFs are written to `figures/`.
