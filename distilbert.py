import os
import sys
from functools import partial

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
if sys.platform == "darwin" and os.environ.get("PYTORCH_MPS_HIGH_WATERMARK_RATIO") is None:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    set_seed,
)
from datasets import Dataset
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

import argparse
import torch

parser = argparse.ArgumentParser(description="BERT-family NER on EWT (DistilBERT default for speed)")
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Force CPU training (use on Apple Silicon to avoid MPS issues)",
)
parser.add_argument(
    "--mps",
    action="store_true",
    help="Optional compatibility flag; on Apple Silicon, MPS is already used unless --cpu",
)
parser.add_argument(
    "--checkpoint",
    default="distilbert-base-uncased",
    help="HF model id (default: DistilBERT — faster than bert-base; use bert-base-uncased for full BERT)",
)
parser.add_argument(
    "--early-stopping-patience",
    type=int,
    default=2,
    help="Stop if eval F1 does not improve for this many epochs (0 = disabled)",
)
parser.add_argument("--train", default="en_ewt-ud-train.iob2", help="Training IOB2 path")
parser.add_argument("--dev", default="en_ewt-ud-dev.iob2", help="Dev IOB2 path")
parser.add_argument("--out", default="predictions_bert.iob2", help="Dev predictions output IOB2")
parser.add_argument(
    "--skip-test",
    action="store_true",
    help="Do not run prediction on the held-out test file",
)
parser.add_argument(
    "--run-test",
    action="store_true",
    help="Force test predictions even when using split dev/train paths",
)
args = parser.parse_args()

# ── config ────────────────────────────────────────────────────────────────────
MODEL_CHECKPOINT = args.checkpoint
OUTPUT_DIR       = "./results_bert"
MAX_SEQ_LENGTH   = 128

TRAIN_BATCH_SIZE = 8   # used on CPU; overridden on CUDA/MPS when not --cpu
EVAL_BATCH_SIZE  = 8
LEARNING_RATE    = 2e-5
NUM_EPOCHS       = 3
WEIGHT_DECAY     = 0.01

TRAIN_FILE       = args.train
DEV_FILE         = args.dev
TEST_FILE        = "en_ewt-ud-test-masked.iob2"
OUTPUT_FILE      = args.out
TEST_OUTPUT_FILE = "test_predictions_bert.iob2"

# Split experiments: skip masked test unless user explicitly wants it.
SKIP_TEST_PREDICT = (
    args.skip_test
    or ("splits/" in args.train.replace("\\", "/"))
    or ("splits/" in args.dev.replace("\\", "/"))
) and not args.run_test

# label set — EWT only uses these 7 labels
LABEL_LIST = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
LABEL2ID   = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL   = {i: l for i, l in enumerate(LABEL_LIST)}


def ewt_tokenize_and_align(examples, tokenizer, max_length):
    """Module-level for HF dataset caching / multiprocessing compatibility."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_length,
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids      = tokenized_inputs.word_ids(batch_index=i)
        prev_word_idx = None
        label_ids     = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != prev_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            prev_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def read_ewt(path):
    """
    Reads an EWT .iob2 file and returns a list of dicts:
        {"tokens": [...], "ner_tags": [...]}

    Tab-separated rows: parts[0] = 1-based word index, parts[1] = token,
    parts[2] = IOB2 label. Lines starting with # are comments; empty lines
    are sentence boundaries.
    """
    sentences = []
    tokens, tags = [], []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#"):
                continue
            if line.strip() == "":
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue
            parts = line.split("\t")
            token = parts[1]
            label = parts[2]
            tag_id = LABEL2ID.get(label, LABEL2ID["O"])
            tokens.append(token)
            tags.append(tag_id)

    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})

    return sentences


def write_iob2_with_predictions(
    source_path, sentences_data, prediction_logits, output_path, tokenizer, label_list
):
    """
    Read source IOB2 (comments and token lines), replace the NER column
    (index 2) with first-subword predictions.
    """
    pred_ids = np.argmax(prediction_logits, axis=2)
    n_sent = len(sentences_data)

    parent = os.path.dirname(os.path.abspath(output_path))
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(source_path, encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        sent_idx = 0
        for line in f_in:
            line = line.rstrip("\n")
            if line.startswith("#"):
                f_out.write(line + "\n")
                continue
            if line.strip() == "":
                f_out.write("\n")
                sent_idx += 1
                continue
            parts = line.split("\t")
            if sent_idx >= n_sent:
                raise ValueError(
                    f"More token lines in {source_path} than sentences ({n_sent})"
                )
            sentence_preds  = pred_ids[sent_idx]
            tokens_in_sent  = sentences_data[sent_idx]["tokens"]
            encoding        = tokenizer(
                tokens_in_sent,
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                is_split_into_words=True,
            )
            word_ids = encoding.word_ids()
            seen, word_pred_map = set(), {}
            for pos, wid in enumerate(word_ids):
                if wid is not None and wid not in seen:
                    seen.add(wid)
                    word_pred_map[wid] = label_list[sentence_preds[pos]]
            word_pos = int(parts[0]) - 1
            parts[2] = word_pred_map.get(word_pos, "O")
            f_out.write("\t".join(parts) + "\n")


# ── metrics ───────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)
    true_predictions, true_labels = [], []
    for pred, lab in zip(predictions, labels):
        curr_preds, curr_labels = [], []
        for p, l in zip(pred, lab):
            if l != -100:
                curr_preds.append(LABEL_LIST[p])
                curr_labels.append(LABEL_LIST[l])
        true_predictions.append(curr_preds)
        true_labels.append(curr_labels)
    return {
        "precision": precision_score(true_labels, true_predictions),
        "recall":    recall_score(true_labels, true_predictions),
        "f1":        f1_score(true_labels, true_predictions),
    }


def main():
    print("SCRIPT STARTED")
    use_cuda = torch.cuda.is_available() and not args.cpu
    use_mps  = torch.backends.mps.is_available() and not args.cpu
    if args.cpu:
        print("Device: CPU (--cpu)")
    elif use_cuda:
        print("Device: CUDA")
    elif use_mps:
        print("Device: MPS (Apple GPU)")
    else:
        print("Device: CPU")
    set_seed(42)

    print("Loading EWT data...")
    train_data = read_ewt(TRAIN_FILE)
    dev_data   = read_ewt(DEV_FILE)

    train_dataset = Dataset.from_list(train_data)
    dev_dataset   = Dataset.from_list(dev_data)

    print(f"  train sentences : {len(train_dataset)}")
    print(f"  dev   sentences : {len(dev_dataset)}")

    print(f"Checkpoint: {MODEL_CHECKPOINT}")
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

    tok_fn = partial(
        ewt_tokenize_and_align,
        tokenizer=tokenizer,
        max_length=MAX_SEQ_LENGTH,
    )

    print("Tokenizing...")
    remove_cols     = ["tokens", "ner_tags"]
    tokenized_train = train_dataset.map(
        tok_fn, batched=True, remove_columns=remove_cols
    )
    tokenized_dev   = dev_dataset.map(
        tok_fn, batched=True, remove_columns=remove_cols
    )
    tokenized_test = None
    if not SKIP_TEST_PREDICT:
        test_data = read_ewt(TEST_FILE)
        test_dataset = Dataset.from_list(test_data)
        print(f"  test  sentences : {len(test_dataset)}")
        tokenized_test = test_dataset.map(
            tok_fn, batched=True, remove_columns=remove_cols
        )
    print("Tokenization done")

    train_bs, eval_bs = TRAIN_BATCH_SIZE, EVAL_BATCH_SIZE
    if use_cuda:
        train_bs, eval_bs = 24, 32
    elif use_mps:
        train_bs, eval_bs = 12, 16

    use_bf16 = bool(use_cuda and torch.cuda.is_bf16_supported())
    use_fp16 = bool(use_cuda and not use_bf16)
    load_dtype = (
        torch.bfloat16 if use_bf16 else torch.float16 if use_fp16 else torch.float32
    )

    print("Loading model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        torch_dtype=load_dtype,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    callbacks = []
    if args.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )

    targs_kw = dict(
        output_dir=OUTPUT_DIR,
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=train_bs,
        per_device_eval_batch_size=eval_bs,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        save_strategy="epoch" if callbacks else "no",
        load_best_model_at_end=bool(callbacks),
        metric_for_best_model="eval_f1" if callbacks else None,
        greater_is_better=True,
        report_to="none",
        fp16=use_fp16,
        bf16=use_bf16,
        use_cpu=args.cpu,
        dataloader_pin_memory=use_cuda,
        dataloader_num_workers=0,
    )
    if callbacks:
        targs_kw["save_total_limit"] = 1
    training_args = TrainingArguments(**targs_kw)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    print("Starting training...")
    trainer.train()

    print("\nEvaluating on dev set...")
    dev_predictions, dev_labels, _ = trainer.predict(tokenized_dev)
    results = compute_metrics((dev_predictions, dev_labels))

    print("\n── BERT / DISTILBERT RESULTS (dev) ──")
    for k, v in results.items():
        print(f"  {k:12s}: {v:.4f}")

    print(f"\nSaving dev predictions to {OUTPUT_FILE} ...")
    write_iob2_with_predictions(
        DEV_FILE, dev_data, dev_predictions, OUTPUT_FILE, tokenizer, LABEL_LIST
    )
    print(f"Predictions saved to {OUTPUT_FILE}")
    print("\nTo evaluate run:")
    print(f"  python span_f1.py {DEV_FILE} {OUTPUT_FILE}")

    print("\nTest set has masked NER labels; skipping F1 on test.")
    if SKIP_TEST_PREDICT:
        print("(Skipping test-file prediction for this run; use default train/dev or --run-test to enable.)")
    else:
        test_predictions, _, _ = trainer.predict(tokenized_test)
        print(f"Saving test predictions to {TEST_OUTPUT_FILE} ...")
        write_iob2_with_predictions(
            TEST_FILE, test_data, test_predictions, TEST_OUTPUT_FILE, tokenizer, LABEL_LIST
        )
        print(f"Test predictions saved to {TEST_OUTPUT_FILE}")


if __name__ == "__main__":
    main()